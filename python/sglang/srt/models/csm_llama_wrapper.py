import os
import json
import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.llama import LlamaForCausalLM

logger = logging.getLogger(__name__)

class CsmLlamaWrapper(LlamaForCausalLM):
    """
    Ultimate CSM Wrapper: Self-Driving + Hybrid Embeddings + Logit Masking.
    Robust against CUDA crashes and hallucinations.
    """
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        
        self.backbone_hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # Config Constants
        self.audio_token_id = getattr(config, "audio_token_id", 128002)
        self.audio_eos = getattr(config, "audio_eos_token_id", 128003)
        self.codebook_pad = getattr(config, "codebook_pad_token_id", 2050)

        # Depth Decoder Setup
        self._depth_path = os.environ.get("CSM_DEPTH_DECODER_PATH", None)
        self.depth_decoder = None
        self.num_codebooks = 32
        
        self._depth_dtype = torch.float16
        if hasattr(config, "torch_dtype"):
            self._depth_dtype = config.torch_dtype

        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda")

        # --- Load Audio Embeddings ---
        # Look for audio_embeds.pt in the checkpoint directory
        chk_path = config._name_or_path
        if not os.path.isabs(chk_path) and os.path.exists(chk_path):
             embed_path = os.path.join(chk_path, "audio_embeds.pt")
        else:
             embed_path = "/home/shaan/Projects/csm_decoder/csm-1b-backbone-llama/audio_embeds.pt"

        if os.path.exists(embed_path):
            logger.info(f"Loading CSM audio embeddings from {embed_path}")
            self.audio_embeds = nn.Embedding.from_pretrained(
                torch.load(embed_path, map_location=self.device, weights_only=True),
                freeze=True
            ).to(self.device, dtype=self._depth_dtype)
        else:
            logger.warning(f"CRITICAL: Audio embeddings not found at {embed_path}. Backbone will hallucinate!")
            self.audio_embeds = None

        # --- State Management ---
        self._csm_req_state: dict[int, int] = {}
        self._csm_backbone_h_table: dict[int, torch.Tensor] = {}
        self._csm_depth_past_table: dict[int, Any] = {}
        self._csm_cached_cb0: dict[int, int] = {}
        self._csm_virtual_pos: dict[int, int] = {}

    def _get_hybrid_embeddings(self, input_ids):
        """Use Audio Embeds for codebooks (0-2050), Text Embeds otherwise."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        if self.audio_embeds is not None:
            # Check for audio range
            is_audio = (input_ids >= 0) & (input_ids <= self.codebook_pad)
            if is_audio.any():
                # Safety clamp for embedding lookup
                safe_ids = torch.clamp(input_ids[is_audio], max=self.audio_embeds.num_embeddings - 1)
                inputs_embeds[is_audio] = self.audio_embeds(safe_ids).to(inputs_embeds.dtype)
        return inputs_embeds

    def _apply_audio_mask(self, logits, row_idx):
        """Masks logits in-place: allow only Codebook tokens + Audio EOS."""
        eos_val = logits[row_idx, self.audio_eos].clone()
        logits[row_idx, self.codebook_pad + 1:] = float("-inf")
        logits[row_idx, self.audio_eos] = eos_val

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        
        if forward_batch.forward_mode.is_decode():
            return self._forward_decode(
                input_ids, positions, forward_batch, pp_proxy_tensors, input_embeds
            )
        else:
            return self._forward_prefill(
                input_ids, positions, forward_batch, pp_proxy_tensors, input_embeds
            )

    def _forward_prefill(self, input_ids, positions, forward_batch, pp, input_embeds):
        if input_embeds is None:
            input_embeds = self._get_hybrid_embeddings(input_ids)

        output = super().forward(
            input_ids=input_ids, 
            positions=positions, 
            forward_batch=forward_batch, 
            pp_proxy_tensors=pp, 
            input_embeds=input_embeds
        )

        # Init State Machine & Mask Prefill Logits
        req_indices = forward_batch.req_pool_indices.tolist()
        seq_lens = forward_batch.seq_lens.tolist()
        start_idx = 0
        input_ids_cpu = input_ids.tolist()
        positions_cpu = positions.tolist()
        total_tokens = len(input_ids_cpu)
        
        # View into logits
        logits = output.next_token_logits

        for i, seq_len in enumerate(seq_lens):
            end_idx = start_idx + seq_len
            # Guard against chunked prefill where seq_lens may exceed the current tensor span
            if end_idx <= start_idx or start_idx >= total_tokens:
                start_idx = end_idx
                continue
            last_idx = min(end_idx - 1, total_tokens - 1)
            last_tok = input_ids_cpu[last_idx]
            last_pos = positions_cpu[last_idx]
            req_id = req_indices[i]

            if last_tok == self.audio_token_id:
                # Prompt ended with <|AUDIO|>. Next token MUST be audio.
                self._csm_req_state[req_id] = 0
                self._csm_virtual_pos[req_id] = int(last_pos) + 1
                self._csm_depth_past_table.pop(req_id, None)
                self._csm_cached_cb0.pop(req_id, None)

                # MASK THE FIRST TOKEN TO STOP HALLUCINATION
                if logits is not None:
                    self._apply_audio_mask(logits, i)
            else:
                self._csm_req_state[req_id] = -1
            
            start_idx = end_idx

        return output

    def _forward_decode(self, input_ids, positions, forward_batch, pp, input_embeds):
        B = input_ids.shape[0]
        device = input_ids.device
        req_indices = forward_batch.req_pool_indices.to(torch.int64)
        input_ids_list = input_ids.view(-1).tolist()

        phase_tensor = torch.zeros(B, dtype=torch.int64, device=device) # 0=Backbone, 1=Depth
        
        # 1. Update State
        for i in range(B):
            req_id = int(req_indices[i])
            tok = input_ids_list[i]
            
            step = self._csm_req_state.get(req_id, -1)

            if tok == self.audio_token_id:
                step = 0 # Start
                self._csm_virtual_pos[req_id] = int(positions[i].item())
                self._csm_depth_past_table.pop(req_id, None)
            elif tok == self.audio_eos:
                step = -1 # End
            elif step >= 0:
                # In Audio
                if step == 0:
                    self._csm_cached_cb0[req_id] = tok
                    step = 1
                elif step < self.num_codebooks - 1:
                    step += 1
                else:
                    step = 0
                    self._csm_virtual_pos[req_id] += 1
                    self._csm_depth_past_table.pop(req_id, None)
            
            self._csm_req_state[req_id] = step
            if step > 0:
                phase_tensor[i] = 1 

        backbone_mask = (phase_tensor == 0)
        depth_mask = (phase_tensor == 1)

        # 2. Backbone Forward
        bb_input_ids = input_ids.clone()
        bb_positions = positions.clone()

        for i in range(B):
            req_id = int(req_indices[i])
            step = self._csm_req_state.get(req_id, -1)
            # Sample-and-Hold
            if step >= 0:
                if req_id in self._csm_cached_cb0:
                    bb_input_ids[i] = self._csm_cached_cb0[req_id]
                if req_id in self._csm_virtual_pos:
                    bb_positions[i] = self._csm_virtual_pos[req_id]

        bb_input_embeds = self._get_hybrid_embeddings(bb_input_ids)
        hidden_states = self.model(
            input_ids=bb_input_ids, positions=bb_positions, forward_batch=forward_batch,
            pp_proxy_tensors=pp, input_embeds=bb_input_embeds
        )

        next_token_logits = torch.empty((B, self.vocab_size), dtype=torch.float32, device=device)

        # 3. Backbone Logits
        if backbone_mask.any():
            bb_idx = backbone_mask.nonzero(as_tuple=False).view(-1)
            bb_lp_out = self.logits_processor(bb_input_ids, hidden_states, self.lm_head, forward_batch)
            logits = bb_lp_out.next_token_logits[bb_idx]

            for local_i, global_i in enumerate(bb_idx.tolist()):
                req_id = int(req_indices[global_i])
                step = self._csm_req_state.get(req_id, -1)
                # Mask if in Audio Start (Phase 0)
                if step == 0:
                    self._apply_audio_mask(logits, local_i)

            next_token_logits[bb_idx] = logits
            
            last_h = hidden_states if hidden_states.dim() == 2 else hidden_states[:, -1, :]
            for row in bb_idx.tolist():
                req_id = int(req_indices[row])
                self._csm_backbone_h_table[req_id] = last_h[row].detach()

        # 4. Depth Decoder
        if depth_mask.any():
            if self.depth_decoder is None:
                self._load_depth_decoder()

            dd_idx = depth_mask.nonzero(as_tuple=False).view(-1)
            raw_input_ids = input_ids[dd_idx]

            # --- CRITICAL FIX FOR CUDA CRASH ---
            # Clamp inputs to valid codebook range [0, 2050].
            # If backbone hallucinated (e.g. 55655), this forces it to 2050, preventing OOB.
            dd_input_ids = torch.clamp(raw_input_ids, min=0, max=self.codebook_pad)
            
            if (raw_input_ids > self.codebook_pad).any():
                logger.warning(f"CSM: Sanitized invalid inputs: {raw_input_ids[raw_input_ids > self.codebook_pad]}")

            backbone_h_list = []
            past_batch = []
            depth_pos_list = [] 
            hidden_dtype = self.model.embed_tokens.weight.dtype
            
            for row in dd_idx.tolist():
                req_id = int(req_indices[row])
                step = self._csm_req_state[req_id]
                
                h = self._csm_backbone_h_table.get(req_id)
                if h is None: h = torch.zeros(self.backbone_hidden_size, device=device, dtype=hidden_dtype)
                
                backbone_h_list.append(h.to(dtype=hidden_dtype))
                past_batch.append(self._csm_depth_past_table.get(req_id))
                depth_pos_list.append(step)

            backbone_last_h = torch.stack(backbone_h_list, dim=0).unsqueeze(1)
            dd_input_2d = dd_input_ids.unsqueeze(-1) if dd_input_ids.dim() == 1 else dd_input_ids

            depth_logits_rows = []
            new_past_list = []

            for i in range(len(dd_idx)):
                # Explicit position helps select the correct Head
                rel_pos = torch.tensor([depth_pos_list[i]], device=device, dtype=torch.long)
                
                try:
                    out = self.depth_decoder(
                        input_ids=dd_input_2d[i : i+1],
                        backbone_last_hidden_state=backbone_last_h[i : i+1],
                        past_key_values=past_batch[i],
                        use_cache=True,
                        cache_position=rel_pos 
                    )
                    depth_logits_rows.append(out.logits[:, -1, :])
                    new_past_list.append(out.past_key_values)
                except Exception as e:
                    logger.warning(f"CSM Depth Error: {e}")
                    fallback = torch.full((1, self.depth_decoder.config.vocab_size), -100.0, device=device, dtype=hidden_dtype)
                    depth_logits_rows.append(fallback)
                    new_past_list.append(None)

            depth_logits = torch.cat(depth_logits_rows, dim=0)
            
            # Align vocabulary
            padded = torch.full((len(dd_idx), self.vocab_size), float("-inf"), device=device, dtype=next_token_logits.dtype)
            valid = min(depth_logits.shape[-1], self.vocab_size)
            padded[:, :valid] = depth_logits[:, :valid]
            next_token_logits[dd_idx] = padded
            
            for i, row in enumerate(dd_idx.tolist()):
                req_id = int(req_indices[row])
                self._csm_depth_past_table[req_id] = new_past_list[i]

        return LogitsProcessorOutput(next_token_logits=next_token_logits, hidden_states=None)

    def _load_depth_decoder(self):
        if not self._depth_path:
             raise RuntimeError("CSM_DEPTH_DECODER_PATH must be set")
        
        cfg_path = os.path.join(self._depth_path, "config.json")
        with open(cfg_path, "r") as f:
            d = json.load(f)
        
        self.num_codebooks = int(d.get("num_codebooks", 32))
        self.depth_decoder = AutoModelForCausalLM.from_pretrained(
             self._depth_path, 
             torch_dtype=self._depth_dtype,
             trust_remote_code=True, 
             local_files_only=True
        ).to(self.device)

EntryClass = CsmLlamaWrapper