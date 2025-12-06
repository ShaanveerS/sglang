# python/sglang/srt/models/csm_llama_wrapper.py

from typing import Any, Dict, List, Optional
import logging

import os
import torch
import torch.nn as nn

from transformers import AutoConfig, CsmDepthDecoderForCausalLM  # HF CSM depth decoder
from transformers import PretrainedConfig
import json

from sglang.srt.layers.logits_processor import LogitsProcessorOutput, LogitsMetadata
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.llama import LlamaForCausalLM

logger = logging.getLogger(__name__)


class CsmLlamaWrapper(LlamaForCausalLM):
    """
    CSM wrapper for SGLang:

    - Backbone = SGLang LlamaForCausalLM (with RadixAttention + KV cache)
    - Depth decoder = HF CsmDepthDecoderForCausalLM (with its own HF cache)
    - Routing is controlled via ForwardBatch.model_specific_states["csm_phase"]:
        0 => backbone step (codebook 0 / normal LM)
        1 => depth step    (codebooks 1..N, uses backbone_last_hidden_state)
    """

    def __init__(
        self,
        config,  # LlamaConfig
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # Initialize the backbone as a normal SGLang LlamaForCausalLM
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

        self.backbone_hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # Depth decoder lazy-load info. We only load when phase==1 is actually used.
        self._depth_path = os.environ.get("CSM_DEPTH_DECODER_PATH", None)
        self.depth_decoder = None
        # Default num_codebooks before loading depth config
        self.num_codebooks = 32
        # Choose dtype for depth to roughly match backbone
        self._depth_dtype = None
        cfg_dtype = getattr(config, "torch_dtype", None)
        if isinstance(cfg_dtype, torch.dtype):
            self._depth_dtype = cfg_dtype
        elif isinstance(cfg_dtype, str):
            s = cfg_dtype.lower()
            if s in ("bfloat16", "bf16"):
                self._depth_dtype = torch.bfloat16
            elif s in ("float16", "fp16", "half"):
                self._depth_dtype = torch.float16
        self.device = next(self.parameters()).device
        # NEW: per-request tables (keyed by req_pool_idx)
        self._csm_backbone_h_table: dict[int, torch.Tensor] = {}
        self._csm_depth_past_table: dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,           # [B, T], T=1 for decode
        positions: torch.Tensor,           # [B, T]
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        B = input_ids.shape[0]
        device = input_ids.device

        # ---------------- model-specific state ----------------
        ms: Dict[str, Any] = forward_batch.model_specific_states or {}
        phase_in = ms.get("csm_phase", None)
        if phase_in is None:
            # Boot fallback: treat all rows as backbone to behave like a normal LLaMA
            phase = torch.zeros(B, dtype=torch.int64, device=device)
        else:
            phase = phase_in.to(device)  # 0=backbone, 1=depth
        assert phase.shape[0] == B, "csm_phase must match batch size"

        backbone_mask = (phase == 0)
        depth_mask = (phase == 1)

        num_backbone = int(backbone_mask.sum())
        num_depth = int(depth_mask.sum())

        # If all rows are backbone (common in boot smoke test), delegate to base class
        if num_depth == 0:
            return super().forward(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                input_embeds=input_embeds,
                get_embedding=get_embedding,
            )

        # Map batch rows to global request slots
        req_indices = forward_batch.req_pool_indices.to(torch.int64)  # [B]

        # Output logits
        next_token_logits = torch.empty(
            (B, self.vocab_size),
            dtype=torch.float32,
            device=device,
        )

        # Always run backbone on the full batch to keep KV semantics and enable depth fallback.
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            input_embeds=input_embeds,
        )

        bb_lp_out = self.logits_processor(  # type: ignore[attr-defined]
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )
        logits_full = bb_lp_out.next_token_logits  # [B, vocab]

        # ======================================================
        # 1) BACKBONE: use normal SGLang Llama stack + KV cache
        # ======================================================
        if num_backbone > 0:
            bb_idx = backbone_mask.nonzero(as_tuple=False).view(-1)  # [num_backbone]

            if hidden_states.dim() == 3:
                last_h = hidden_states[:, -1, :]        # [B, H]
            elif hidden_states.dim() == 2:
                last_h = hidden_states                  # [B, H]
            else:
                raise RuntimeError(
                    f"Unexpected hidden_states ndim={hidden_states.dim()}"
                )

            # Scatter logits and stash per-request backbone hidden
            next_token_logits[bb_idx] = logits_full[bb_idx]

            for row in bb_idx.tolist():
                req_id = int(req_indices[row])
                # detach so we don't keep autograd graphs around
                self._csm_backbone_h_table[req_id] = last_h[row].detach()

        # ======================================================
        # 2) DEPTH: HF CsmDepthDecoderForCausalLM
        # ======================================================
        if num_depth > 0:
            logger.info("CSM depth step: rows=%d", num_depth)
            dd_idx = depth_mask.nonzero(as_tuple=False).view(-1)  # [num_depth]
            strict = os.environ.get("CSM_NO_DEPTH_FALLBACK") == "1"

            # Lazy load the depth decoder if needed; on failure, fallback or raise.
            try:
                if self.depth_decoder is None:
                    if not self._depth_path:
                        raise RuntimeError(
                            "CSM_DEPTH_DECODER_PATH is not set but depth phase is requested."
                        )
                    # Manually read config to avoid AutoConfig mapping issues
                    cfg_path = os.path.join(self._depth_path, "config.json")
                    with open(cfg_path, "r") as f:
                        cfg_dict = json.load(f)
                    cfg_backbone_h = cfg_dict.get("backbone_hidden_size", self.backbone_hidden_size)
                    if cfg_backbone_h != self.backbone_hidden_size:
                        raise ValueError(
                            f"Depth decoder backbone_hidden_size={cfg_backbone_h}, "
                            f"but backbone hidden_size={self.backbone_hidden_size}"
                        )
                    self.num_codebooks = int(cfg_dict.get("num_codebooks", self.num_codebooks))
                    depth_config = PretrainedConfig.from_dict(cfg_dict)
                    self.depth_decoder = CsmDepthDecoderForCausalLM.from_pretrained(
                        self._depth_path,
                        config=depth_config,
                        torch_dtype=self._depth_dtype,
                        trust_remote_code=True,
                        local_files_only=True,
                    ).to(self.device)
                    logger.info("CSM depth decoder loaded from %s", self._depth_path)
            except Exception as e:
                if strict:
                    logger.exception("CSM depth load failed in strict mode")
                    raise
                logger.warning("CSM depth failed, falling back to backbone for rows=%d: %s", num_depth, e)
                next_token_logits[dd_idx] = logits_full[dd_idx]
                return LogitsProcessorOutput(
                    next_token_logits=next_token_logits,
                    hidden_states=None,
                    input_token_logprobs=None,
                    input_top_logprobs_val=None,
                    input_top_logprobs_idx=None,
                )

            dd_input_ids = input_ids[dd_idx]    # [num_depth, T]

            # Gather backbone_last_hidden_state and HF cache per request
            backbone_last_h_list: List[torch.Tensor] = []
            past_for_depth_batch: List[Any] = []

            hidden_dtype = self.model.embed_tokens.weight.dtype

            for row in dd_idx.tolist():
                req_id = int(req_indices[row])

                h = self._csm_backbone_h_table.get(req_id, None)
                if h is None:
                    # Fallback if scheduler messed up phases: zeros
                    h = torch.zeros(
                        self.backbone_hidden_size,
                        dtype=hidden_dtype,
                        device=self.device,
                    )
                else:
                    # ensure on correct device/dtype
                    h = h.to(device=self.device, dtype=hidden_dtype)

                backbone_last_h_list.append(h)
                past_for_depth_batch.append(self._csm_depth_past_table.get(req_id, None))

            backbone_last_h = torch.stack(backbone_last_h_list, dim=0)  # [num_depth, H]

            # Some CSM decoders expect per-sample Cache objects, not a batched list.
            # Call per-row to be compatible.
            depth_logits_rows: List[torch.Tensor] = []
            new_past_list: List[Any] = []
            dd_input_ids_2d = dd_input_ids if dd_input_ids.dim() == 2 else dd_input_ids.unsqueeze(-1)
            for local_row in range(len(dd_idx)):
                try:
                    out = self.depth_decoder(
                        input_ids=dd_input_ids_2d[local_row : local_row + 1],
                        backbone_last_hidden_state=backbone_last_h[local_row : local_row + 1],  # [1, H]
                        past_key_values=past_for_depth_batch[local_row],
                        use_cache=True,
                        logits_to_keep=1,  # keep last token logits (avoid empty slice)
                    )
                    depth_logits_rows.append(out.logits[:, -1, :])  # [1, vocab]
                    new_past_list.append(out.past_key_values)
                except Exception as e:
                    if strict:
                        logger.exception("CSM depth forward failed in strict mode")
                        raise
                    logger.warning("CSM depth forward failed, fallback to backbone for one row: %s", e)
                    row_slice = dd_idx[local_row : local_row + 1]
                    depth_logits_rows.append(logits_full[row_slice])
                    new_past_list.append(None)
            depth_logits = torch.cat(depth_logits_rows, dim=0)  # [num_depth, depth_vocab]
            new_past = new_past_list

            # Align depth vocab (codebook size) into backbone vocab space.
            depth_vocab = depth_logits.shape[1]
            padded = torch.full(
                (len(dd_idx), self.vocab_size),
                float("-inf"),
                device=depth_logits.device,
                dtype=depth_logits.dtype,
            )
            padded[:, :depth_vocab] = depth_logits
            padded = padded.to(next_token_logits.dtype)

            # Write back updated per-req cache and logits
            for local_row, batch_row in enumerate(dd_idx.tolist()):
                req_id = int(req_indices[batch_row])
                self._csm_depth_past_table[req_id] = new_past[local_row]

            # DEBUG/diagnostic: optionally force depth logits for all rows if shapes match
            if padded.shape[0] == next_token_logits.shape[0]:
                next_token_logits[:] = padded
            else:
                next_token_logits[dd_idx] = padded

        return LogitsProcessorOutput(
            next_token_logits=next_token_logits,
            hidden_states=None,
            input_token_logprobs=None,
            input_top_logprobs_val=None,
            input_top_logprobs_idx=None,
        )


# Let SGLang auto-discover this class when it scans /srt/models
EntryClass = CsmLlamaWrapper
