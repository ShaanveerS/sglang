import os
from typing import Optional, List, Any, Dict

import torch
import torch.nn as nn
from transformers import (
    CsmDepthDecoderForCausalLM,
    AutoConfig,
)

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.llama import LlamaForCausalLM


class CsmBackboneWithDepth(nn.Module):
    """
    SGLang model that:
      - Uses SGLang's LlamaForCausalLM as backbone (with KV cache).
      - Uses HF CsmDepthDecoderForCausalLM as depth decoder (with HF Cache).
      - Shares vocab between them (assumes same tokenizer).
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # Backbone: standard SGLang LLaMA
        self.backbone = LlamaForCausalLM(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.model = self.backbone.model  # expose for weight-loading utilities
        self.backbone_hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # Depth decoder: HF model, loaded from separate path
        depth_path = os.environ.get("CSM_DEPTH_DECODER_PATH", None)
        if depth_path is None:
            raise ValueError(
                "Set CSM_DEPTH_DECODER_PATH to the HF path of your CsmDepthDecoderForCausalLM "
                "checkpoint (e.g. 'my-org/csm-1b-depth-decoder')."
            )

        depth_config = AutoConfig.from_pretrained(depth_path)

        # Choose dtype for depth decoder (match backbone if possible)
        desired_dtype = None
        if hasattr(config, "torch_dtype"):
            cfg_dtype = getattr(config, "torch_dtype")
            if isinstance(cfg_dtype, torch.dtype):
                desired_dtype = cfg_dtype
            elif isinstance(cfg_dtype, str):
                s = cfg_dtype.lower()
                if s in ("bfloat16", "bf16"):
                    desired_dtype = torch.bfloat16
                elif s in ("float16", "fp16", "half"):
                    desired_dtype = torch.float16

        self.depth_decoder = CsmDepthDecoderForCausalLM.from_pretrained(
            depth_path,
            config=depth_config,
            torch_dtype=desired_dtype,
        )

        # Put depth decoder on same device as backbone params
        device = next(self.backbone.parameters()).device
        self.depth_decoder.to(device)

        # Basic consistency checks when metadata exists
        if getattr(depth_config, "backbone_hidden_size", self.backbone_hidden_size) != self.backbone_hidden_size:
            raise ValueError(
                f"Depth decoder backbone_hidden_size={getattr(depth_config, 'backbone_hidden_size', None)}, "
                f"but backbone hidden_size={self.backbone_hidden_size}"
            )

        self.num_codebooks = getattr(depth_config, "num_codebooks", 32)
        self.device = device

    @property
    def config(self):
        # Expose backbone config as main config
        return self.backbone.config

    @property
    def lm_head(self):
        # Use backbone lm_head for codebook-0 tokens
        return self.backbone.lm_head

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,  # [B, 1] during decode
        positions: torch.Tensor,  # [B, 1]
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        """
        Mixed forward:
          - For rows with csm_phase == 0, use SGLang backbone (with KV).
          - For rows with csm_phase == 1, use HF depth decoder (HF cache).
        """
        del input_embeds, get_embedding  # not used in this wrapper

        B = input_ids.shape[0]
        device = input_ids.device

        # --- Read / init model-specific state --- #
        ms: Dict[str, Any] = forward_batch.model_specific_states
        if ms is None or "csm_phase" not in ms:
            raise RuntimeError(
                "CSM model expects forward_batch.model_specific_states['csm_phase'] "
                "tensor of shape [batch_size]."
            )

        phase: torch.Tensor = ms["csm_phase"].to(device)  # 0=backbone, 1=depth
        assert phase.shape[0] == B

        backbone_mask = (phase == 0)
        depth_mask = (phase == 1)
        num_backbone = int(backbone_mask.sum())
        num_depth = int(depth_mask.sum())

        # Prepare per-seq backbone hidden state buffer
        backbone_h: Optional[torch.Tensor] = ms.get("csm_backbone_h", None)
        state_dtype = next(self.backbone.parameters()).dtype
        if backbone_h is None or backbone_h.shape != (B, self.backbone_hidden_size):
            backbone_h = torch.zeros(
                (B, self.backbone_hidden_size),
                dtype=state_dtype,
                device=device,
            )

        # Prepare per-seq HF depth cache list
        depth_past_list: Optional[List[Any]] = ms.get("csm_depth_past", None)
        if depth_past_list is None or len(depth_past_list) != B:
            depth_past_list = [None for _ in range(B)]

        # Storage for final logits [B, vocab_size]
        next_token_logits = torch.empty(
            (B, self.vocab_size),
            dtype=torch.float32,
            device=device,
        )

        # =====================================================================
        # 1) BACKBONE STEP (codebook 0)
        # =====================================================================
        if num_backbone > 0:
            bb_idx = backbone_mask.nonzero(as_tuple=False).squeeze(-1)  # [num_backbone]
            bb_input_ids = input_ids[bb_idx]
            bb_positions = positions[bb_idx]

            hidden_states_bb = self.backbone.model(
                bb_input_ids,
                bb_positions,
                forward_batch,  # uses full ForwardBatch for KV layout
                input_embeds=None,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            # hidden_states_bb: [num_backbone, 1, hidden_dim] during decode
            last_h_bb = hidden_states_bb[:, -1, :]  # [num_backbone, hidden_dim]

            # Update per-row backbone_h buffer
            backbone_h[bb_idx] = last_h_bb.to(state_dtype)

            # LM head over last_h_bb
            logits_bb = self.backbone.lm_head(last_h_bb)  # [num_backbone, vocab_size]
            next_token_logits[bb_idx] = logits_bb.to(torch.float32)

        # =====================================================================
        # 2) DEPTH STEP (codebook 1..num_codebooks-1)
        # =====================================================================
        if num_depth > 0:
            dd_idx = depth_mask.nonzero(as_tuple=False).squeeze(-1)  # [num_depth]
            dd_input_ids = input_ids[dd_idx]  # [num_depth, 1]

            # Collect backbone_last_hidden_state for these rows
            backbone_last_h = backbone_h[dd_idx]  # [num_depth, hidden_dim]
            backbone_last_h = backbone_last_h.unsqueeze(1)  # [num_depth, 1, hidden_dim]

            # Collect HF cache for these rows
            past_for_depth_batch: List[Any] = [depth_past_list[i] for i in dd_idx.tolist()]

            # HF call (argument names may differ across versions; adjust if needed)
            depth_outputs = self.depth_decoder(
                input_ids=dd_input_ids,
                backbone_last_hidden_state=backbone_last_h,
                past_key_values=past_for_depth_batch,
                use_cache=True,
            )
            depth_logits = depth_outputs.logits[:, -1, :]  # [num_depth, vocab_size]
            new_past = getattr(depth_outputs, "past_key_values", None)

            # Update per-row depth_past_list
            if isinstance(new_past, (list, tuple)) and len(new_past) == num_depth:
                for local_row, global_row in enumerate(dd_idx.tolist()):
                    depth_past_list[global_row] = new_past[local_row]
            else:
                # Fallback: store the same structure for all rows (best-effort)
                for global_row in dd_idx.tolist():
                    depth_past_list[global_row] = new_past

            next_token_logits[dd_idx] = depth_logits.to(torch.float32)

        # Save updated states back into ForwardBatch so scheduler can keep them
        ms["csm_backbone_h"] = backbone_h
        ms["csm_depth_past"] = depth_past_list
        forward_batch.model_specific_states = ms

        # Wrap into LogitsProcessorOutput; sampler will handle temperature, top-k, etc.
        logits_output = LogitsProcessorOutput(
            next_token_logits=next_token_logits,
            hidden_states=None,
            input_token_logprobs=None,
            input_top_logprobs_val=None,
            input_top_logprobs_idx=None,
        )
        return logits_output

    # Optional: forward weight loading to backbone
    def load_weights(self, weights):
        return self.backbone.load_weights(weights)


# Register into SGLang model registry
EntryClass = [CsmBackboneWithDepth]


