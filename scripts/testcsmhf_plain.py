# scripts/test_csm_depth_harness.py
import os
import json
import torch
from transformers import PretrainedConfig, CsmDepthDecoderForCausalLM

DEPTH_PATH = os.environ.get("CSM_DEPTH_DECODER_PATH", "/home/shaan/Projects/csm_decoder/csm-1b-depth-decoder")

def main():
    cfg_path = os.path.join(DEPTH_PATH, "config.json")
    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)
    depth_config = PretrainedConfig.from_dict(cfg_dict)
    dtype = torch.float16
    device = "cuda"

    dec = CsmDepthDecoderForCausalLM.from_pretrained(
        DEPTH_PATH,
        config=depth_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)

    print("config hidden_size:", depth_config.hidden_size)
    print("config backbone_hidden_size:", getattr(depth_config, "backbone_hidden_size", None))
    print("model embed_tokens weight shape:", tuple(dec.model.embed_tokens.weight.shape))
    embed_hidden = dec.model.embed_tokens.weight.shape[1]  # backbone_hidden_size
    print("model embed (backbone) hidden:", embed_hidden)

    # HF depth expects backbone_last_hidden_state with backbone_hidden_size
    H = embed_hidden
    vocab_depth = depth_config.vocab_size  # expected 2051

    # Fabricate a deterministic backbone last hidden state
    torch.manual_seed(0)
    backbone_last_h = torch.randn(1, H, device=device, dtype=dtype)
    print("backbone_last_h shape:", tuple(backbone_last_h.shape))

    # Codebook-0 token to feed (e.g., argmax of backbone logits); pick 1 for test
    cb0 = torch.tensor([[1]], device=device)

    # Step 1: depth for codebook-1
    out1 = dec(
        input_ids=cb0,                          # [1, 1]
        backbone_last_hidden_state=backbone_last_h,  # [1, H]
        past_key_values=None,
        use_cache=True,
        logits_to_keep=1,                       # match our wrapper usage
    )
    logits1 = out1.logits[:, -1, :]             # [1, vocab_depth]
    tok1 = logits1.argmax(dim=-1)               # [1]

    # Step 2: depth for codebook-2 using cache
    out2 = dec(
        input_ids=tok1.unsqueeze(1),            # [1, 1]
        backbone_last_hidden_state=None,        # per HF: omit after first step
        past_key_values=out1.past_key_values,
        use_cache=True,
        logits_to_keep=1,
    )
    logits2 = out2.logits[:, -1, :]             # [1, vocab_depth]
    tok2 = logits2.argmax(dim=-1)

    # Optional: pad to backbone vocab (128256) like our wrapper does
    padded1 = torch.full((1, 128256), float("-inf"), device=device, dtype=logits1.dtype)
    padded1[:, :vocab_depth] = logits1
    padded2 = torch.full((1, 128256), float("-inf"), device=device, dtype=logits2.dtype)
    padded2[:, :vocab_depth] = logits2

    print("depth vocab size:", vocab_depth)
    print("tok1 (step1 argmax):", tok1.item())
    print("tok2 (step2 argmax):", tok2.item())
    print("logits1 shape:", logits1.shape, "padded1 shape:", padded1.shape)
    print("logits2 shape:", logits2.shape, "padded2 shape:", padded2.shape)
    print("any NaN step1:", torch.isnan(logits1).any().item(), "any NaN step2:", torch.isnan(logits2).any().item())

if __name__ == "__main__":
    main()