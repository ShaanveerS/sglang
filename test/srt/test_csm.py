import torch
from transformers import AutoModel, AutoModelForCausalLM

DEVICE = "cuda"
DTYPE = torch.float16

CSM_NAME = "sesame/csm-1b"
DEPTH_LORA_PATH = "/home/shaan/Projects/csm_decoder/csm-depth-trunk-llama"

def test_depth_trunk_matches_csm():
    print("[1] Loading original CSM depth decoder:", CSM_NAME)
    csm = AutoModel.from_pretrained(
        CSM_NAME,
        trust_remote_code=True,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    depth_module = csm.depth_decoder               # CsmDepthDecoderForCausalLM
    depth_model = depth_module.model               # CsmDepthDecoderModel

    print("Depth hidden_size:", depth_model.config.hidden_size)
    print("Depth num_layers:", depth_model.config.num_hidden_layers)

    print("\n[2] Loading exported LLaMA depth trunk:", DEPTH_LORA_PATH)
    llama_depth = AutoModelForCausalLM.from_pretrained(
        DEPTH_LORA_PATH,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    llama_trunk = llama_depth.model                # LlamaModel

    print("LLaMA hidden_size:", llama_trunk.config.hidden_size)
    print("LLaMA num_layers:", llama_trunk.config.num_hidden_layers)

    assert depth_model.config.hidden_size == llama_trunk.config.hidden_size
    assert depth_model.config.num_hidden_layers == llama_trunk.config.num_hidden_layers

    B, T, H = 2, 16, depth_model.config.hidden_size
    inputs_embeds = torch.randn(B, T, H, device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out_ref = depth_model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
        ).last_hidden_state

        out_test = llama_trunk(
            inputs_embeds=inputs_embeds,
            use_cache=False,
        ).last_hidden_state

    diff = (out_ref - out_test).abs().max().item()
    print("Max diff:", diff)
    assert diff < 1e-3

if __name__ == "__main__":
    test_depth_trunk_matches_csm()