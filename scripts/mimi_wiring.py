# scripts/decode_csm_with_mimi.py
import torch
from transformers import AutoModel

# Path to Mimi (pull once): "kyutai/mimi" or your local copy
MIMI_ID = "kyutai/mimi"

# These should match your CSM config
AUDIO_TOKEN_ID = 128002
CODEBOOK_EOS_TOKEN_ID = 0
NUM_CODEBOOKS = 32

def extract_codebook_frames(prompt_ids, gen_ids):
    all_ids = list(prompt_ids) + list(gen_ids)
    frames = []
    in_audio = False
    cur = []
    for tok in all_ids:
        if tok == AUDIO_TOKEN_ID:
            in_audio = True
            cur = []
            continue
        if not in_audio:
            continue
        if tok == CODEBOOK_EOS_TOKEN_ID:
            break
        cur.append(tok)
        if len(cur) == NUM_CODEBOOKS:
            frames.append(cur)
            cur = []
    return torch.tensor(frames, dtype=torch.long)  # [T, 32]

def decode_with_mimi(frames: torch.Tensor, device="cuda"):
    # frames: [T, num_codebooks]
    model = AutoModel.from_pretrained(MIMI_ID, trust_remote_code=True).to(device)
    # Mimi expects shape [batch, num_codebooks, T]
    cb = frames.transpose(0, 1).unsqueeze(0).to(device)  # [1, 32, T]
    with torch.no_grad():
        audio = model.decode_codebook(cb)  # returns samples, shape [1, 1, samples]
    return audio[0, 0].cpu()

def main():
    # Example: replace with your actual ids from SGLang response
    prompt_ids = [128000, 2323, 220, AUDIO_TOKEN_ID]
    gen_ids = [1] * 32  # fake single frame of nonzero codes

    frames = extract_codebook_frames(prompt_ids, gen_ids)
    if frames.numel() == 0:
        print("No frames found")
        return
    waveform = decode_with_mimi(frames)
    print("Decoded waveform shape:", waveform.shape)
    # Optionally save
    import soundfile as sf
    sf.write("out.wav", waveform.numpy(), 16000)

if __name__ == "__main__":
    main()