"""Audio Flamingo 2 (3B; NVIDIA, ICML 2025) adapter for the AudioGround baseline driver.

Official code : https://github.com/NVIDIA/audio-flamingo  branch `audio_flamingo_2`
                (cloned to eval/third_party/audio-flamingo), inference path = inference_HF_pretrained/
                inference.py  (the official HF-checkpoint inference script).
Weights       : HF nvidia/audio-flamingo-2 -> <ckpt>/safe_ckpt/*.safetensors (full Flamingo state
                dict incl. AF-CLAP, audio transformer, xattn, Qwen2.5-3B LM) + <ckpt>/clap_ckpt/epoch_16.pt.
                LM/tokenizer skeleton: Qwen/Qwen2.5-3B copied to <ckpt>/Qwen2.5-3B (the yaml's
                `Qwen/Qwen2.5-3B`, loaded locally; its weights are overwritten by safe_ckpt anyway).
Runtime code  : patched COPY eval/third_party/af2_inference_cpu (= inference_HF_pretrained) with ONE
                change: src/factory.py CLAP() used `device_id = f'cuda:{torch.cuda.current_device()}'`
                unconditionally -> falls back to 'cpu' when CUDA is unavailable. Also the two git-LFS
                files of my_laion_clap (bpe vocab, audioset_textmap.npy) were filled from the pip
                laion-clap==1.1.3 wheel (listed in the official requirements) since git-lfs is absent.
Config        : official configs/inference.yaml unchanged except absolute paths (clap checkpoint,
                LM path, cache_dir).

Long audio (OFFICIAL BEHAVIOUR KEPT): AF-CLAP sliding windows of window_length=10 s,
  window_overlap=0 s, max_num_window=12 (=> up to 120 s per the released inference.yaml; the AF2
  paper's "up to 5 min" long-audio setting is NOT what the released HF inference config uses, and we
  keep the released config). Audio is read at 16 kHz (librosa resample), peak-normalised, zero-padded
  to a multiple of 10 s; num_windows = ceil(T/10 s) (1 for T<=10 s, capped at 12; audio > 120 s is
  truncated to the first 120 s). So 30 s clipped audio -> 3 windows, <=60 s original audio -> <=6
  windows: the WHOLE clip is seen. Each window -> AF-CLAP (fusion, 10 s = exactly max_len so no random
  crop; deterministic) -> audio transformer -> gated xattn into Qwen2.5-3B.
Prompt        : official format  "<audio>{prompt.lower().strip()}<SEP>"  (the official script
                lower-cases the question; kept).
Decoding      : official script samples (do_sample=True, top_k=30, top_p=0.95, max_new_tokens=256;
                README notes temperature=0 for benchmarking). We use the shared paper setting
                num_beams=4, do_sample=False, max_new_tokens=100 (Flamingo.generate supports beams
                natively by repeat_interleaving audio_x). Output post-processing identical to official.
Precision     : official script runs the model in fp32 (precision=amp_bf16 -> cast_dtype None, and the
                autocast object is never entered). We keep fp32 by default (CPU and GPU).
"""
from __future__ import annotations

import os
import sys

RUNNERS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNERS)
AF2_CODE = "eval/third_party/af2_inference_cpu"
DEFAULT_CKPT = "ckpt/baselines/af2"
os.environ.setdefault("HF_HOME", "ckpt/hf_home")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import common  # noqa: E402


class Model:
    def __init__(self, ckpt_dir, device, **kw):
        import json
        import yaml
        import torch
        from safetensors.torch import load_file

        sys.path.insert(0, AF2_CODE)
        from src.factory import create_model_and_transforms  # noqa
        import inference as af2_inf  # official helper functions (load_audio / get_num_windows)
        from utils import Dict2Class, get_cast_dtype  # noqa

        ckpt_dir = ckpt_dir or DEFAULT_CKPT
        config = yaml.load(open(os.path.join(AF2_CODE, "configs", "inference.yaml")), Loader=yaml.FullLoader)
        model_config = config["model_config"]
        clap_config = config["clap_config"]
        args = Dict2Class(config["train_config"])
        clap_config["checkpoint"] = os.path.join(ckpt_dir, "clap_ckpt", "epoch_16.pt")
        model_config["lang_encoder_path"] = os.path.join(ckpt_dir, "Qwen2.5-3B")
        model_config["tokenizer_path"] = os.path.join(ckpt_dir, "Qwen2.5-3B")
        model_config["cache_dir"] = os.path.join(os.environ["HF_HOME"], "af2_cache")
        self.clap_config = clap_config
        self.af2_inf = af2_inf

        model, tokenizer = create_model_and_transforms(
            **model_config, clap_config=clap_config, use_local_files=args.offline,
            gradient_checkpointing=args.gradient_checkpointing,
            freeze_lm_embeddings=args.freeze_lm_embeddings)
        model = model.to(device)
        model.eval()
        with open(os.path.join(ckpt_dir, "safe_ckpt", "metadata.json")) as f:
            metadata = json.load(f)
        state_dict = {}
        for chunk_name in metadata:
            state_dict.update(load_file(os.path.join(ckpt_dir, "safe_ckpt", f"{chunk_name}.safetensors")))
        missing, unexpected = model.load_state_dict(state_dict, False)
        print(f"[af2] loaded {len(state_dict)} tensors; missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
        if missing:
            print("[af2] missing (first 10):", missing[:10], flush=True)
        del state_dict
        self.cast_dtype = get_cast_dtype(args.precision)  # None -> fp32 (official)
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.torch = torch

    def generate(self, audio_path, prompt, gen):
        torch, tok = self.torch, self.tokenizer
        audio_clips, audio_embed_mask = self.af2_inf.load_audio(audio_path, self.clap_config)
        audio_clips = audio_clips.to(self.device, dtype=self.cast_dtype)
        audio_embed_mask = audio_embed_mask.to(self.device, dtype=self.cast_dtype)
        sample = f"<audio>{str(prompt).lower().strip()}{tok.sep_token}"
        text = tok(sample, max_length=512, padding="longest", truncation="only_first", return_tensors="pt")
        input_ids = text["input_ids"].to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                audio_x=audio_clips.unsqueeze(0),
                audio_x_mask=audio_embed_mask.unsqueeze(0),
                lang_x=input_ids,
                eos_token_id=tok.eos_token_id,
                max_new_tokens=gen["max_new_tokens"],
                num_beams=gen["num_beams"],
                do_sample=gen["do_sample"],
                num_return_sequences=1,
            )[0]
        out = tok.decode(output).split(tok.sep_token)[-1].replace(tok.eos_token, "") \
            .replace(tok.pad_token, "").replace("<|endofchunk|>", "")
        return out.strip()


if __name__ == "__main__":
    common.main(Model, "af2")
