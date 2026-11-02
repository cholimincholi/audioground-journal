"""Builds tiny random Whisper / BEATs / LLaMA (+ fake SALMONN ckpt) for CPU smoke tests.
  python tools/make_tiny_models.py --out tests/tiny
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only check: never touch a GPU
import argparse
import os
import shutil
import sys

import torch
from transformers import LlamaConfig, LlamaForCausalLM, WhisperConfig, WhisperModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CK = os.environ.get("AG_CKPT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../ckpt")))

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests/tiny"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0)

# LLaMA (vicuna tokenizer)
ld = os.path.join(a.out, "llama")
LlamaForCausalLM(LlamaConfig(vocab_size=32000, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                             num_attention_heads=4, max_position_embeddings=2048)).save_pretrained(ld)
for f in ["tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(os.path.join(CK, "vicuna-7b-v1.5", f), ld)

# Whisper encoder (80 mel, 1500 positions -> 30 s input) + real feature extractor config
wd = os.path.join(a.out, "whisper")
WhisperModel(WhisperConfig(d_model=64, encoder_layers=2, encoder_attention_heads=2, decoder_layers=1,
                           decoder_attention_heads=2, encoder_ffn_dim=128, decoder_ffn_dim=128, num_mel_bins=80,
                           max_source_positions=1500, vocab_size=51865)).save_pretrained(wd)
shutil.copy(os.path.join(CK, "whisper-large-v2", "preprocessor_config.json"), wd)

# BEATs with real cfg but tiny transformer
from audioground.beats.BEATs import BEATs, BEATsConfig
real = torch.load(os.path.join(CK, "beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"), map_location="cpu")
cfg = dict(real["cfg"])
cfg.update(encoder_layers=1, encoder_embed_dim=64, encoder_ffn_embed_dim=128, encoder_attention_heads=2,
           finetuned_model=False)
b = BEATs(BEATsConfig(cfg))
torch.save({"cfg": cfg, "model": b.state_dict()}, os.path.join(a.out, "beats.pt"))

# fake SALMONN ckpt: vanilla model (no timestamp, no ATE, LoRA r=8) trainable weights
from audioground.config import load_config
from audioground.model import AudioGround
c = load_config(overrides=[f"model.llama_path={ld}", f"model.whisper_path={wd}",
                           f"model.beats_path={os.path.join(a.out, 'beats.pt')}", "model.salmonn_ckpt=",
                           "model.timestamp_cond=false", "model.ate=none", "model.lora_rank=8", "model.lora_alpha=32"]).model
m = AudioGround(c)
sd = m.trainable_state_dict()
for k in sd:
    if "lora_B" in k:
        sd[k] = torch.randn_like(sd[k]) * 0.02    # non-zero so the merge is testable
torch.save({"model": sd}, os.path.join(a.out, "salmonn_fake.pth"))
print("tiny models written to", a.out, "| fake SALMONN tensors:", len(sd))
