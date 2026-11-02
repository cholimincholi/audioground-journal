"""Real-weight check on CPU (no GPU): SALMONN-7B in vanilla mode must caption a demo clip sensibly;
then the full AudioGround config does one forward/backward + a beam-4 generation.
  CUDA_VISIBLE_DEVICES= python tests/real_cpu_check.py [7b|13b]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only check: never touch a GPU
import os, sys, time
import torch
from omegaconf import OmegaConf
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from audioground.config import load_config
from audioground.data import AudioGroundDataset, collate
from audioground.model import AudioGround
torch.set_num_threads(int(os.environ.get("THREADS", 32)))
size = sys.argv[1] if len(sys.argv) > 1 else "7b"
cfgp = os.path.join(ROOT, f"configs/audioground_{size}.yaml")
demo = "eval/third_party/SALMONN/resource/audio_demo"
rows = [{"audio": f"{demo}/duck.wav", "instruction": "Please describe the audio in detail.", "response": "Ducks quack from 1.0 to 3.5 seconds."},
        {"audio": f"{demo}/gunshots.wav", "instruction": "Please describe the audio in detail.", "response": "Gunshots from 0.5 to 2.0 seconds."}]
g = OmegaConf.create({"num_beams": 4, "do_sample": False, "max_new_tokens": 40, "min_length": 1, "repetition_penalty": 1.0, "length_penalty": 1.0})

def run(tag, ov, train_step):
    cfg = load_config([cfgp], ov + ["train.device=cpu"])
    t = time.time(); torch.manual_seed(0)
    m = AudioGround(cfg.model); print(f"[{tag}] build {time.time()-t:.0f}s")
    ds = AudioGroundDataset(rows, cfg.model, cfg.data, cfg.model.whisper_path, train=True)
    b = collate([ds[0], ds[1]])
    if train_step:
        m.train(); t = time.time()
        out = m(b); out["loss"].backward()
        ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[{tag}] fwd+bwd loss {out['loss'].item():.3f} ({time.time()-t:.0f}s) trainable {ntr/1e6:.1f}M "
              f"beta.grad {m.ate.beta.grad if hasattr(m.ate,'beta') else None}")
    m.eval(); t = time.time()
    outs = m.generate(b, g)
    print(f"[{tag}] generate ({time.time()-t:.0f}s):"); [print("   ", o) for o in outs]
    del m

run("vanilla-SALMONN", ["model.timestamp_cond=false", "model.ate=none", "model.beats_align=pad", "model.lora=false"], False)
run("AudioGround-default", [], True)
