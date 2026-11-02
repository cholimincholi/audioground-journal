"""Writes random-noise + tone dummy wavs and train/val/bench JSONL for smoke tests."""
import json, os, sys
import numpy as np, soundfile as sf
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests/dummy")
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
rows = []
for i, dur in enumerate([12.0, 45.0, 75.3, 30.0, 61.0, 8.0]):
    sr = 16000 if i % 2 == 0 else 22050          # also exercises resampling
    t = np.arange(int(dur * sr)) / sr
    wav = 0.05 * rng.standard_normal(len(t))
    s, e = round(dur * 0.2, 1), round(dur * 0.2 + min(4.0, dur * 0.3), 1)
    m = (t >= s) & (t < e); wav[m] += 0.5 * np.sin(2 * np.pi * 880 * t[m])
    p = os.path.join(out, f"a{i}.wav"); sf.write(p, wav.astype(np.float32), sr)
    rows.append({"id": f"a{i}", "audio": p, "instruction": "When does the beeping tone occur?",
                 "response": f"The beeping tone occurs from {s} to {e} seconds.", "query": "a beeping tone", "spans": [[s, e]]})
def w(name, rs):
    with open(os.path.join(out, name), "w") as f:
        for r in rs: f.write(json.dumps(r) + "\n")
w("train.jsonl", rows); w("val.jsonl", rows[:2])
w("bench.jsonl", [{k: v for k, v in r.items() if k not in ("instruction", "response")} for r in rows])
print("dummy data in", out)
