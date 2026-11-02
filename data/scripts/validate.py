#!/usr/bin/env python
"""Validate a variant: re-extract every segment from the source clip memmap and check it matches the wav at
the recorded start time (correlation), check GT intervals == target segment label edges, durations, rounding."""
import json, sys, random
import numpy as np, pandas as pd, soundfile as sf
R = "data"
jsonl, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200
rows = [json.loads(l) for l in open(jsonl)]
random.seed(0); rows = random.sample(rows, min(n, len(rows)))
pools = pd.read_parquet(f"{R}/pools/pools.parquet"); aid2row = dict(zip(pools.audiocap_id, range(len(pools))))
mm = np.memmap(f"{R}/clips/clips_16k.int16", dtype=np.int16, mode="r")
bad, cors, nseg = 0, [], 0
for r in rows:
    y, sr = sf.read(r["audio"], dtype="float32"); m = r["meta"]
    assert sr == 16000 and abs(len(y) / sr - m["duration"]) < 0.01, (len(y) / sr, m["duration"])
    for s in m["segments"]:
        i = aid2row[s["audiocap_id"]]; o = int(pools.offset[i])
        a, b = int(round(s["crop"][0] * sr)), int(round(s["crop"][1] * sr))
        x = mm[o + a:o + b].astype(np.float32) / 32768
        st0 = int(round(s["start"] * sr)); k = len(x) // 4; xa = x[k:-k]; best = -1
        for lag in range(-24, 25):
            st = st0 + lag
            if st + k < 0: continue
            wa = y[max(0, st + k):st + len(x) - k]
            if len(wa) != len(xa): continue
            best = max(best, float(np.dot(xa, wa) / (np.linalg.norm(xa) * np.linalg.norm(wa) + 1e-9)))
        c = best; cors.append(c); nseg += 1
    for iv in r["gt_intervals"]:
        assert iv == [round(iv[0], 1), round(iv[1], 1)] and 0 <= iv[0] < iv[1] <= m["duration"] + 1e-6, (iv, m["duration"])
    assert all(abs(g[0] - e[0]) <= 0.051 and (abs(g[1] - e[1]) <= 0.051 or (g[1] < e[1] <= g[1] + 0.1 and e[1] > m["duration"] - 0.1)) for g, e in zip(r["gt_intervals"], m["gt_exact"]))
    # GT == outer label edges of target segments (silence/nogap: exact segment edges)
    if m["junction"]["mode"] in ("silence", "nogap", "bgmix") and not m.get("strong"):
        tg = {}
        for s in m["segments"]:
            if s["role"] == "target": tg.setdefault(s["tgt"], []).append(s)
        edges = sorted([round(v[0]["start"], 1), round(v[-1]["end"], 1)] for v in tg.values())
        ok = all(abs(a - b) <= 0.11 for e, g in zip(edges, sorted(r["gt_intervals"])) for a, b in zip(e, g))
        if not ok: bad += 1; print("GT mismatch", m["id"], edges, r["gt_intervals"])
cors = np.array(cors)
print("nonzero-but-low corr (0<c<0.9):", int(((cors>0)&(cors<0.9)).sum())); print(f"GT mismatches: {bad}; {len(rows)} samples, {nseg} segments; segment corr: min {cors.min():.4f} p1 {np.percentile(cors,1):.4f} median {np.median(cors):.4f}; <0.9: {(cors<0.9).sum()}")
