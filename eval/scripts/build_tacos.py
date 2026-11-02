"""TACOS (Primus et al., WASPAA 2025) test split -> benchmark JSONL in our format (real Freesound recordings, natural-language
region captions with onset/offset; R1-1 / R5-M4 real-audio evaluation).

One query = one (recording, caption) pair; its ground truth is every annotated region with that caption (all kept).
The released TACOS audio are 15-30-s excerpts of the Freesound files (start_time_s..end_time_s in metadata.csv) and the
annotations are relative to the excerpt, so every recording is used whole. (A 120-s window with a uniformly drawn start
would be cut for longer recordings; none occurs.)
Audio is decoded from MP3 and written as 16-kHz mono WAV.

usage: python eval/scripts/build_tacos.py
out  : eval/benchmarks/tacos/original.jsonl, audio_original/*.wav, stats.json
"""
import collections
import csv
import json
import os

import numpy as np
import soundfile as sf

SRC = "eval/benchmarks/_dl_tacos"
OUT = "eval/benchmarks/tacos"
SR = 16000
MAXL = 120.0


def find_audio(fn):
    for root in (f"{SRC}/audio", f"{SRC}/audio/audio"):
        p = os.path.join(root, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


def load_16k_mono(p):
    import torch
    import torchaudio
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(1))
    if sr != SR:
        x = torchaudio.functional.resample(x, sr, SR)
    return x.numpy()


def main():
    os.makedirs(f"{OUT}/audio_original", exist_ok=True)
    test = [r["filename"] for r in csv.DictReader(open(f"{SRC}/test_split.csv"))]
    ann = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{SRC}/annotations_strong.csv")):
        ann[(r["filename"], r["text"].strip())].append((float(r["onset"]), float(r["offset"])))
    by_file = collections.defaultdict(list)
    for (fn, text), iv in ann.items():
        by_file[fn].append((text, sorted(iv)))
    rng = np.random.default_rng(0)
    rows, n_win, n_drop, n_missing = [], 0, 0, 0
    for fn in sorted(test):
        if fn not in by_file:
            continue
        try:
            x = load_16k_mono(find_audio(fn))
        except Exception as e:  # unreadable / missing file
            n_missing += 1
            print("skip", fn, e)
            continue
        L = len(x) / SR
        stem = os.path.splitext(fn)[0]
        whole = None
        for k, (text, iv) in enumerate(by_file[fn]):
            iv = [(max(0.0, s), min(L, e)) for s, e in iv if min(L, e) - max(0.0, s) > 0]
            if not iv:
                n_drop += 1
                continue
            if L <= MAXL:
                if whole is None:
                    whole = f"{OUT}/audio_original/{stem}.wav"
                    sf.write(whole, x, SR)
                path, t0, Lw = whole, 0.0, L
            else:
                s0, e0 = iv[0]
                lo, hi = max(0.0, e0 - MAXL), min(s0, L - MAXL)
                t0 = float(rng.uniform(lo, hi)) if hi > lo else max(0.0, min(s0, L - MAXL))
                t0 = round(t0, 2)
                Lw = MAXL
                path = f"{OUT}/audio_original/{stem}_q{k}_{t0:.2f}.wav"
                sf.write(path, x[int(t0 * SR): int(t0 * SR) + int(MAXL * SR)], SR)
                n_win += 1
            gt = [[round(max(0.0, s - t0), 2), round(min(Lw, e - t0), 2)] for s, e in iv]
            gt = [g for g in gt if g[1] - g[0] > 0]
            if not gt:
                n_drop += 1
                continue
            rows.append(dict(qid=f"{stem}_{k}", bench="tacos", setting="original", query=text, vid=stem, audio=path,
                             L=round(Lw, 3), gt=gt, clip_start=t0, clip_branch="whole" if L <= MAXL else "window120"))
    with open(f"{OUT}/original.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    g = rows
    st = dict(queries=len(g), recordings=len({r["vid"] for r in g}), windowed_queries=n_win, dropped_queries=n_drop,
              missing_audio=n_missing, multi_interval_share=float(np.mean([len(r["gt"]) > 1 for r in g])),
              gt_len_median=float(np.median([e - s for r in g for s, e in r["gt"]])),
              gt_len_lt5_share=float(np.mean([e - s < 5 for r in g for s, e in r["gt"]])),
              L_median=float(np.median([r["L"] for r in g])))
    json.dump(st, open(f"{OUT}/stats.json", "w"), indent=1)
    print(st)


if __name__ == "__main__":
    main()
