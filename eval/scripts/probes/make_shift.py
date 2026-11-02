"""Probe 1 data: time-shift test on the 325 v1-val grounding samples.

For every sample and shift D in {0,3,7,15} s:  wav' = [filler prefix of D s] + [0.4 s silence] + [original wav]
(D=0: original wav unchanged).  GT shifted by D+0.4.
Filler: ONE foreground AudioCaps clip from data/pools (seeded per sample), not used in the sample, SBERT cosine
< 0.33 to every caption (=pool row) of the sample. Clips are <= 10 s, so for D=15 the prefix is
[10.0 s of clip A] + [0.4 s silence] + [4.6 s crop of a 2nd clip B] (B satisfies the same constraints and is
also < 0.33 to A) -- i.e. one extra junction, like the training data. For D=3/7 a random crop of A.
Output: data/probes/shift/wav/<id>_d<D>.wav, data/probes/shift/shift_d<D>.jsonl
Run with the data-generation Python environment.
"""
import json
import os

import numpy as np
import pandas as pd
import soundfile as sf

R = "data"
OUT = f"{R}/probes/shift"
SR = 16000
DELTAS = [0, 3, 7, 15]
GAP = 0.4
TAU = 0.33


def main():
    os.makedirs(f"{OUT}/wav", exist_ok=True)
    pools = pd.read_parquet(f"{R}/pools/pools.parquet")
    E = np.load(f"{R}/emb/emb.npy").astype(np.float32)
    row_of = {int(a): i for i, a in enumerate(pools.audiocap_id.values)}
    fg = np.where(~pools.is_background.values)[0]
    dur = pools.dur.values
    mm = np.memmap(f"{R}/clips/clips_16k.int16", dtype=np.int16, mode="r")

    def clip(r):
        o, n = int(pools.offset.values[r]), int(pools.n_samples.values[r])
        return np.asarray(mm[o:o + n])

    rows = [json.loads(l) for l in open(f"{R}/variants/default/val.jsonl")]
    rows = [r for r in rows if r["task"] == "grounding"]
    assert len(rows) == 325
    outs = {d: [] for d in DELTAS}
    for k, r in enumerate(rows):
        rid = r["meta"]["id"]
        rng = np.random.default_rng([20261009, k])
        used = [row_of[int(s["audiocap_id"])] for s in r["meta"]["segments"]]
        M = E[used]

        def pick(extra=()):
            MM = np.concatenate([M, E[list(extra)]]) if extra else M
            while True:
                c = int(fg[rng.integers(0, len(fg))])
                if c in used or c in extra or dur[c] < 9.99:   # full 10-s clips so every D (<=10) fits in one clip
                    continue
                if np.max(MM @ E[c]) < TAU:
                    return c
        A = pick()
        B = pick((A,))
        a, b = clip(A), clip(B)
        orig, sr = sf.read(r["audio"], dtype="int16")
        assert sr == SR and orig.ndim == 1
        L0 = len(orig) / SR
        gap = np.zeros(int(round(GAP * SR)), np.int16)
        for d in DELTAS:
            fill = []
            if d == 0:
                path, y, sh = r["audio"], None, 0.0
            else:
                if d <= 10:
                    n = d * SR; st = int(rng.integers(0, len(a) - n + 1))
                    pre = a[st:st + n]
                    fill = [dict(audiocap_id=int(pools.audiocap_id.values[A]), crop=[st / SR, (st + n) / SR])]
                else:
                    nb = int(round((d - 10 - GAP) * SR)); st = int(rng.integers(0, len(b) - nb + 1))
                    pre = np.concatenate([a[:10 * SR], gap, b[st:st + nb]])
                    fill = [dict(audiocap_id=int(pools.audiocap_id.values[A]), crop=[0.0, 10.0]),
                            dict(audiocap_id=int(pools.audiocap_id.values[B]), crop=[st / SR, (st + nb) / SR])]
                assert len(pre) == d * SR
                y = np.concatenate([pre, gap, orig])
                sh = d + GAP
                path = f"{OUT}/wav/{rid}_d{d}.wav"
                sf.write(path, y, SR, subtype="PCM_16")
            gt = [[round(s + sh, 4), round(e + sh, 4)] for s, e in r["gt_intervals"]]
            seg_starts = [s["start"] for s in r["meta"]["segments"]]
            outs[d].append(dict(id=rid, audio=path, instruction=r["instruction"], gt=gt, L=round(L0 + sh, 4),
                                delta=d, shift=sh, orig_gt=r["gt_intervals"], orig_seg_starts=seg_starts,
                                filler=fill))
    for d in DELTAS:
        with open(f"{OUT}/shift_d{d}.jsonl", "w") as f:
            for o in outs[d]:
                f.write(json.dumps(o) + "\n")
    print("done", {d: len(v) for d, v in outs.items()})


if __name__ == "__main__":
    main()
