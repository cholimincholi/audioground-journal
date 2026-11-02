#!/usr/bin/env python
"""Render missing audio for an existing variant JSONL from its row metadata (2026-10-08 server rebuild).

synth.py re-run on a new machine reproduces >99.8% of rows bit-exactly, but a few filler choices flip
(float-level SBERT differences). To keep the snapshot JSONL (which E0 seed 0 was trained/evaluated on) as the
single source of truth, this script renders every row whose `audio` file is missing from meta.segments
(audiocap_id, crop, start), meta.junction_len and meta.background, using the same mixing code as synth.render.

  python render_from_meta.py <variant.jsonl> [--check N]   # --check: re-render N existing rows and compare
"""
import argparse, json, os, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth  # noqa: E402

SR = synth.SR
R = "data"


def init():
    pools = pd.read_parquet(f"{R}/pools/pools.parquet")
    synth.G.update(cfg=dict(root=R), pools=pools)
    synth.G["a2r"] = dict(zip(pools.audiocap_id.values.tolist(), range(len(pools))))


def render_row(r):
    m = r["meta"]
    a2r = synth.G["a2r"]
    lay = dict(segs=[dict(row=a2r[s["audiocap_id"]], a=s["crop"][0], b=s["crop"][1]) for s in m["segments"]])
    starts = [s["start"] for s in m["segments"]]
    j = m["junction"]
    mode = j["mode"] if isinstance(j, dict) else str(j)
    jcfg = {**synth.DEFAULT_CFG["junction"], **(j if isinstance(j, dict) else {}), "mode": mode}
    bg = None
    if m.get("background"):
        b = m["background"]
        bg = dict(rows=[a2r[x] for x in b["audiocap_ids"]], snr=b["snr_db"], offset=b["offset"])
    return synth.render(lay, starts, m["duration"], m["junction_len"], jcfg, bg)


def work(args):
    r, check = args
    y = render_row(r)
    if check:
        x, _ = sf.read(r["audio"], dtype="int16")
        if len(x) != len(y):
            return ("lenmismatch", len(x) - len(y))
        d = np.abs(x.astype(np.int32) - y.astype(np.int32))
        return ("exact" if d.max() == 0 else "close" if d.max() <= 2 else "diff", int(d.max()))
    os.makedirs(os.path.dirname(r["audio"]), exist_ok=True)
    tmp = r["audio"] + f".tmp{os.getpid()}.wav"
    sf.write(tmp, y, SR, subtype="PCM_16")
    os.replace(tmp, r["audio"])
    return ("written", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    rows = [json.loads(l) for p in a.jsonl for l in open(p)]
    if a.check:
        ex = [r for r in rows if os.path.exists(r["audio"])]
        idx = np.random.default_rng(0).choice(len(ex), min(a.check, len(ex)), replace=False)
        jobs = [(ex[i], True) for i in idx]
    else:
        seen, jobs = set(), []
        for r in rows:
            if r["audio"] not in seen and not os.path.exists(r["audio"]):
                seen.add(r["audio"]); jobs.append((r, False))
    with Pool(a.workers, initializer=init) as p:
        res = p.map(work, jobs, chunksize=4)
    from collections import Counter
    print(Counter(k for k, _ in res), "max sample diff:", max([v for _, v in res], default=0))


if __name__ == "__main__":
    main()
