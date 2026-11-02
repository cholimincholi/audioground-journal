#!/usr/bin/env python
"""strongcrop_lead: event-level targets from AudioSet-Strong for a share of the grounding instructions (R4-1, R5-M3).

Starting from the final construction (variants/lead), a fraction FRAC of the training grounding instructions is changed:
the target segment is replaced by an AudioSet-Strong event cropped from a training clip, i.e. the inserted audio is only
the annotated event (0.5-8 s), and the query is the event label ("Identify when \"dog barking\" is heard ...").
Because the inserted segment is exactly the event, the answer boundary is event-level and still deterministic.
Everything else of the row is kept: fillers, leading segment, junctions, instruction template, response format; the
segments after the target are shifted by the change of the target length.

Eligible events: 0.5-8 s, fully inside the clip, not overlapped by another labelled event of the clip except long
background events (>= 80 % of the clip). Clips used in the validation split are excluded. A candidate event is rejected
if its label is similar to the caption of any filler of the row (SBERT all-MiniLM-L6-v2 cosine >= tau = 0.33, the
pairing threshold of AudioGround-IT), so that the queried sound is not also present elsewhere.
The validation split is unchanged (checkpoint selection on the same data as all other variants).

usage: python data/scripts/make_strongcrop.py [--frac 0.2]
out  : variants/strongcrop_lead/{train,val}.jsonl, audio_store/strongcrop/*.wav, variants/strongcrop_lead/stats.json
"""
import argparse
import collections
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_from_meta as RM  # noqa: E402

R = "data"
SRC, DST = f"{R}/variants/lead", f"{R}/variants/strongcrop_lead"
AUD = f"{R}/audio_store/strongcrop"
TAU = 0.33


def label_phrase(name):
    return name.split(",")[0].strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2, help="share of training grounding instructions to convert")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    os.makedirs(DST, exist_ok=True); os.makedirs(AUD, exist_ok=True)
    rng = np.random.default_rng(20261010)
    train = [json.loads(l) for l in open(f"{SRC}/train.jsonl")]
    val_lines = open(f"{SRC}/val.jsonl").read()
    val_ids = {s["audiocap_id"] for l in val_lines.splitlines() for s in json.loads(l)["meta"]["segments"]}
    pools = pd.read_parquet(f"{R}/pools/pools.parquet")
    st = pd.read_parquet(f"{R}/pools/strong_events.parquet")
    r2a = dict(zip(range(len(pools)), pools.audiocap_id.values.tolist()))
    dur = dict(zip(range(len(pools)), pools.dur.values.tolist()))
    ev = collections.defaultdict(list)
    for e in st.itertuples():
        ev[e.row].append((float(e.on), float(e.off), e.name))
    cands = []   # (audiocap_id, on, off, phrase)
    for row, E in ev.items():
        aid = r2a[row]
        if aid in val_ids:
            continue
        L = dur[row]
        for on, off, nm in E:
            if not (0.5 <= off - on <= 8.0) or on < 0 or off > L:
                continue
            if any(n2 != nm and not (o2 >= off or f2 <= on) and (f2 - o2) < 0.8 * L for o2, f2, n2 in E):
                continue
            cands.append((aid, on, off, label_phrase(nm)))
    print(f"eligible Strong events: {len(cands)} from {len({c[0] for c in cands})} clips")

    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    phrases = sorted({c[3] for c in cands})
    pe = dict(zip(phrases, sb.encode(phrases, normalize_embeddings=True, batch_size=256)))
    cap = dict(zip(pools.audiocap_id.values.tolist(), pools.caption.values.tolist()))

    g_idx = [i for i, r in enumerate(train) if r["task"] == "grounding"
             and sum(s["role"] == "target" for s in r["meta"]["segments"]) == 1]
    n_target = int(round(a.frac * sum(r["task"] == "grounding" for r in train)))
    order = rng.permutation(g_idx)
    changed, used = [], collections.Counter()
    num = re.compile(r"\d+(?:\.\d+)?")
    for i in order:
        if len(changed) >= n_target:
            break
        r = train[i]; m = r["meta"]
        fill_caps = [cap[s["audiocap_id"]] for s in m["segments"] if s["role"] != "target"]
        fe = sb.encode(fill_caps, normalize_embeddings=True) if fill_caps else np.zeros((0, 384))
        for _ in range(30):
            aid, on, off, ph = cands[rng.integers(len(cands))]
            if used[(aid, on)] >= 2:          # each event at most twice
                continue
            if len(fe) and float((fe @ pe[ph]).max()) >= TAU:
                continue
            break
        else:
            continue
        used[(aid, on)] += 1
        segs = [dict(s) for s in m["segments"]]
        k = next(j for j, s in enumerate(segs) if s["role"] == "target")
        old = segs[k]["end"] - segs[k]["start"]
        new = off - on
        d = new - old
        segs[k].update(audiocap_id=aid, crop=[round(on, 4), round(off, 4)], caption=ph, end=segs[k]["start"] + new,
                       strong_event=True)
        for s in segs[k + 1:]:
            s["start"] += d; s["end"] += d
        s0, e0 = segs[k]["start"], segs[k]["end"]
        q_old = re.search(r'"([^"]+)"', r["instruction"])
        inst = r["instruction"].replace(q_old.group(0), f'"{ph}"', 1) if q_old else r["instruction"]
        nums = num.findall(r["response"])
        resp = r["response"]
        if len(nums) >= 2:
            resp = resp.replace(nums[0], f"{s0:.1f}", 1)
            tail = resp.split(f"{s0:.1f}", 1)
            resp = tail[0] + f"{s0:.1f}" + tail[1].replace(nums[1], f"{e0:.1f}", 1)
        nm = dict(m, segments=segs, duration=round(m["duration"] + d, 3), gt_exact=[[round(s0, 4), round(e0, 4)]],
                  span=round(new, 3), strongcrop=dict(audiocap_id=aid, on=on, off=off, label=ph, replaced_len=round(old, 3)))
        nr = dict(r, instruction=inst, response=resp, gt_intervals=[[round(s0, 1), round(e0, 1)]], meta=nm,
                  audio=f"{AUD}/{m['id']}.wav")
        train[i] = nr
        changed.append(i)
    print(f"converted {len(changed)} of {n_target} requested grounding instructions")

    with Pool(a.workers, initializer=RM.init) as p:
        res = p.map(RM.work, [(train[i], False) for i in changed], chunksize=4)
    print(collections.Counter(k for k, _ in res))
    with open(f"{DST}/train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    open(f"{DST}/val.jsonl", "w").write(val_lines)
    spans = [train[i]["meta"]["span"] for i in changed]
    stats = dict(converted=len(changed), grounding_total=sum(r["task"] == "grounding" for r in train), train_total=len(train),
                 share_of_grounding=len(changed) / sum(r["task"] == "grounding" for r in train), share_of_all=len(changed) / len(train),
                 event_len_median=float(np.median(spans)), event_len_mean=float(np.mean(spans)),
                 distinct_labels=len({train[i]["meta"]["strongcrop"]["label"] for i in changed}))
    json.dump(stats, open(f"{DST}/stats.json", "w"), indent=1)
    print(stats)
    for i in changed[:3]:
        print(train[i]["instruction"], "->", train[i]["response"], train[i]["gt_intervals"])


if __name__ == "__main__":
    main()
