#!/usr/bin/env python
"""Summarize all variants: table stats, layout identity vs default, tau constraint check, strong coverage."""
import json, glob, os
import numpy as np, pandas as pd
R = "data"
E = np.load(f"{R}/emb/emb.npy"); p = pd.read_parquet(f"{R}/pools/pools.parquet"); a2r = dict(zip(p.audiocap_id, range(len(p))))
def load(v):
    out = {}
    for s in ("train", "val"):
        for l in open(f"{R}/variants/{v}/{s}.jsonl"):
            r = json.loads(l); out[r["meta"]["id"]] = r
    return out
lay = lambda r: [(s["audiocap_id"], s["crop"]) for s in r["meta"]["segments"]]
d = load("default")
res = {}
for v in sorted(os.listdir(f"{R}/variants")):
    if v.startswith("_") or not os.path.exists(f"{R}/variants/{v}/stats.json"): continue
    x = load(v); st = json.load(open(f"{R}/variants/{v}/stats.json"))
    same = {t: [0, 0] for t in ["grounding", "duration", "ordering", "frequency"]}
    for k, r in d.items():
        same[r["task"]][1] += 1; same[r["task"]][0] += (k in x and lay(x[k]) == lay(r))
    # tau check: max cosine between any filler and any target clip; min cosine target-vs-composed partner
    mx = []
    for r in x.values():
        tg = [a2r[s["audiocap_id"]] for s in r["meta"]["segments"] if s["role"] == "target"]
        fl = [a2r[s["audiocap_id"]] for s in r["meta"]["segments"] if s["role"] == "filler"]
        if fl: mx.append(float(np.max(E[fl] @ E[list(set(tg))].T)))
    res[v] = dict(table={t: {k: st[t][k] for k in ("samples", "avg_len_s", "avg_query_words", "total_hours")} for t in st if t in same or t == "total"},
                  onset_hist=st["total"]["onset_bin_hist"], len_hist=st["total"]["len_bin_hist"],
                  span_hist={t: st[t]["span_bin_hist"] for t in ("grounding", "duration")},
                  same_layout_as_default={t: f"{a}/{b}" for t, (a, b) in same.items()},
                  same_audio_as_default=int(sum(k in x and x[k]["audio"] == r["audio"] for k, r in d.items())),
                  max_filler_target_sim=dict(max=round(max(mx), 4), p99=round(float(np.percentile(mx, 99)), 4)),
                  strong_status={t: st[t].get("strong_status") for t in same} if "strong_status" in st["total"] else None)
    if v == "bgmix":
        snr = [r["meta"]["background"]["snr_db"] for r in x.values()]
        res[v]["bg_snr"] = dict(min=min(snr), max=max(snr), mean=round(float(np.mean(snr)), 2))
        res[v]["bg_unique_clips"] = len(set(a for r in x.values() for a in r["meta"]["background"]["audiocap_ids"]))
    if v == "strong":
        dd = [(r["gt_intervals"], d[k]["gt_intervals"]) for k, r in x.items()]
        ch = [(g, o) for g, o in dd if g != o]
        shrink = [sum(b - a for a, b in o) - sum(b - a for a, b in g) for g, o in ch]
        res[v]["changed_gt_samples"] = len(ch); res[v]["mean_shrink_s"] = round(float(np.mean(shrink)), 3) if shrink else 0
json.dump(res, open(f"{R}/variants/variants_summary.json", "w"), indent=1)
for v, r in res.items():
    print("=" * 10, v); print(json.dumps({k: r[k] for k in r if k != "table"}))
    for t, s in r["table"].items(): print(f"   {t:10s} {s}")
