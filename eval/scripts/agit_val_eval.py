"""Per-task evaluation on the AudioGround-IT held-out validation split (R2-4: frequency task; R1-3).

usage: python eval/scripts/agit_val_eval.py runs/E0_main/1/preds/agitval/agit_val_all.jsonl [more ...] [--B 2000]

grounding : R1@0.5/0.7 and mean IoU of the first interval
duration  : R1@0.5/0.7, mean |pred span - GT span| (s)
ordering  : order accuracy (onset order of the two intervals, in mention order, matches GT), mean IoU per event
frequency : count accuracy (#intervals == k), count MAE, set-F1@0.5 (one-to-one greedy matching), GT recall@0.5
All tasks: parse rate (extended parser, all intervals). Unparseable = miss / wrong. 95% bootstrap CI on the headline metric.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agl_eval.parser import parse_all  # noqa: E402


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def set_match(pred, gt, thr):
    used, tp = set(), 0
    for g in gt:
        best, bj = 0.0, None
        for j, p in enumerate(pred):
            if j in used:
                continue
            v = iou(p, g)
            if v > best:
                best, bj = v, j
        if bj is not None and best >= thr - 1e-6:
            used.add(bj); tp += 1
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1, rec


def ci(x, B, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (B, len(x)), replace=True).mean(1)
    return 100 * x.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def evaluate(path, B):
    rows = [json.loads(l) for l in open(path)]
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        t = r["task"]
        gt = [list(map(float, g)) for g in r["gt"]]
        pr = [sorted(map(float, p[:2])) for p in parse_all(r.get("raw_output") or "", "extended")]
        d = per[t]
        d["parsed"].append(float(bool(pr)))
        if t in ("grounding", "duration"):
            v = iou(pr[0], gt[0]) if pr else 0.0
            d["iou"].append(v); d["r05"].append(float(v >= 0.5 - 1e-6)); d["r07"].append(float(v >= 0.7 - 1e-6))
            if pr:
                d["span_err"].append(abs((pr[0][1] - pr[0][0]) - (gt[0][1] - gt[0][0])))
        elif t == "ordering":
            ok = len(pr) >= 2 and ((pr[0][0] < pr[1][0]) == (gt[0][0] < gt[1][0]))
            d["order_acc"].append(float(ok))
            d["iou"].append(np.mean([iou(pr[i], gt[i]) if len(pr) > i else 0.0 for i in range(2)]))
        elif t == "frequency":
            k = len(gt)
            d["count_acc"].append(float(len(pr) == k)); d["count_mae"].append(abs(len(pr) - k))
            f1, rec = set_match(pr, gt, 0.5)
            d["setf1"].append(f1); d["gt_recall"].append(rec)
            d["k"].append(k)
    out = {}
    for t, d in per.items():
        o = {"n": len(d["parsed"]), "parse_rate": 100 * np.mean(d["parsed"])}
        head = {"grounding": "r05", "duration": "r05", "ordering": "order_acc", "frequency": "count_acc"}[t]
        m, lo, hi = ci(d[head], B)
        o[head] = m; o[head + "_ci"] = [lo, hi]
        for key in ("r07", "iou", "setf1", "gt_recall"):
            if d.get(key):
                o[key] = 100 * float(np.mean(d[key]))
        if d.get("span_err"):
            o["span_err_s"] = float(np.mean(d["span_err"]))
        if d.get("count_mae"):
            o["count_mae"] = float(np.mean(d["count_mae"]))
            ks = np.array(d["k"]); acc = np.array(d["count_acc"])
            o["count_acc_by_k"] = {int(k): 100 * float(acc[ks == k].mean()) for k in sorted(set(ks.tolist()))}
        out[t] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    for p in a.files:
        print(p)
        print(json.dumps(evaluate(p, a.B), indent=1))


if __name__ == "__main__":
    main()
