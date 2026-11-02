"""Parse raw outputs and compute parse-success rate + R1@{0.5,0.7} (paper protocol).

Parser: first timestamp pair joined by 'to', '-', or 'and'; reversed if end < start. R1 over parsed responses
(paper) and also over all responses (unparsed = miss) for transparency. Multiple GT intervals -> max IoU.

GT is read from the first present key among --gt_keys: a [s, e] pair or a list of pairs.
  python tools/score.py preds.jsonl [--gt_keys spans,timestamps,gt,relevant_windows]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audioground.data import parse_first_span


def as_intervals(g):
    if g is None:
        return None
    if isinstance(g, dict):
        g = [g.get("start"), g.get("end")]
    if len(g) == 2 and all(isinstance(x, (int, float)) for x in g):
        return [tuple(map(float, g))]
    return [tuple(map(float, x)) for x in g]


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def score(path, gt_keys):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows); parsed = 0; hits = {0.5: 0, 0.7: 0}; no_gt = 0
    for r in rows:
        gt = None
        for k in gt_keys:
            if k in r and r[k] is not None:
                gt = as_intervals(r[k]); break
        if gt is None and "start" in r and "end" in r:
            gt = [(float(r["start"]), float(r["end"]))]
        sp = parse_first_span(r.get("raw_output", ""))
        if sp is None:
            continue
        parsed += 1
        if gt is None:
            no_gt += 1; continue
        best = max(iou(sp, g) for g in gt)
        for th in hits:
            hits[th] += best >= th
    res = {"file": path, "n": n, "parsed": parsed, "parse_rate": parsed / max(n, 1), "no_gt": no_gt}
    for th, h in hits.items():
        res[f"R1@{th}_parsed"] = 100 * h / max(parsed - no_gt, 1)
        res[f"R1@{th}_all"] = 100 * h / max(n - no_gt, 1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--gt_keys", default="spans,timestamps,gt,relevant_windows,timestamp,interval")
    a = ap.parse_args()
    for f in a.files:
        print(json.dumps(score(f, a.gt_keys.split(","))))
