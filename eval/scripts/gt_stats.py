"""GT duration / position distributions per benchmark x setting -> results/gt_stats.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agl_eval import EVAL_ROOT  # noqa: E402
from agl_eval.data import BENCHES, load  # noqa: E402

DUR_BINS = [0, 5, 10, 20, 30, 1e9]  # [lo, hi)
CEN_BINS = [0, 10, 20, 30, 40, 50, 60.0001]


def hist(x, bins):
    h, _ = np.histogram(x, bins=bins)
    return {f"{bins[i]:g}-{bins[i+1]:g}": int(h[i]) for i in range(len(h))}


def main():
    out = {}
    for b in BENCHES:
        for s in ("original", "clipped"):
            R = load(b, s)
            W = [(r, w) for r in R for w in r["gt"]]
            d = np.array([w[1] - w[0] for _, w in W])
            # also float-robust durations (round to 1e-6) for bin edges like 30.000000000000004
            dr = np.round(d, 6)
            c = np.array([(w[0] + w[1]) / 2 for _, w in W])
            on = np.array([w[0] / r["L"] for r, w in W])
            L = np.array([r["L"] for r in R])
            first = np.array([r["gt"][0][1] - r["gt"][0][0] for r in R])
            out[f"{b}/{s}"] = dict(
                n_queries=len(R), n_recordings=len({r["vid"] for r in R}), n_gt_intervals=len(W),
                gt_per_query=dict(zip(*map(lambda a: a.tolist(), np.unique([len(r["gt"]) for r in R], return_counts=True)))),
                L_min=float(L.min()), L_median=float(np.median(L)), L_max=float(L.max()),
                dur_mean=float(d.mean()), dur_median=float(np.median(d)), dur_min=float(d.min()), dur_max=float(d.max()),
                dur_hist_raw_float=hist(d, DUR_BINS), dur_hist_rounded=hist(dr, DUR_BINS),
                n_dur_gt30_float=int((d > 30).sum()), n_dur_ge30_rounded=int((dr >= 30).sum()),
                first_interval_dur_hist=hist(np.round(first, 6), DUR_BINS),
                center_hist=hist(c, CEN_BINS),
                norm_onset_hist=hist(on, [0, .2, .4, .6, .8, 1.0001]),
                frac_touch_start=float(np.mean([w[0] <= 0.05 for _, w in W])),
                frac_touch_end=float(np.mean([w[1] >= r["L"] - 0.05 for r, w in W])),
            )
            print(b, s, json.dumps(out[f"{b}/{s}"]))
    json.dump(out, open(os.path.join(EVAL_ROOT, "results", "gt_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
