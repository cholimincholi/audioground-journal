"""Trivial baselines for every benchmark x {original, clipped}; writes results/trivial_baselines.{json,md}.

Tuning protocol (best fixed interval, 1-s grid, objective R1@0.5 unless noted):
  * Clotho-Moment : tuned on Clotho-Moment VAL (4,918 queries; clipped val for Clipped).
  * UnAV / TUT    : no val split -> (a) leave-one-out on the benchmark itself,
                    (b) transfer of the Clotho-val-tuned interval (absolute seconds, and
                        normalized by recording length).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agl_eval import BENCH_ROOT, EVAL_ROOT  # noqa: E402
from agl_eval import baselines as tb  # noqa: E402
from agl_eval.data import BENCHES, load  # noqa: E402
from agl_eval.metrics import TOL, bootstrap_ci  # noqa: E402

B_BOOT = 10000


def clotho_val():
    recs = []
    for l in open(os.path.join(BENCH_ROOT, "clotho_moment", "clotho_moment_val_release.jsonl")):
        a = json.loads(l)
        recs.append(dict(qid=a["qid"], gt=[list(map(float, w)) for w in a["relevant_windows"]], L=float(a["duration"])))
    return recs


def ci_pair(M):
    return {t: [round(x, 2) for x in bootstrap_ci((M >= t - TOL).astype(float), None, B_BOOT, 0)] for t in (0.5, 0.7)}


def main():
    out = {}
    val = {"original": clotho_val()}
    val["clipped"] = tb.clip_records(val["original"])
    grid = tb.candidate_grid(60.0, 1.0)
    grid30 = tb.candidate_grid(30.0, 1.0)
    ngrid = tb.candidate_grid(1.0, 1 / 60)
    tuned = {}
    for setting in ("original", "clipped"):
        g = grid if setting == "original" else grid30
        for obj in (0.5, 0.7):
            iv, sc = tb.tune_best_fixed(val[setting], g, obj)
            niv, nsc = tb.tune_best_fixed(val[setting], ngrid if setting == "original" else ngrid, obj, normalized=True)
            tuned[(setting, obj)] = (iv, niv)
            out[f"clotho_val_tuned/{setting}/obj{obj}"] = dict(interval=iv.tolist(), val_r1=sc,
                                                               norm_interval=niv.tolist(), norm_val_r1=nsc)
    for bench in BENCHES:
        for setting in ("original", "clipped"):
            R = load(bench, setting)
            key = f"{bench}/{setting}"
            res = {"n": len(R)}
            maxL = max(r["L"] for r in R)
            # 1) Clotho-val-tuned fixed interval (absolute + normalized)
            for obj in (0.5, 0.7):
                iv, niv = tuned[(setting, obj)]
                r1, M = tb.r1_of_interval(R, iv)
                res[f"valtuned_abs_obj{obj}"] = dict(interval=iv.tolist(), r1=r1, ci=ci_pair(M))
                r1n, Mn = tb.r1_of_interval(R, niv, normalized=True)
                res[f"valtuned_norm_obj{obj}"] = dict(interval=[round(x, 4) for x in niv.tolist()], r1=r1n, ci=ci_pair(Mn))
            # 2) LOO best fixed on the benchmark itself (for small benchmarks; also for clotho as reference)
            if bench != "clotho_moment":
                g = tb.candidate_grid(np.ceil(maxL), 1.0)
                for obj in (0.5, 0.7):
                    r1, chosen = tb.loo_best_fixed(R, g, obj)
                    from collections import Counter
                    res[f"loo_abs_obj{obj}"] = dict(r1=r1, most_common_choice=[list(map(float, k)) for k, _ in Counter(chosen).most_common(1)])
                    r1n, chosen_n = tb.loo_best_fixed(R, ngrid, obj, normalized=True)
                    res[f"loo_norm_obj{obj}"] = dict(r1=r1n)
                # oracle (in-sample) best fixed for reference = upper bound of any fixed interval
                iv, sc = tb.tune_best_fixed(R, g, 0.5)
                r1, M = tb.r1_of_interval(R, iv)
                res["oracle_insample_abs"] = dict(interval=iv.tolist(), r1=r1)
            # 3) centered fixed lengths
            for ln in (10, 20, 26, 30):
                r1, M = tb.centered(R, ln)
                res[f"centered_{ln}s"] = dict(r1=r1, ci=ci_pair(M))
            if setting == "clipped":
                r1, M = tb.whole(R)
                res["whole_window"] = dict(r1=r1, ci=ci_pair(M))
            r1, M = tb.whole(R)
            res["whole_recording"] = dict(r1=r1)
            # 4) random intervals with a duration prior (model-specific version in make_tables.py)
            gtd = [b - a for r in R for a, b in r["gt"]]
            res["random_gtdur"] = tb.random_with_durations(R, gtd)
            res["random_dur26.5"] = tb.random_with_durations(R, [26.5])
            res["random_dur10"] = tb.random_with_durations(R, [10.0])
            out[key] = res
            print(key, {k: (v.get("r1") if isinstance(v, dict) and "r1" in v else v) for k, v in res.items() if k != "n"})
    os.makedirs(os.path.join(EVAL_ROOT, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(EVAL_ROOT, "results", f"trivial_baselines_tol{TOL:g}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
