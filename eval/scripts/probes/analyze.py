"""Aggregate probe 1 (shift) and probe 2 (onset prior) outputs -> eval/results/probes.{json,md tables}.
  python eval/scripts/probes/analyze.py
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "eval")
from agl_eval.parser import parse_first  # noqa: E402
from agl_eval.metrics import max_iou, TOL  # noqa: E402

RUNS = "runs"
CKPTS = [("v1", "E0_main/1"), ("v1", "E0_main/2"), ("v1", "E7_salmonn_ft/0"),
         ("v2", "E0L_main/0"), ("v2", "E0L_main/1"), ("v2", "E0L_main/2"), ("v2", "E7L_salmonn_ft/0")]
DELTAS = [0, 3, 7, 15]


def rj(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def near(x, pos, tol=0.5):
    return any(abs(x - p) <= tol for p in pos)


def probe1(run):
    d = f"{RUNS}/{run}/preds/probe_shift"
    recs = {}
    for D in DELTAS:
        r = rj(f"{d}/shift_d{D}.jsonl")
        if r is None:
            return None
        recs[D] = {x["id"]: x for x in r}
    on0 = {i: (parse_first(x["raw_output"]) or [None])[0] for i, x in recs[0].items()}
    rows = []
    for D in DELTAS:
        R = recs[D]; n = len(R)
        preds = {i: parse_first(x["raw_output"]) for i, x in R.items()}
        par = [i for i in R if preds[i] is not None]
        ious = np.array([max_iou(preds[i], R[i]["gt"]) for i in R])
        on = np.array([preds[i][0] for i in par])
        sh = D + 0.4 if D else 0.0
        old = [near(preds[i][0], R[i]["orig_seg_starts"]) for i in par]
        new = [near(preds[i][0], [s + sh for s in R[i]["orig_seg_starts"]]) for i in par]
        both = [i for i in par if on0.get(i) is not None]
        mv = np.array([preds[i][0] - on0[i] for i in both])
        rows.append(dict(delta=D, n=n, parse=100 * len(par) / n,
                         r05=100 * np.mean(ious >= 0.5 - TOL), r07=100 * np.mean(ious >= 0.7 - TOL),
                         pct_zero=100 * np.mean(on == 0.0), mean_on=float(on.mean()),
                         mean_move=float(mv.mean()) if D else 0.0, true_shift=sh,
                         pct_follow=100 * float(np.mean(np.abs(mv - sh) <= 1.0)) if D else float("nan"),
                         pct_stay=100 * float(np.mean(np.abs(mv) <= 1.0)) if D else float("nan"),
                         pct_oldgrid=100 * float(np.mean(old)), pct_newgrid=100 * float(np.mean(new)),
                         pct_oldgrid_only=100 * float(np.mean([o and not w for o, w in zip(old, new)]))))
    return rows


def probe2(run):
    d = f"{RUNS}/{run}/preds/probe_prior"
    out = {}
    for name in ("clotho600", "val325"):
        r = rj(f"{d}/{name}.jsonl")
        if r is None:
            return None
        lt1 = np.array([x["p_lt1"] for x in r]); eq0 = np.array([x["p_eq0"] for x in r])
        far = np.array([x["gt_onset"] >= 10 for x in r])
        out[name] = dict(n=len(r), n_far=int(far.sum()), p_lt1=float(lt1.mean()), p_eq0=float(eq0.mean()),
                         p_lt1_far=float(lt1[far].mean()), p_eq0_far=float(eq0[far].mean()),
                         gt_lt1=float(np.mean([x["gt_onset"] < 1 for x in r])),
                         top0=float(np.mean([x["top_first"] == "0" for x in r])))
    return out


def train_freq():
    res = {}
    for v in ("default", "lead"):
        n = g = a0 = g0 = 0
        for l in open(f"data/variants/{v}/train.jsonl"):
            r = json.loads(l); m = re.search(r"\d+(?:\.\d+)?", r["response"]); n += 1
            z = bool(m and m.group().startswith("0.")); a0 += z
            if r["task"] == "grounding":
                g += 1; g0 += z
        res[v] = dict(all=100 * a0 / n, grounding=100 * g0 / g)
    return res


def main():
    res = dict(probe1={}, probe2={}, train_freq=train_freq())
    for v, run in CKPTS:
        res["probe1"][run] = probe1(run)
        res["probe2"][run] = probe2(run)
    json.dump(res, open("eval/results/probes.json", "w"), indent=1)
    L = ["| data | ckpt | Δ (s) | parse % | R1@0.5 | R1@0.7 | % onset=0.0 | mean onset | mean Δonset (true) | % followed shift (±1 s) | % onset unmoved (±1 s) | % on old grid | % on shifted grid |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, run in CKPTS:
        for r in res["probe1"][run] or []:
            D = r["delta"]
            L.append(f"| {v} | {run} | {D} | {r['parse']:.1f} | {r['r05']:.1f} | {r['r07']:.1f} | {r['pct_zero']:.1f} | "
                     f"{r['mean_on']:.1f} | " + (f"{r['mean_move']:+.1f} ({r['true_shift']:.1f})" if D else "—") + " | "
                     + (f"{r['pct_follow']:.1f} | {r['pct_stay']:.1f}" if D else "— | —")
                     + f" | {r['pct_oldgrid']:.1f} | {r['pct_newgrid']:.1f} |")
    L += ["", "| data | ckpt | Clotho P(on<1s) | Clotho P(on=0.0) | Clotho P(on<1s), GT≥10s | Clotho P(on=0.0), GT≥10s | val P(on<1s) | val P(on=0.0) | val P(on<1s), GT≥10s | val P(on=0.0), GT≥10s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for v, run in CKPTS:
        p = res["probe2"][run]
        if p is None:
            continue
        c, w = p["clotho600"], p["val325"]
        L.append(f"| {v} | {run} | {c['p_lt1']:.3f} | {c['p_eq0']:.3f} | {c['p_lt1_far']:.3f} | {c['p_eq0_far']:.3f} | "
                 f"{w['p_lt1']:.3f} | {w['p_eq0']:.3f} | {w['p_lt1_far']:.3f} | {w['p_eq0_far']:.3f} |")
    print("\n".join(L))
    print(json.dumps(res["train_freq"]))
    print({k: (v["clotho600"]["n_far"], v["val325"]["n_far"], v["clotho600"]["gt_lt1"], v["val325"]["gt_lt1"])
           for k, v in res["probe2"].items() if v})
    open("eval/results/probes_tables.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
