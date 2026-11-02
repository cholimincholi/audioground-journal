#!/usr/bin/env python
"""Turn any runs/ folder into the paper tables.

Discovers
  baselines : <runs>/baselines/<model>/<bench>_<setting>[.<prompt_id>][.shardIofK].jsonl
  our exps  : <runs>/<exp>/<seed>/preds/<tag>/<bench>_<setting>.jsonl  (row = <exp>, or <exp>@<tag> if tag != default)
              also <runs>/<exp>/<seed>/<bench>_<setting>.jsonl and <runs>/<exp>/<bench>_<setting>.jsonl (seed "0")
  skipped   : <runs>/{logs,_smoke,_queue,_bench,_*}; metrics.jsonl / scores.json are not prediction files
  (files with ".limitN" and the "_smoke" folder are ignored)
Record format (one JSON per line): a query id under qid|id|query_id, and either a text
response under response|output|prediction|pred_text|text|answer|generation, or an
already-parsed interval under pred ([s,e]) or pred_relevant_windows ([[s,e,(score)],...]).

Produces (in --out, default eval/results/tables):
  main_table_{old,miss}.{md,tex}  R1@0.5/0.7, all benches x {Original,Clipped}; exps mean±std over seeds;
                                  trivial-baseline rows appended
  parse_rates.md                  parse success rate per system/bench/setting (paper parser; + extended)
  entries.csv / entries.json      every (system, seed, bench, setting) entry: both protocols, 95% CIs
                                  (paired bootstrap, 10k), parse rate, diagnostics
  pvalues.csv                     paired bootstrap p-values for all system pairs (first seed of exps)
  tut_multi_interval.md           set-F1 / mean-over-GT / union-IoU (parse_all) on TUT
  breakdown_position.md           Clotho-Moment R1 by GT-center bin (paper Table), both protocols
  breakdown_duration.md           Clotho-Moment R1 + mean GT/pred duration by GT-duration bin
  diagnostics.md                  predicted duration / onset stats, unique intervals, onset std
  random_dur_baseline.md          "random interval with the model's duration distribution"
  pred_dur_hist_<bench>_<setting>.png  predicted-duration histograms (+GT)
Usage:
  python experiments/eval/make_tables.py [--runs runs] [--B 10000]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "eval")
from agl_eval import baselines as tb  # noqa: E402
from agl_eval import metrics as M  # noqa: E402
from agl_eval.data import BENCH_PRETTY, BENCHES, SETTINGS, load  # noqa: E402
from agl_eval.parser import parse_all, parse_first  # noqa: E402

FN_RE = re.compile(r"^(?P<bench>clotho_moment|unav100|tut2017)_(?P<setting>original|clipped)"
                   r"(?:\.(?P<pid>(?!shard|limit)[A-Za-z0-9_\-]+))?(?:\.shard\d+of\d+)?(?P<limit>\.limit\d+)?\.jsonl$")
ALIASES = {}
SKIP_TOP = {"logs", "_smoke", "_queue", "_bench"}
QID_KEYS = ("qid", "id", "query_id")
TXT_KEYS = ("raw_output", "response", "output", "prediction", "pred_text", "text", "answer", "generation")
BASELINE_ORDER = ["qwen2_audio", "qwen25_omni", "gama", "desta25", "salmonn", "af2", "timeaudio"]
PRETTY = {"qwen2_audio": "Qwen2-Audio-Instruct", "qwen25_omni": "Qwen2.5-Omni", "gama": "GAMA",
          "desta25": "DeSTA2.5-Audio", "salmonn": "SALMONN", "af2": "AF2", "timeaudio": "TimeAudio"}
THR = (0.5, 0.7)


# ------------------------------------------------------------------ discovery / loading
def discover(runs):
    """returns {(system, kind, seed, prompt_id, bench, setting): [files]}"""
    found = defaultdict(list)
    seen = {}  # realpath -> first path (symlinked seed dirs, e.g. E1/E2/E5 -> E0_main/0, are aliases)
    # non-symlinked paths first so the real owner keeps the row
    paths = sorted(glob.glob(os.path.join(runs, "**", "*.jsonl"), recursive=True),
                   key=lambda q: (os.path.realpath(q) != os.path.abspath(q), q))
    for p in paths:
        rp = os.path.realpath(p)
        if rp in seen:
            ALIASES.setdefault(os.path.relpath(p, runs).split(os.sep)[0], os.path.relpath(seen[rp], runs).split(os.sep)[0])
            continue
        seen[rp] = p
        rel = os.path.relpath(p, runs).split(os.sep)
        if "_smoke" in rel or any(x.startswith(".") for x in rel):
            continue
        m = FN_RE.match(rel[-1])
        if not m or m.group("limit"):
            continue
        if rel[0] in SKIP_TOP or rel[0].startswith("_"):
            continue
        if rel[0] == "baselines":
            if len(rel) != 3:
                continue
            system, kind, seed = rel[1], "baseline", "0"
        else:
            kind = "ours"
            if len(rel) == 5 and rel[2] == "preds":      # <exp>/<seed>/preds/<tag>/<file>
                system, seed = rel[0] + ("" if rel[3] == "default" else f"@{rel[3]}"), rel[1]
            elif len(rel) == 3:                           # <exp>/<seed>/<file>
                system, seed = rel[0], rel[1]
            elif len(rel) == 2:                           # <exp>/<file>
                system, seed = rel[0], "0"
            else:
                continue
        found[(system, kind, seed, m.group("pid") or "v1", m.group("bench"), m.group("setting"))].append(p)
    return found


def load_preds(files):
    out = {}
    for p in sorted(files, key=lambda x: (".shard" in x, x)):
        for l in open(p):
            l = l.strip()
            if not l:
                continue
            j = json.loads(l)
            q = next((str(j[k]) for k in QID_KEYS if k in j), None)
            if q is None:
                continue
            if j.get("error") and q in out:
                continue
            out[q] = j
    return out


def to_pred(j, mode="paper"):
    """-> (pred [s,e] or None, preds_all list, text or None)"""
    if "pred" in j and j["pred"] is not None and not isinstance(j["pred"], str):
        p = list(map(float, j["pred"][:2]))
        if p[1] < p[0]:
            p = p[::-1]
        pa = [list(map(float, x[:2])) for x in j.get("preds_all", [p])]
        return p, pa, None
    if "pred_relevant_windows" in j and j["pred_relevant_windows"]:
        w = [sorted(map(float, x[:2])) for x in j["pred_relevant_windows"]]
        return w[0], w, None
    txt = next((j[k] for k in TXT_KEYS if k in j and j[k] is not None), None)
    if txt is None and isinstance(j.get("pred"), str):
        txt = j["pred"]
    if txt is None:
        return None, [], None
    return parse_first(txt, mode), parse_all(txt, mode), txt


def build_records(gt_recs, preds, mode="paper"):
    recs = []
    for g in gt_recs:
        j = preds.get(str(g["qid"]))
        p, pa, txt = to_pred(j, mode) if j is not None else (None, [], None)
        recs.append(dict(qid=g["qid"], gt=g["gt"], L=g["L"], pred=p, preds_all=pa, has_output=j is not None,
                         gt_original=g.get("gt_original", g["gt"])))
    return recs


# ------------------------------------------------------------------ helpers
def fmt(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def msd(vals, nd=2):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "--"
    if len(vals) == 1:
        return fmt(vals[0], nd)
    return f"{np.mean(vals):.{nd}f}±{np.std(vals, ddof=1):.{nd}f}"


def sys_sort_key(s):
    name, kind = s
    return (0 if kind == "baseline" else 1, BASELINE_ORDER.index(name) if name in BASELINE_ORDER else 99, name)


def trivial_rows(gt):
    """{row_name: {(bench,setting): (r05, r07)}}  -- recomputed with the current IoU tolerance."""
    rows = defaultdict(dict)
    val = [dict(gt=[list(map(float, w)) for w in json.loads(l)["relevant_windows"]], L=float(json.loads(l)["duration"]))
           for l in open("eval/benchmarks/clotho_moment/clotho_moment_val_release.jsonl")]
    vals = {"original": val, "clipped": tb.clip_records(val)}
    for setting in SETTINGS:
        grid = tb.candidate_grid(60.0 if setting == "original" else 30.0, 1.0)
        iv, _ = tb.tune_best_fixed(vals[setting], grid, 0.5)
        for b in BENCHES:
            R = gt[(b, setting)]
            r, _ = tb.r1_of_interval(R, iv)
            rows["Best fixed (Clotho-val tuned)"][(b, setting)] = (r[0.5], r[0.7])
            rows["_notes"][setting] = f"Clotho-val-tuned fixed interval for {setting}: [{iv[0]:g}, {iv[1]:g}] s (objective R1@0.5, 1-s grid, applied as absolute seconds clipped to [0, L])"
            if b != "clotho_moment":
                g2 = tb.candidate_grid(np.ceil(max(x["L"] for x in R)), 1.0)
                r, _ = tb.loo_best_fixed(R, g2, 0.5)
                rows["Best fixed (leave-one-out on benchmark)"][(b, setting)] = (r[0.5], r[0.7])
            for ln in (10, 20, 26, 30):
                r, _ = tb.centered(R, ln)
                rows[f"Centered {ln} s"][(b, setting)] = (r[0.5], r[0.7])
            if setting == "clipped":
                r, _ = tb.whole(R)
                rows["Whole 30-s window"][(b, setting)] = (r[0.5], r[0.7])
    return rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="eval/results/tables")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--prompt-id", default="v1", help="which prompt variant to tabulate for baselines")
    ap.add_argument("--parse-mode", default="paper", choices=["paper", "extended"],
                    help="parser used for the headline numbers (paper = to/hyphen/and only)")
    ap.add_argument("--no-trivial", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gt = {(b, s): load(b, s) for b in BENCHES for s in SETTINGS}
    found = discover(args.runs)
    entries = {}   # (system, kind, seed, bench, setting) -> dict
    recs_by = {}
    for (system, kind, seed, pid, b, s), files in sorted(found.items()):
        if kind == "baseline" and pid != args.prompt_id:
            continue
        preds = load_preds(files)
        R = build_records(gt[(b, s)], preds, args.parse_mode)
        Rext = build_records(gt[(b, s)], preds, "extended")
        n_out = sum(r["has_output"] for r in R)
        e = M.entry_with_ci(R, B=args.B, seed=0)
        e.update(system=system, kind=kind, seed=seed, prompt_id=pid, bench=b, setting=s, n_outputs=n_out,
                 complete=n_out == len(R), parse_rate_extended=100 * np.mean([r["pred"] is not None for r in Rext]),
                 files=files)
        e["diag"] = M.prediction_diagnostics(R)
        if b == "tut2017":
            mi = M.multi_interval_scores(R)
            e["multi"] = {k: 100 * float(v.mean()) for k, v in mi.items()}
        durs = [r["pred"][1] - r["pred"][0] for r in R if r["pred"] is not None]
        e["random_own_dur"] = tb.random_with_durations(gt[(b, s)], durs) if durs and min(len(durs), 1) else None
        entries[(system, kind, seed, b, s)] = e
        recs_by[(system, kind, seed, b, s)] = R
        print(f"{system:>14s} {seed:>4s} {b:>13s} {s:>8s} n_out={n_out:5d}/{len(R)} parse={e['parse_rate']:.1f} "
              f"old={e['old@0.5']:.2f}/{e['old@0.7']:.2f} miss={e['miss@0.5']:.2f}/{e['miss@0.7']:.2f}", flush=True)

    systems = sorted({(k[0], k[1]) for k in entries}, key=sys_sort_key)
    if ALIASES:
        print("aliases skipped (symlink to the same predictions):", ALIASES)
    cols = [(b, s) for b in BENCHES for s in SETTINGS]

    def agg(system, kind, b, s, key):
        return [entries[k][key] for k in entries if k[0] == system and k[1] == kind and k[3] == b and k[4] == s]

    triv = {} if args.no_trivial else trivial_rows(gt)

    # ---------------- main tables
    for proto in ("old", "miss"):
        hdr = "| System | " + " | ".join(f"{BENCH_PRETTY[b]} {s} R1@0.5 | R1@0.7 | mIoU" for b, s in cols) + " |"
        lines = [f"# Main table — protocol: {proto} "
                 f"({'R1 over parsed responses only (paper)' if proto == 'old' else 'parse failure = miss, R1 over all queries'})",
                 "", hdr, "|" + "---|" * (1 + 3 * len(cols))]
        tex = []
        for system, kind in systems:
            cells = []
            for b, s in cols:
                for t in THR:
                    cells.append(msd(agg(system, kind, b, s, f"{proto}@{t}")))
                cells.append(msd(agg(system, kind, b, s, f"miou_{proto}")))
            name = PRETTY.get(system, system) + ("" if kind == "baseline" else " (ours)")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
            tex.append(f"{name} & " + " & ".join(c.replace("±", "$\\pm$") for c in cells) + " \\\\")
        if triv:
            lines.append("| **Trivial baselines** |" + " |" * (3 * len(cols)))
            tex.append("\\midrule")
            for row, d in triv.items():
                if row == "_notes":
                    continue
                cells = [(fmt(d[c][i]) if c in d else "--") if i < 2 else "--" for c in cols for i in (0, 1, 2)]
                lines.append(f"| {row} | " + " | ".join(cells) + " |")
                tex.append(f"{row} & " + " & ".join(cells) + " \\\\")
        if triv:
            lines += [""] + [f"- {v}" for v in triv["_notes"].values()]
            for a, o in sorted(ALIASES.items()):
                lines.append(f"- {a}: symlink to the same predictions as {o}; not listed separately")
            lines.append(f"- IoU tolerance TOL={M.TOL:g} (hit iff IoU >= θ - TOL); set AGL_IOU_TOL=0 for strict float comparison")
        open(os.path.join(args.out, f"main_table_{proto}.md"), "w").write("\n".join(lines) + "\n")
        open(os.path.join(args.out, f"main_table_{proto}.tex"), "w").write("\n".join(tex) + "\n")

    # ---------------- parse rates
    lines = ["# Parse success rate (%) — paper parser [extended parser]", "",
             "| System | " + " | ".join(f"{BENCH_PRETTY[b]} {s}" for b, s in cols) + " |", "|" + "---|" * (1 + len(cols))]
    for system, kind in systems:
        cells = []
        for b, s in cols:
            a, x = agg(system, kind, b, s, "parse_rate"), agg(system, kind, b, s, "parse_rate_extended")
            no = agg(system, kind, b, s, "complete")
            cells.append("--" if not a else f"{msd(a, 1)} [{msd(x, 1)}]" + ("" if all(no) else " (incomplete)"))
        lines.append(f"| {PRETTY.get(system, system)} | " + " | ".join(cells) + " |")
    open(os.path.join(args.out, "parse_rates.md"), "w").write("\n".join(lines) + "\n")

    # ---------------- entries csv/json (CIs)
    flds = ["system", "kind", "seed", "prompt_id", "bench", "setting", "n", "n_outputs", "complete", "n_parsed", "parse_rate",
            "parse_rate_extended"]
    with open(os.path.join(args.out, "entries.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(flds + [c for p in ("old", "miss")
                           for c in [f"{p}@{t}{x}" for t in THR for x in ("", "_lo", "_hi")] + [f"miou_{p}{x}" for x in ("", "_lo", "_hi")]]
                   + ["parse_lo", "parse_hi"])
        for k, e in entries.items():
            row = [e[x] for x in flds]
            for p in ("old", "miss"):
                for t in THR:
                    row += [round(e[f"{p}@{t}"], 2), round(e[f"{p}@{t}_ci"][0], 2), round(e[f"{p}@{t}_ci"][1], 2)]
                row += [round(e[f"miou_{p}"], 2), round(e[f"miou_{p}_ci"][0], 2), round(e[f"miou_{p}_ci"][1], 2)]
            row += [round(e["parse_rate_ci"][0], 2), round(e["parse_rate_ci"][1], 2)]
            w.writerow(row)
    json.dump({"|".join(k): e for k, e in entries.items()}, open(os.path.join(args.out, "entries.json"), "w"),
              indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # ---------------- paired bootstrap p-values (first seed of each system)
    first = {}
    for k in sorted(entries):
        first.setdefault((k[0], k[1], k[3], k[4]), k)
    with open(os.path.join(args.out, "pvalues.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bench", "setting", "protocol", "thr", "A", "B", "delta(A-B)", "ci_lo", "ci_hi", "p"])
        for b, s in cols:
            here = [(sy, kd) for sy, kd in systems if (sy, kd, b, s) in first]
            for i, A in enumerate(here):
                for Bsys in here[i + 1:]:
                    RA, RB = recs_by[first[A + (b, s)]], recs_by[first[Bsys + (b, s)]]
                    pa, ia, ha = M.per_query_hits(RA)
                    pb, ib, hb = M.per_query_hits(RB)
                    for t in THR:
                        for proto in ("old", "miss"):
                            mA, mB = (pa, pb) if proto == "old" else (None, None)
                            d, lo, hi, p = M.paired_bootstrap_p(ha[t].astype(float), hb[t].astype(float), mA, mB, B=args.B, seed=0)
                            w.writerow([b, s, proto, t, A[0], Bsys[0], round(d, 2), round(lo, 2), round(hi, 2), p])

    # ---------------- TUT multi-interval
    lines = ["# TUT-SED 2017 multi-interval scoring (all pairs from parse_all; 0 if unparsable)", "",
             "| System | setting | max-IoU R1@0.5 (miss) | R1@0.7 (miss) | set-F1@0.5 | set-F1@0.7 | GT-recall@0.5 | GT-recall@0.7 | union-IoU |",
             "|---|---|---|---|---|---|---|---|---|"]
    for system, kind in systems:
        for s in SETTINGS:
            ks = [k for k in entries if k[0] == system and k[1] == kind and k[3] == "tut2017" and k[4] == s]
            if not ks:
                continue
            g = lambda key: msd([entries[k]["multi"][key] for k in ks])  # noqa: E731
            lines.append(f"| {PRETTY.get(system, system)} | {s} | {msd([entries[k]['miss@0.5'] for k in ks])} | "
                         f"{msd([entries[k]['miss@0.7'] for k in ks])} | {g('set_f1@0.5')} | {g('set_f1@0.7')} | "
                         f"{g('mean_gt@0.5')} | {g('mean_gt@0.7')} | {g('union_iou')} |")
    open(os.path.join(args.out, "tut_multi_interval.md"), "w").write("\n".join(lines) + "\n")

    # ---------------- Clotho breakdowns (bins on ORIGINAL GT, as in the paper tables)
    cen_bins = [0, 10, 20, 30, 40, 50, 60.001]
    dur_bins = [0, 5, 10, 20, 30]
    for name, bins, keyf in (("position", cen_bins, lambda g: (g[0] + g[1]) / 2), ("duration", dur_bins, lambda g: g[1] - g[0])):
        lines = [f"# Clotho-Moment breakdown by GT {name} (bins on original-recording GT; duration bins exclude >=30 s as in paper; float durations as stored)", ""]
        for s in SETTINGS:
            lines += [f"## {s}", "", "| System | bin | N | parse% | R1@0.5 old | R1@0.7 old | R1@0.5 miss | R1@0.7 miss | GT dur | pred dur |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for system, kind in systems:
                ks = [k for k in entries if k[0] == system and k[1] == kind and k[3] == "clotho_moment" and k[4] == s]
                if not ks:
                    continue
                for i in range(len(bins) - 1):
                    vals = defaultdict(list)
                    for k in ks:
                        R = recs_by[k]
                        sel = [r for r in R if bins[i] <= keyf(r["gt_original"][0]) < bins[i + 1]]
                        if not sel:
                            continue
                        e = M.r1(sel)
                        for kk in ("parse_rate", "old@0.5", "old@0.7", "miss@0.5", "miss@0.7"):
                            vals[kk].append(e[kk])
                        vals["gt"].append(np.mean([r["gt"][0][1] - r["gt"][0][0] for r in sel]))
                        pd_ = [r["pred"][1] - r["pred"][0] for r in sel if r["pred"] is not None]
                        vals["pd"].append(np.mean(pd_) if pd_ else float("nan"))
                        n = len(sel)
                    if vals:
                        lines.append(f"| {PRETTY.get(system, system)} | {bins[i]:g}-{bins[i+1]:g} | {n} | {msd(vals['parse_rate'], 1)} | "
                                     f"{msd(vals['old@0.5'], 1)} | {msd(vals['old@0.7'], 1)} | {msd(vals['miss@0.5'], 1)} | "
                                     f"{msd(vals['miss@0.7'], 1)} | {msd(vals['gt'], 1)} | {msd(vals['pd'], 1)} |")
            lines.append("")
        open(os.path.join(args.out, f"breakdown_{name}.md"), "w").write("\n".join(lines) + "\n")

    # ---------------- diagnostics + random-with-own-duration baseline
    lines = ["# Prediction diagnostics (paper parser; first seed for ours)", "",
             "| System | bench | setting | n_parsed | unique intervals | unique % | dur mean±std | dur median | onset mean | onset std | norm-onset std | top interval (count) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rl = ["# Random interval with the model's own predicted-duration distribution (onset ~ U[0, L-d], 200 draws)", "",
          "| System | bench | setting | model R1@0.5 (miss) | random R1@0.5 | model R1@0.7 (miss) | random R1@0.7 |", "|---|---|---|---|---|---|---|"]
    for (system, kind, b, s), k in sorted(first.items(), key=lambda x: (sys_sort_key(x[0][:2]), x[0][2:])):
        e, d = entries[k], entries[k]["diag"]
        if d.get("n_parsed", 0):
            top = d["top5_intervals"][0] if d["top5_intervals"] else None
            lines.append(f"| {PRETTY.get(system, system)} | {b} | {s} | {d['n_parsed']} | {d['unique_intervals']} | {100*d['unique_frac']:.1f} | "
                         f"{d['dur_mean']:.1f}±{d['dur_std']:.1f} | {d['dur_median']:.1f} | {d['onset_mean']:.1f} | {d['onset_std']:.2f} | "
                         f"{d['norm_onset_std']:.3f} | {top[0]} ({top[1]}) |")
        if e["random_own_dur"]:
            r = e["random_own_dur"]
            rl.append(f"| {PRETTY.get(system, system)} | {b} | {s} | {fmt(e['miss@0.5'])} | {fmt(r[0.5])}±{fmt(r['0.5_std'])} | "
                      f"{fmt(e['miss@0.7'])} | {fmt(r[0.7])}±{fmt(r['0.7_std'])} |")
    open(os.path.join(args.out, "diagnostics.md"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(args.out, "random_dur_baseline.md"), "w").write("\n".join(rl) + "\n")

    # ---------------- histograms
    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for b, s in cols:
            ks = [first[(sy, kd, b, s)] for sy, kd in systems if (sy, kd, b, s) in first]
            if not ks:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            maxL = max(r["L"] for r in gt[(b, s)])
            bins = np.arange(0, maxL + 2, 2)
            ax.hist([w[1] - w[0] for r in gt[(b, s)] for w in r["gt"]], bins=bins, histtype="stepfilled", alpha=.25,
                    color="k", density=True, label="GT")
            for k in ks:
                dd = [r["pred"][1] - r["pred"][0] for r in recs_by[k] if r["pred"] is not None]
                if dd:
                    ax.hist(np.clip(dd, 0, maxL + 1), bins=bins, histtype="step", density=True, label=PRETTY.get(k[0], k[0]))
            ax.set_xlabel("predicted duration (s)"); ax.set_ylabel("density"); ax.set_title(f"{BENCH_PRETTY[b]} — {s}")
            ax.legend(fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(args.out, f"pred_dur_hist_{b}_{s}.png"), dpi=130); plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
