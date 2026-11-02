#!/usr/bin/env python
"""Paper tables straight from prediction files (unified protocol: parse failure = miss, R1 over all queries).

  Tables/main_table1.tex   Table 3: Original setting, R1@0.5 / R1@0.7 / mIoU, all systems (+ trivial baselines)
  Tables/clipped_table.tex Clipped setting (search space reduced)
  Tables/ablation.tex      Table 4: temporal components, three seeds
  Tables/parse_rates.tex   parse-success rate of every system / benchmark / setting (appendix)
Ours: mean and standard deviation over seeds (subscript). Missing or incomplete prediction files -> "--" and listed on stdout.
Bold: best LALM/ours value per column (AM-DETR, trivial baselines and the in-domain/fine-tuned reference rows excluded).

usage: python experiments/eval/paper_tables.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "experiments/eval")
sys.path.insert(0, "eval")
import make_tables as T  # noqa: E402
from agl_eval import metrics as M  # noqa: E402
from agl_eval.data import load  # noqa: E402

R = "runs"
OUT = "results/tables"
BENCHES = ["clotho_moment", "unav100", "tut2017"]
BNAME = {"clotho_moment": "Clotho-Moment", "unav100": "UnAV-100 subset", "tut2017": "TUT-Sound Events 2017"}
GT = {(b, s): load(b, s) for b in BENCHES for s in ("original", "clipped")}
TACOS_GT = load("tacos", "original")
MISSING = []

# AM-DETR: task-specific retrieval model, numbers reported in [Clotho-moment] (R1 only)
AMDETR = {("clotho_moment", "original"): (87.50, 81.86), ("unav100", "original"): (61.00, 39.00),
          ("tut2017", "original"): (14.42, 5.77), ("clotho_moment", "clipped"): (57.96, 29.40),
          ("unav100", "clipped"): (40.00, 19.00), ("tut2017", "clipped"): (13.46, 6.73)}


def files_for(src, b, s):
    """src = ('base', model) | ('ours', exp, tag, seeds) -> list of file-lists (one per seed)."""
    if src[0] == "base":   # shared prompt v1: <bench>_<setting>.jsonl or its .shardIofK parts (other prompt ids carry .<pid>.)
        d = f"{R}/baselines/{src[1]}"
        fs = glob.glob(f"{d}/{b}_{s}.jsonl") + glob.glob(f"{d}/{b}_{s}.shard*of*.jsonl")
        return [fs] if fs else []
    _, exp, tag, seeds = src
    return [[f"{R}/{exp}/{k}/preds/{tag}/{b}_{s}.jsonl"] for k in seeds if os.path.exists(f"{R}/{exp}/{k}/preds/{tag}/{b}_{s}.jsonl")]


def stats(src, b, s, need):
    """-> dict metric -> (mean, std, n) or None if incomplete."""
    per = []
    for fl in files_for(src, b, s):
        recs = T.build_records(GT[(b, s)], T.load_preds(fl))
        if sum(r["has_output"] for r in recs) < len(recs):
            continue
        e = M.r1(recs, thrs=(0.5, 0.7, 0.9))
        per.append((e["miss@0.5"], e["miss@0.7"], e["miou_miss"], e["parse_rate"], e["old@0.5"], e["miss@0.9"]))
    if len(per) < need:
        MISSING.append(f"{src} {b} {s}: {len(per)}/{need} complete")
        return None
    a = np.array(per)
    sd = a.std(0, ddof=1) if len(a) > 1 else np.full(6, np.nan)
    return dict(r05=(a[:, 0].mean(), sd[0]), r07=(a[:, 1].mean(), sd[1]), miou=(a[:, 2].mean(), sd[2]),
                parse=(a[:, 3].mean(), sd[3]), old05=(np.nanmean(a[:, 4]), sd[4]), r09=(a[:, 5].mean(), sd[5]), n=len(a))


def interval_stats(fn, b, s):
    recs = [dict(gt=g["gt"], pred=list(fn(g["L"])), L=g["L"]) for g in GT[(b, s)]]
    e = M.r1(recs)
    return dict(r05=(e["miss@0.5"], np.nan), r07=(e["miss@0.7"], np.nan), miou=(e["miou_miss"], np.nan), n=1)


def cell(v, bold=False):
    if v is None:
        return "--"
    m, sd = v
    txt = f"{m:.2f}"
    if bold:
        txt = f"\\textbf{{{txt}}}"
    if np.isfinite(sd):
        txt += f"$_{{\\pm{sd:.1f}}}$"
    return txt


def rows_block(rows, setting, metrics=("r05", "r07", "miou"), benches=None):
    benches = benches or BENCHES
    """rows: list of (label, stats-per-bench dict or None, eligible_for_bold) -> LaTeX lines with per-column bold."""
    best = {}
    for b in benches:
        for m in metrics:
            vals = [r[1][b][m][0] for r in rows if r[2] and r[1] and r[1].get(b) and m in r[1][b]]
            best[(b, m)] = max(vals) if vals else None
    out = []
    for label, st, elig in rows:
        if label.startswith("\\midrule") or label.startswith("\\rowcolor"):
            out.append(label)
            continue
        cells = []
        for b in benches:
            for m in metrics:
                v = st.get(b, {}).get(m) if st and st.get(b) else None
                cells.append(cell(v, bold=bool(elig and v is not None and best[(b, m)] is not None
                                                   and abs(v[0] - best[(b, m)]) < 1e-9)))
        out.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return out


BASE = [("Qwen2-Audio-Instruct~\\cite{chu2024qwen2}$^\\dagger$", "qwen2_audio"),
        ("Qwen2.5-Omni~\\cite{xu2025qwen25omnitechnicalreport}$^\\dagger$", "qwen25_omni"),
        ("GAMA~\\cite{ghosh2024gama}$^\\dagger$", "gama"),
        ("DeSTA2.5-Audio~\\cite{lu2025desta2}$^\\dagger$", "desta25"),
        ("AF2~\\cite{AF2}$^\\dagger$", "af2")]
SALMONN = ("SALMONN-7B~\\cite{tang2023salmonn}$^\\dagger$", ("ours", "E7_salmonn_zeroshot", "promptv1", (0,)), 1)
TIMEAUDIO = ("TimeAudio~\\cite{wang2025listening}$^{\\dagger}$", "timeaudio")
QWENFT = ("Qwen2.5-Omni, fine-tuned on AudioGround-IT", ("ours", "E8L_qwen25omni_ft", "default", (0,)), 1)
CONF = ("AudioGround ($\\rho_0=22\\%$, ATE$_h$)", ("ours", "E0_main", "default", (0, 1, 2)), 3)
OURS = ("\\textbf{AudioGround} ($\\rho_0=0\\%$, ATE$_l$)", ("ours", "E1L_fi_ts_learned", "default", (0, 1, 2)), 3)
MIXED = ("\\quad + junction augmentation (Sec.~\\ref{sec:real})", ("ours", "E4La_mixjunction", "default", (0, 1, 2)), 3)


def by_bench(src, s, need=1):
    return {b: stats(src, b, s, need) for b in BENCHES}


def trivial_rows(s):
    rows = []
    if s == "original":
        rows.append(("Fixed interval [7, 37]\\,s (tuned on Clotho-Moment val)",
                     {b: interval_stats(lambda L: (min(7.0, L), min(37.0, L)), b, s) for b in BENCHES}, False))
        rows.append(("Centered 26-s interval",
                     {b: interval_stats(lambda L: (max(0, L / 2 - 13), min(L, L / 2 + 13)), b, s) for b in BENCHES}, False))
    else:
        rows.append(("Whole 30-s window",
                     {b: interval_stats(lambda L: (0.0, L), b, s) for b in BENCHES}, False))
        rows.append(("Centered 26-s interval",
                     {b: interval_stats(lambda L: (max(0, L / 2 - 13), min(L, L / 2 + 13)), b, s) for b in BENCHES}, False))
    return rows


def header(metrics_label, ncol_per, benches=None):
    benches = benches or BENCHES
    h = ["\\toprule", "\\multirow{2}{*}{Model} & " + " & ".join(
        f"\\multicolumn{{{ncol_per}}}{{c}}{{\\textbf{{{BNAME[b]}}}}}" for b in benches) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + i * ncol_per}-{1 + (i + 1) * ncol_per}}}" for i in range(len(benches))),
         "& " + " & ".join([metrics_label] * len(benches)) + " \\\\", "\\midrule"]
    return h


def amdetr_row(s):
    cells = []
    for b in BENCHES:
        r = AMDETR[(b, s)]
        cells += [f"{r[0]:.2f}", f"{r[1]:.2f}", "--"]
    return "AM-DETR~\\cite{Clotho-moment}$^\\ddagger$ (task-specific) & " + " & ".join(cells) + " \\\\"


def main_table(s):
    rows = [("\\rowcolor{gray!15}\\multicolumn{10}{c}{\\textbf{LALMs, OOD (no benchmark-specific training)}} \\\\", None, False)]
    for label, m in BASE:
        rows.append((label, by_bench(("base", m), s), True))
    rows.append((SALMONN[0], by_bench(SALMONN[1], s, SALMONN[2]), True))
    rows.append((TIMEAUDIO[0], by_bench(("base", TIMEAUDIO[1]), s), True))
    rows.append(("\\midrule", None, False))
    if s == "original":
        rows.append((QWENFT[0] + " (one run)", by_bench(QWENFT[1], s, QWENFT[2]), True))
    rows.append((CONF[0], by_bench(CONF[1], s, CONF[2]), True))
    rows.append((OURS[0], by_bench(OURS[1], s, OURS[2]), True))
    rows.append(("\\midrule", None, False))
    rows.append(("\\rowcolor{gray!15}\\multicolumn{10}{c}{\\textbf{Trivial baselines (query-independent)}} \\\\", None, False))
    rows += trivial_rows(s)
    body = rows_block(rows, s)
    return body


def write_main():
    body = main_table("original")
    tex = [r"% Generated by experiments/eval/paper_tables.py -- do not edit by hand.",
           r"\begin{table*}[htb!]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.0pt}",
           r"\renewcommand{\arraystretch}{1.12}",
           r"\caption{\rev{Retrieval performance in the Original setting (mean and standard deviation over seeds as subscript). $^\dagger$Reproduced by the authors. $^\ddagger$Task-specific model; numbers reported in~\cite{Clotho-moment}. Bold marks the best LALM per column, and trivial baselines ignore the query.}}",
           r"\label{tab:main}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l ccc ccc ccc}"]
    tex += header("R1@0.5 & R1@0.7 & mIoU", 3)
    tex += [amdetr_row("original"), "\\midrule"] + body + ["\\bottomrule", "\\end{tabular}", "} % end resizebox", "\\end{table*}"]
    open(f"{OUT}/main_table1.tex", "w").write("\n".join(tex) + "\n")

    body = main_table("clipped")
    tex = [r"% Generated by experiments/eval/paper_tables.py -- do not edit by hand.",
           r"\begin{table*}[htb!]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.0pt}",
           r"\renewcommand{\arraystretch}{1.12}",
           r"\caption{\rev{Retrieval performance in the Clipped setting (a 30-s window that contains the target), same notation as Table~\ref{tab:main}. The window halves the search space of Clotho-Moment, and in 85.3\% of its queries the target covers at least half of the window, so answering the whole window reaches 85.3 R1@0.5 on Clotho-Moment, above every model.}}",
           r"\label{tab:clipped}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l ccc ccc ccc}"]
    tex += header("R1@0.5 & R1@0.7 & mIoU", 3)
    tex += [amdetr_row("clipped"), "\\midrule"] + body + ["\\bottomrule", "\\end{tabular}", "} % end resizebox", "\\end{table*}"]
    open(f"{OUT}/clipped_table.tex", "w").write("\n".join(tex) + "\n")


ABL = [("SALMONN-7B (no fine-tuning)", ("ours", "E7_salmonn_zeroshot", "promptv1", (0,)), 1, False),
       ("SALMONN-7B, fine-tuned", ("ours", "E7L_salmonn_ft", "default", (0, 1, 2)), 3, True),
       ("\\quad + FI", ("ours", "E1L_fi", "default", (0, 1, 2)), 3, True),
       ("\\quad + FI + TS", ("ours", "E1L_fi_ts", "default", (0, 1, 2)), 3, True),
       ("\\quad + FI + TS + ATE$_s$", ("ours", "E1L_fi_ts_sin", "default", (0, 1, 2)), 3, True),
       ("\\quad + FI + TS + ATE$_h$ (conference default)", ("ours", "E1L_fi_ts_hybrid", "default", (0, 1, 2)), 3, True),
       ("\\quad + FI + TS + ATE$_l$ (\\textbf{AudioGround})", ("ours", "E1L_fi_ts_learned", "default", (0, 1, 2)), 3, True)]


# Conference version (Interspeech 2026), Table 4 exactly as reported: one run, R1 over parsed responses only, rho0 = 22%.
CONF_REPORTED = [
    ("SALMONN", [2.30, 0.00, 11.00, 0.00, 0.00, 0.00]),
    ("\\quad + FI + TS", [26.65, 12.33, 18.18, 11.11, 7.84, 1.96]),
    ("\\quad + FI + TS + ATE$_s$", [26.82, 11.99, 23.23, 14.14, 9.80, 4.90]),
    ("\\quad + FI + TS + ATE$_l$", [25.41, 10.57, 20.20, 14.14, 8.82, 3.92]),
    ("\\quad + FI + TS + ATE$_h$ (conference model)", [27.47, 12.68, 23.23, 14.14, 10.78, 4.90]),
]
ABL22 = [("SALMONN-7B, fine-tuned", ("ours", "E7_salmonn_ft", "default", (0, 1, 2)), 3, False),
         ("\\quad + FI + TS + ATE$_l$", ("ours", "E1_fi_ts_learned", "default", (0, 1, 2)), 3, False),
         ("\\quad + FI + TS + ATE$_h$ (conference model)", ("ours", "E0_main", "default", (0, 1, 2)), 3, False)]


def write_ablation():
    _write_ablation(["clotho_moment", "unav100", "tut2017"], "ablation", "tab:ablation", True)


def _write_ablation(benches, fname, label, main):
    K = {"clotho_moment": 0, "unav100": 1, "tut2017": 2}
    nc = 1 + 4 * len(benches)
    conf_lines = []
    for lab, v in CONF_REPORTED:
        cells = []
        for b in benches:
            k = K[b]
            r5, r7 = v[2 * k], v[2 * k + 1]
            cells += ["--" if r5 is None else (f"{r5:.2f}" if lab.startswith("SALMONN") or "FI" in lab and "no FI" not in lab else f"{r5:.1f}"),
                      "--" if r7 is None else f"{r7:.2f}", "--", "--"]
        conf_lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    rows0 = [(lab, by_bench(src, "original", need), elig) for lab, src, need, elig in ABL]
    rows0.insert(1, ("\\midrule", None, False))
    rows0.append(("\\midrule", None, False))   # time embedding without timestamp conditioning (R1-2: TS vs. sinusoidal base vs. residual MLP)
    for lab, exp in [("\\quad + FI + ATE$_s$ (no TS)", "E1L_fi_sin"), ("\\quad + FI + ATE$_h$ (no TS)", "E1L_fi_hybrid")]:
        rows0.append((lab, by_bench(("ours", exp, "default", (0, 1, 2)), "original", 3), True))
    common = (r"(A) Conference version~\cite{kim2026audioground} as reported there: training data with $\rho_0=22\%$, one run, R1 over parsed responses only. (B) AudioGround-IT with $\rho_0=0\%$, three seeds. R1@0.9 was not reported in the conference version. Mean with the standard deviation over seeds as subscript. FI: frame-level interpolation; TS: timestamp conditioning of the Q-Former; ATE$_s$, ATE$_l$, ATE$_h$: sinusoidal, learned, and hybrid absolute time embedding.")
    if main:
        cap = r"\caption{\rev{Contribution of the temporal components (Original setting). (A) Conference version~\cite{kim2026audioground} as reported there ($\rho_0=22\%$, one run, parsed responses only). (B) $\rho_0=0\%$, mean and standard deviation over three seeds. FI denotes frame-level interpolation, TS timestamp conditioning, and ATE$_s$/ATE$_l$/ATE$_h$ the sinusoidal/learned/hybrid time embedding. Bold marks the best mean among the fine-tuned models of (B).}}"
        tex = [r"% Generated by experiments/eval/paper_tables.py -- do not edit by hand.", r"\begin{table*}[t!]", r"\centering", r"\small",
               r"\setlength{\tabcolsep}{4.0pt}", r"\renewcommand{\arraystretch}{1.12}", cap, f"\\label{{{label}}}",
               r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l " + " ".join(["cccc"] * len(benches)) + "}"]
    else:
        cap = r"\caption{\rev{Contribution of the temporal components on TUT-Sound Events 2017 (104 queries, about one point per query; Clotho-Moment and UnAV-100: Table~\ref{tab:ablation}). " + common + r" The variants are not separated on this benchmark (Section~\ref{sec:limitations}).}}"
        tex = [r"% Generated by experiments/eval/paper_tables.py -- do not edit by hand.", r"\begin{table}[t!]", r"\centering", r"\scriptsize",
               r"\setlength{\tabcolsep}{3.0pt}", r"\renewcommand{\arraystretch}{1.12}", cap, f"\\label{{{label}}}",
               r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{l cccc}"]
    tex += header("R1@0.5 & R1@0.7 & R1@0.9 & mIoU", 4, benches)
    tex += [f"\\rowcolor{{gray!15}}\\multicolumn{{{nc}}}{{l}}{{\\textbf{{(A) Conference version}}}} \\\\"]
    tex += conf_lines + [r"\midrule"]
    tex += [f"\\rowcolor{{gray!15}}\\multicolumn{{{nc}}}{{l}}{{\\textbf{{(B) $\\rho_0=0\\%$}}}} \\\\"]
    tex += rows_block(rows0, "original", metrics=("r05", "r07", "r09", "miou"), benches=benches)
    tex += ["\\bottomrule", "\\end{tabular}", "} % end resizebox", "\\end{table*}" if main else "\\end{table}"]
    open(f"{OUT}/{fname}.tex", "w").write("\n".join(tex) + "\n")


def write_parse():
    systems = [(l, ("base", m), 1) for l, m in BASE] + [(SALMONN[0], SALMONN[1], 1), (TIMEAUDIO[0], ("base", "timeaudio"), 1),
                                                         (QWENFT[0], QWENFT[1], 1), (CONF[0], CONF[1], 3), (OURS[0], OURS[1], 3)]
    lines = [r"% Generated by experiments/eval/paper_tables.py -- do not edit by hand.",
             r"\begin{table}[t!]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.0pt}",
             r"\caption{\rev{Parse-success rate (\%) of every system in Tables~\ref{tab:main} and~\ref{tab:clipped}, i.e., the share of responses from which the parser extracts an interval. Orig. denotes Original and Clip. denotes Clipped.}}",
             r"\label{tab:parse}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{l cc cc cc}", r"\toprule",
             r"\multirow{2}{*}{Model} & \multicolumn{2}{c}{Clotho} & \multicolumn{2}{c}{UnAV} & \multicolumn{2}{c}{TUT} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}", r"& Orig. & Clip. & Orig. & Clip. & Orig. & Clip. \\", r"\midrule"]
    for lab, src, need in systems:
        cells = []
        for b in BENCHES:
            for s in ("original", "clipped"):
                st = stats(src, b, s, need)
                cells.append("--" if st is None else f"{st['parse'][0]:.1f}")
        lines.append(lab.replace("$^\\dagger$", "").replace("$^{\\dagger}$", "") + " & " + " & ".join(cells) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    open(f"{OUT}/parse_rates.tex", "w").write("\n".join(lines) + "\n")


def ci_cells(src, need):
    """95% bootstrap CI (10k resamples of the queries) of R1@0.5 and R1@0.7; trained models: per-query hits averaged over seeds."""
    cells = []
    for b in BENCHES:
        H5, H7 = [], []
        for fl in files_for(src, b, "original"):
            recs = T.build_records(GT[(b, "original")], T.load_preds(fl))
            if sum(r["has_output"] for r in recs) < len(recs):
                continue
            _, _, h = M.per_query_hits(recs)
            H5.append(h[0.5].astype(float)); H7.append(h[0.7].astype(float))
        if len(H5) < need:
            cells += ["--", "--"]
            continue
        for H in (H5, H7):
            pt, lo, hi = M.bootstrap_ci(np.mean(H, 0), None, B=10000, seed=0)
            cells.append(f"{pt:.1f} [{lo:.1f}, {hi:.1f}]")
    return cells


def write_ci():
    systems = [(l, ("base", m), 1) for l, m in BASE] + [(SALMONN[0], SALMONN[1], 1), (TIMEAUDIO[0], ("base", "timeaudio"), 1),
                                                         (QWENFT[0], QWENFT[1], 1), (CONF[0], CONF[1], 3)]
    systems += [(lab, src, need) for lab, src, need, _ in ABL[1:]]
    systems += [("\\quad + FI + ATE$_s$ (no TS)", ("ours", "E1L_fi_sin", "default", (0, 1, 2)), 3),
                ("\\quad + FI + ATE$_h$ (no TS)", ("ours", "E1L_fi_hybrid", "default", (0, 1, 2)), 3)]
    lines = [r"% Generated by experiments/eval/paper_tables.py -- do not edit by hand.",
             r"\begin{table*}[t!]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.0pt}", r"\renewcommand{\arraystretch}{1.1}",
             r"\caption{\rev{Bootstrap 95\% confidence intervals (10,000 resamples of the queries) of the LALM entries of Table~\ref{tab:main} and of the fine-tuned models of Table~\ref{tab:ablation} (Original setting). For models trained with three seeds, the per-query hits are averaged over the seeds before resampling, so the intervals reflect the uncertainty due to the finite benchmark, and their widths are not comparable with those of the single-run rows; the seed spread is given in the main tables.}}",
             r"\label{tab:ci}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l cc cc cc}", r"\toprule",
             r"\multirow{2}{*}{Model} & \multicolumn{2}{c}{\textbf{Clotho-Moment}} & \multicolumn{2}{c}{\textbf{UnAV-100 subset}} & \multicolumn{2}{c}{\textbf{TUT-Sound Events 2017}} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}", r"& R1@0.5 & R1@0.7 & R1@0.5 & R1@0.7 & R1@0.5 & R1@0.7 \\", r"\midrule"]
    for k, (lab, src, need) in enumerate(systems):
        if k == len(BASE) + 4:
            lines.append(r"\midrule")
        lines.append(lab.replace("$^\\dagger$", "").replace("$^{\\dagger}$", "") + " & " + " & ".join(ci_cells(src, need)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"]
    open(f"{OUT}/ci_table.tex", "w").write("\n".join(lines) + "\n")


def write_multi():
    """TUT-SE2017 (Original) multi-interval scoring, grounding vs. frequency question, three ATEs x three seeds."""
    from agl_eval.parser import parse_all

    def iou(a, g):
        i = max(0.0, min(a[1], g[1]) - max(a[0], g[0])); u = max(a[1], g[1]) - min(a[0], g[0]); return i / u if u > 0 else 0.0
    g = GT[("tut2017", "original")]
    rows_out = []
    for lab, exp in (("ATE$_s$", "E1L_fi_ts_sin"), ("ATE$_l$ (ours)", "E1L_fi_ts_learned"), ("ATE$_h$", "E1L_fi_ts_hybrid")):
        for qlab, tag in (("grounding", "default"), ("frequency", "freqprompt")):
            vals = {k: [] for k in ("n", "r1", "r5", "f1", "rec")}
            for sd in (0, 1, 2):
                f = f"{R}/{exp}/{sd}/preds/{tag}/tut2017_original.jsonl"
                if not (os.path.exists(f) and os.path.exists(f"{R}/{exp}/{sd}/preds/{tag}/DONE")):
                    continue
                recs = T.build_records(g, T.load_preds([f]))
                mi = M.multi_interval_scores(recs)
                raw = [T.load_preds([f])[str(x["qid"])]["raw_output"] for x in g]
                P = [parse_all(t, "paper") for t in raw]
                vals["n"].append(np.mean([len(p) for p in P]))
                vals["r1"].append(100 * np.mean([bool(p) and any(iou(p[0], gg) >= 0.5 - 1e-6 for gg in x["gt"]) for p, x in zip(P, g)]))
                vals["r5"].append(100 * np.mean([any(iou(q, gg) >= 0.5 - 1e-6 for q in p[:5] for gg in x["gt"]) for p, x in zip(P, g)]))
                vals["f1"].append(100 * float(mi["set_f1@0.5"].mean())); vals["rec"].append(100 * float(mi["mean_gt@0.5"].mean()))
            rows_out.append((lab, qlab, vals))
    best = {k: max((np.mean(v[k]) for _, _, v in rows_out if len(v[k]) == 3), default=None) for k in ("r1", "r5", "f1", "rec")}
    lines = [r"% Generated by experiments/eval/paper_tables.py -- do not edit by hand.",
             r"\begin{table}[t]", r"\centering",
             r"\caption{\rev{Repeated events on TUT-Sound Events 2017 (Original setting), with the same queries asked as a grounding question and in the frequency format. Mean and standard deviation over three seeds; bold marks the highest mean per column.}}",
             r"\label{tab:multi}", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}", r"\renewcommand{\arraystretch}{1.15}",
             r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llccccc}", r"\toprule",
             r"\textbf{Model} & \textbf{Question} & \textbf{Intervals} & \textbf{R1@0.5} & \textbf{R5@0.5} & \textbf{Set-F1} & \textbf{GT recall} \\", r"\midrule"]
    for k, (lab, qlab, v) in enumerate(rows_out):
        if len(v["r1"]) < 3:
            cells = ["--"] * 5
            MISSING.append(f"multi {lab} {qlab}: {len(v['r1'])}/3")
        else:
            cells = [f"{np.mean(v['n']):.1f}"] + [
                (f"\\textbf{{{np.mean(v[m]):.1f}}}" if abs(np.mean(v[m]) - best[m]) < 1e-9 else f"{np.mean(v[m]):.1f}") + f"$_{{\\pm{np.std(v[m], ddof=1):.1f}}}$"
                for m in ("r1", "r5", "f1", "rec")]
        lines.append(f"{lab if qlab == 'grounding' else ''} & {qlab} & " + " & ".join(cells) + r" \\")
        if qlab == "frequency" and k < len(rows_out) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"} % end resizebox", r"\end{table}"]
    open(f"{OUT}/multi_table.tex", "w").write("\n".join(lines) + "\n")


def _r1_sets(exp, bench, tag, seeds=(0, 1, 2)):
    """per seed: (R1@0.5, mIoU, share of answers starting in the last second or after the end)"""
    g = GT[(bench, "original")] if bench != "tacos" else TACOS_GT
    out = []
    for sd in seeds:
        f = f"{R}/{exp}/{sd}/preds/{tag}/{bench}_original.jsonl"
        if not (os.path.exists(f) and os.path.exists(f"{R}/{exp}/{sd}/preds/{tag}/DONE")):
            continue
        recs = T.build_records(g, T.load_preds([f]))
        if sum(r["has_output"] for r in recs) < len(recs):
            continue
        e = M.r1(recs)
        end = np.mean([r["pred"] is not None and r["pred"][0] >= r["L"] - 1.0 for r in recs])
        out.append((e["miss@0.5"], e["miou_miss"], 100 * end))
    return np.array(out)


def _ms(a, k):
    if len(a) == 0:
        return "--"
    return f"{a[:, k].mean():.1f}" + (f"$_{{\\pm{a[:, k].std(ddof=1):.1f}}}$" if len(a) > 1 else "")


JUNCTIONS = [("Silence 0.3--0.5\\,s (base)", "E1L_fi_ts_learned"), ("No gap", "E4La_nogap"), ("Crossfade 0.5--1\\,s", "E4La_crossfade"),
             ("Overlap 1--2\\,s", "E4La_overlap"), ("Background mix", "E4La_bgmix")]


def _short_sets(exp, seeds=(0, 1, 2)):
    """per seed: Clotho-Moment R1@0.5 on the 79 targets shorter than 5 s"""
    from agl_eval.metrics import max_iou
    g = GT[("clotho_moment", "original")]
    sel = np.array([x["gt"][0][1] - x["gt"][0][0] < 5 for x in g])
    out = []
    for sd in seeds:
        f = f"{R}/{exp}/{sd}/preds/default/clotho_moment_original.jsonl"
        if not (os.path.exists(f) and os.path.exists(f"{R}/{exp}/{sd}/preds/default/DONE")):
            continue
        recs = T.build_records(g, T.load_preds([f]))
        I = np.array([max_iou(r["pred"], r["gt"]) if r["pred"] else 0.0 for r in recs])
        out.append((100 * (I[sel] >= .5 - 1e-6).mean(),))
    return np.array(out)


def write_junction():
    lines = [r"% Generated by experiments/eval/paper_tables.py -- do not edit by hand.",
             r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{\rev{Junction augmentation, with R1@0.5 on the three benchmarks, on the 79 Clotho-Moment targets shorter than 5\,s, and on TACOS (mean and standard deviation over three seeds). End is the share of TACOS answers that start in the last second or later. The last row answers the whole recording regardless of the query. A standard deviation of 0.0 is below 0.05.}}",
             r"\label{tab:junction}", r"\begin{tabular}{l ccccc cc}", r"\toprule",
             r"Junction & Clotho-Moment & Clotho $<$5\,s & UnAV-100 & TUT-SE2017 & TACOS & TACOS mIoU & End (\%) \\", r"\midrule"]
    for lab, exp in JUNCTIONS:
        c = _r1_sets(exp, "clotho_moment", "default"); u = _r1_sets(exp, "unav100", "default"); t = _r1_sets(exp, "tut2017", "default")
        sh = _short_sets(exp)
        ta = _r1_sets(exp, "tacos", "tacos")
        if len(ta) == 0 and exp == "E4La_mixjunction":
            ta = _r1_sets(exp, "tacos", "first")
        lines.append(f"{lab} & {_ms(c, 0)} & {_ms(sh, 0)} & {_ms(u, 0)} & {_ms(t, 0)} & {_ms(ta, 0)} & {_ms(ta, 1)} & {_ms(ta, 2)} \\\\")
        if exp == "E1L_fi_ts_learned":
            lines.append(r"\midrule")
    lines += [r"\midrule", r"Whole recording (query-independent) & 0.2 & 0.0 & 13.0 & 8.7 & 39.2 & 45.2 & -- \\"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    open(f"{OUT}/junction_table.tex", "w").write("\n".join(lines) + "\n")


SHORT = [("AudioGround", "E1L_fi_ts_learned"), ("(a) Duration first", "E6La_duration_first"),
         ("(b) Short-target weighting", "E6La_reweight"), ("(c) Event targets", "E4La_strongcrop")]


def write_short():
    from agl_eval.metrics import max_iou
    g = GT[("clotho_moment", "original")]
    bins = [(0, 5), (5, 10), (10, 20), (20, 30)]
    lines = [r"% Generated by experiments/eval/paper_tables.py -- do not edit by hand.",
             r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
             r"\caption{\rev{Duration-aware objectives and event-level targets (R1@0.5 in \%). All rows are single training runs with the same seed. The groups below 5, 5--10, 10--20, and 20--30\,s contain 79, 262, 3,049, and 3,249 targets.}}",
             r"\label{tab:short}", r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcccccc@{}}", r"\toprule",
             r" & \multicolumn{5}{c}{Clotho-Moment} & \\", r"\cmidrule(lr){2-6}", r"Training & All & $<$5\,s & 5--10\,s & 10--20\,s & 20--30\,s & UnAV \\", r"\midrule"]
    for lab, exp in SHORT:
        f = f"{R}/{exp}/0/preds/default/clotho_moment_original.jsonl"
        if not (os.path.exists(f) and os.path.exists(f"{R}/{exp}/0/preds/default/DONE")):
            lines.append(f"{lab} & \\multicolumn{{6}}{{c}}{{\\pend{{running}}}} \\\\"); MISSING.append(f"short {exp}"); continue
        recs = T.build_records(g, T.load_preds([f]))
        I = np.array([max_iou(r["pred"], r["gt"]) if r["pred"] else 0.0 for r in recs])
        d = np.array([r["gt"][0][1] - r["gt"][0][0] for r in recs])
        h = (I >= .5 - 1e-6).astype(float)
        if exp == "E1L_fi_ts_learned":
            base_h = h; dcell = "--"
        else:
            rng = np.random.default_rng(0); n = len(h); idx = rng.integers(0, n, (10000, n)); x = (h[idx] - base_h[idx]).mean(1) * 100
            dcell = f"${100 * (h.mean() - base_h.mean()):+.1f}$ [{np.percentile(x, 2.5):.1f}, {np.percentile(x, 97.5):.1f}]"
        cells = [f"{100 * h.mean():.1f}"] + [f"{100 * (I[(d >= a) & (d < b)] >= .5 - 1e-6).mean():.1f}" for a, b in bins]
        u = _r1_sets(exp, "unav100", "default", (0,))
        cells.append(f"{u[0, 0]:.1f}" if len(u) else "--")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
        if exp == "E1L_fi_ts_learned":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    open(f"{OUT}/short_table.tex", "w").write("\n".join(lines) + "\n")


def write_protocols():
    """R3-5 / R4-3: R1@0.5 over parsed responses only (conference protocol) vs. over all queries (parse failure = miss)."""
    systems = [(l, ("base", m), 1) for l, m in BASE] + [(SALMONN[0], SALMONN[1], 1), (TIMEAUDIO[0], ("base", "timeaudio"), 1),
                                                         (QWENFT[0], QWENFT[1], 1), (OURS[0], OURS[1], 3)]
    lines = [r"% Generated by experiments/eval/paper_tables.py -- do not edit by hand.",
             r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.0pt}",
             r"\caption{\rev{Effect of excluding unparseable responses (Original setting, R1@0.5 in \%). Parsed is the parse-success rate; Excl. is R1 over parsed responses only, as in the conference version; All is R1 over all queries with unparseable responses counted as misses (used throughout this article).}}",
             r"\label{tab:protocols}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{l ccc ccc ccc}", r"\toprule",
             r"\multirow{2}{*}{Model} & \multicolumn{3}{c}{Clotho-Moment} & \multicolumn{3}{c}{UnAV-100} & \multicolumn{3}{c}{TUT-SE2017} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}", r"& Parsed & Excl. & All & Parsed & Excl. & All & Parsed & Excl. & All \\", r"\midrule"]
    for lab, src, need in systems:
        cells = []
        for b in BENCHES:
            st = stats(src, b, "original", need)
            cells += (["--"] * 3 if st is None else [f"{st['parse'][0]:.1f}", f"{st['old05'][0]:.1f}", f"{st['r05'][0]:.1f}"])
        lines.append(lab.replace("$^\\dagger$", "").replace("$^{\\dagger}$", "") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"]
    open(f"{OUT}/protocol_table.tex", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_protocols()
    write_junction()
    write_short()
    write_multi()
    write_ci()
    write_main()
    write_ablation()
    write_parse()
    print("missing / incomplete:")
    for m in sorted(set(MISSING)):
        print("  ", m)
    print("wrote", OUT)
