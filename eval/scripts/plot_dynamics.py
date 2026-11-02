"""Training-dynamics figure: Clotho-Moment (full test, Original) R1@0.5 and breakdown rate (onset 0.0 s and IoU = 0) vs. training step,
for the v1 / v2.1 / v2 constructions (one seed each, checkpoint every 500 steps), plus three-seed final results.

usage: python eval/scripts/plot_dynamics.py [--out paper_v2/Figures/training_dynamics.pdf]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "eval")
from agl_eval.parser import parse_first  # noqa: E402
from agl_eval.metrics import max_iou  # noqa: E402

R = "runs"
# (label, dynamics run, final 3-seed runs, colour, linestyle, marker) - reference categorical slots 2, 3, 1
SERIES = [   # biased constructions on one orange ramp (darker = more answers at 0.0 s); the unbiased v2 in blue
    ("$\\rho_0=22\\%$ (conference)", "E0_dyn/1", "E0_main", "#7a2e0b", "-", "o"),
    ("$\\rho_0=13.7\\%$", "E0D04_dyn/1", "E0D04_main", "#b5461a", "-", "D"),
    ("$\\rho_0=9.4\\%$", "E0D06_dyn/1", "E0D06_main", "#e06a2c", "-", "v"),
    ("$\\rho_0=4.9\\%$", "E0D08_dyn/1", "E0D08_main", "#f0995e", "-", "P"),
    ("$\\rho_0=2.6\\%$", "E0M_dyn/1", "E0M_main", "#f6c49a", "--", "s"),
    ("$\\rho_0=0\\%$", "E0L_dyn/1", "E0L_main", "#2a78d6", "-", "^"),
]
GT_ONSET0 = 0.26  # % of Clotho-Moment test targets starting at exactly 0.0 s


def score(path):
    rows = [json.loads(l) for l in open(path)]
    P = [parse_first(r["raw_output"], "paper") for r in rows]
    iou = np.array([max_iou(p, r["gt"]) if p else 0.0 for p, r in zip(P, rows)])
    # breakdown = the prediction starts at exactly 0.0 s AND does not overlap the target at all (IoU = 0)
    brk = np.array([p is not None and p[0] == 0.0 for p in P]) & (iou == 0)
    return 100 * (iou >= 0.5 - 1e-6).mean(), 100 * brk.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures/training_dynamics.pdf")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#8a8a85",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#4a4a46", "ytick.color": "#4a4a46"})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 4.6), sharex=True, gridspec_kw={"hspace": 0.12})
    table = {}
    for label, dyn, final, c, ls, mk in SERIES:
        pts = []
        for d in sorted(glob.glob(f"{R}/{dyn}/preds/dyn_step*")):
            f = f"{d}/clotho_moment_original.jsonl"
            if os.path.exists(f"{d}/DONE") and os.path.exists(f):
                pts.append((int(d.split("step")[-1]),) + score(f))
        if pts:
            s, r1, z = map(np.array, zip(*pts))
            ax1.plot(s, r1, ls=ls, marker=mk, ms=4, lw=1.6, color=c, label=label)
            ax2.plot(s, z, ls=ls, marker=mk, ms=4, lw=1.6, color=c)
            table[label] = pts
        fin = [score(f"{R}/{final}/{k}/preds/default/clotho_moment_original.jsonl")
               for k in (0, 1, 2) if os.path.exists(f"{R}/{final}/{k}/preds/default/clotho_moment_original.jsonl")]
        if len(fin) == 3:
            r1f, zf = np.array(fin).T
            x = 6350
            ax1.errorbar(x, r1f.mean(), yerr=r1f.std(ddof=1), fmt=mk, ms=5, color=c, capsize=3, lw=1.2)
            ax2.errorbar(x, zf.mean(), yerr=zf.std(ddof=1), fmt=mk, ms=5, color=c, capsize=3, lw=1.2)
            table[label + " | final 3 seeds"] = (r1f.mean(), r1f.std(ddof=1), zf.mean())
    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#e6e6e1", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    ax1.set_ylabel("Clotho-Moment\nR1@0.5 (%)")
    ax2.set_ylabel("Breakdown (%)\nonset 0.0 s, IoU = 0")
    ax2.set_xlabel("Training step")
    ax2.set_ylim(-3, 103)
    ax1.set_ylim(0, 70)
    ax2.set_xticks([500, 1000, 2000, 3000, 4000, 5000, 6000, 6350])
    ax2.set_xticklabels(["500", "1k", "2k", "3k", "4k", "5k", "6k", "final\n(3 seeds)"])
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=4, lw=1.6, label=label) for label, _, _, c, ls, mk in SERIES]
    leg = ax1.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.08, 1.02), ncol=6, fontsize=7.5, frameon=False,
                     handlelength=1.3, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.0, borderpad=0.0)
    fig.canvas.draw()
    lb, ab = leg.get_window_extent(), ax1.get_window_extent()
    print(f"legend x {lb.x0:.0f}-{lb.x1:.0f}, axes x {ab.x0:.0f}-{ab.x1:.0f}")
    for ax in (ax1, ax2):
        ax.axvline(6175, color="#c3c2b7", lw=0.8, ls="--")
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    for k, v in table.items():
        print(k, v if isinstance(v, tuple) else [(s, round(r, 1), round(z, 1)) for s, r, z in v])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
