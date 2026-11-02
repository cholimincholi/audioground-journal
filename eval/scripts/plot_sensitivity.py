"""Sensitivity to the design and data choices (Section V-F): one panel per varied choice, seed 0, Original setting.
Values are the R1@0.5 / R1@0.7 of the sensitivity runs (unified protocol); the default is AudioGround seed 0.
usage: python eval/scripts/plot_sensitivity.py [--out paper_v2/Figures/sensitivity.pdf]
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEF = {"clotho": (62.9, 52.1), "unav": (46.0, 33.0), "tut": (4.8, 1.9)}
PANELS = [  # (title, x label, default x, [(x, clotho, unav, tut)], log-x)
    ("Window length", r"$W_{\mathrm{sec}}$ (s)", 0.33,
     [(0.5, (60.3, 51.1), (42.0, 29.0), (11.5, 1.9)), (1.0, (40.9, 33.8), (36.0, 27.0), (9.6, 1.9))], False),
    ("Query tokens per window", r"$N_Q$", 1,
     [(2, (57.2, 45.9), (35.0, 27.0), (4.8, 0.0)), (4, (42.0, 29.8), (29.0, 18.0), (10.6, 1.9))], False),
    ("Initial embedding scale", r"$\beta_0$", 0.05,
     [(0.01, (62.1, 51.5), (34.0, 24.0), (5.8, 1.0)), (0.2, (55.3, 45.1), (37.0, 29.0), (9.6, 2.9))], True),
    ("Pairing threshold", r"$\tau$", 0.33,
     [(0.17, (60.6, 47.6), (38.0, 25.0), (5.8, 2.9)), (0.25, (54.7, 43.8), (43.0, 30.0), (5.8, 2.9)),
      (0.41, (57.4, 48.6), (32.0, 24.0), (6.7, 1.0)), (0.49, (55.8, 46.4), (38.0, 28.0), (4.8, 0.0))], False),
]
BENCH = [("clotho", "Clotho-Moment", "#2a78d6", "o"), ("unav", "UnAV-100 subset", "#e06a2c", "s"),
         ("tut", "TUT-Sound Events 2017", "#6b6b66", "^")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_v2/Figures/sensitivity.pdf")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#4a4a46", "xtick.color": "#4a4a46", "ytick.color": "#4a4a46"})
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.75), sharey=True, gridspec_kw={"wspace": 0.22})
    for ax, (title, xl, x0, pts, logx) in zip(axes, PANELS):
        rows = sorted([(x0, DEF["clotho"], DEF["unav"], DEF["tut"])] + pts)
        xs = [r[0] for r in rows]
        for k, (key, _, c, mk) in enumerate(BENCH):
            ax.plot(xs, [r[k + 1][0] for r in rows], color=c, marker=mk, ms=4, lw=1.5)
            ax.plot(xs, [r[k + 1][1] for r in rows], color=c, marker=mk, ms=3.5, lw=1.0, ls="--", mfc="white")
        for k, (key, _, c, mk) in enumerate(BENCH):
            ax.plot([x0], [DEF[key][0]], marker="*", ms=9, color=c, mec="black", mew=0.6, zorder=5, ls="none")
        if logx:
            ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=7.5)
        ax.minorticks_off()
        ax.set_xlabel(xl)
        ax.set_title(title, fontsize=8)
        ax.grid(axis="y", color="#eeeeea", lw=0.6)
        ax.set_ylim(-2, 70)
    axes[0].set_ylabel("Recall (%)")
    h = [Line2D([], [], color=c, marker=mk, ms=4, lw=1.5, label=lab) for _, lab, c, mk in BENCH]
    h += [Line2D([], [], color="#4a4a46", lw=1.5, label="R1@0.5"),
          Line2D([], [], color="#4a4a46", lw=1.0, ls="--", marker="o", ms=3.5, mfc="white", label="R1@0.7"),
          Line2D([], [], color="#4a4a46", marker="*", ms=8, mec="black", mew=0.6, ls="none", label=r"AudioGround ($\rho_0=0\%$, ATE$_l$)")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False, fontsize=7.5,
               handlelength=1.8, columnspacing=1.0)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
