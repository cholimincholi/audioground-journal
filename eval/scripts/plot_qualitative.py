"""Qualitative results (Section V-M): three Clotho-Moment queries (one solved by both models, two where the rho0=22% model breaks down), ground truth and the
predictions of the rho0=22% model (FI+TS+ATE_h) and AudioGround (rho0=0%, FI+TS+ATE_l), seed 0, Original setting.
usage: python eval/scripts/plot_qualitative.py [--out paper_v2/Figures/qualitative_results.pdf]
"""
import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "runs"
Q = [("37808", "Crowd cheering"), ("41695", "Gong"), ("43276", "Cat purring")]
MODELS = [("gt", "Ground truth", "#2e8b57"), ("E0_main", r"AudioGround ($\rho_0=22\%$, ATE$_h$)", "#b5461a"),
          ("E1L_fi_ts_learned", r"AudioGround ($\rho_0=0\%$, ATE$_l$)", "#2a78d6")]


def parse(t):
    m = re.findall(r"(\d+(?:\.\d+)?)", t)
    return (float(m[0]), float(m[1])) if len(m) >= 2 else None


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_v2/Figures/qualitative_results.pdf")
    a = ap.parse_args()
    P = {}
    for exp in ("E0_main", "E1L_fi_ts_learned"):
        for l in open(f"{R}/{exp}/0/preds/default/clotho_moment_original.jsonl"):
            d = json.loads(l); P[(exp, str(d["qid"]))] = d
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": "#4a4a46", "xtick.color": "#4a4a46"})
    fig, ax = plt.subplots(figsize=(7.16, 1.7))
    y = 0; yt = []
    for qid, name in Q:
        gt = P[("E1L_fi_ts_learned", qid)]["gt"][0]
        for k, (key, _, c) in enumerate(MODELS):
            iv = gt if key == "gt" else parse(P[(key, qid)]["raw_output"])
            yy = y - 0.34 * k
            ax.plot(iv, [yy, yy], color=c, lw=4.5, solid_capstyle="butt")
            if key != "gt":
                ax.text(61.0, yy, f"IoU {iou(iv, gt):.2f}", va="center", fontsize=6, color="#4a4a46")
        yt.append((y - 0.34, name)); y -= 1.3
    ax.set_yticks([p for p, _ in yt]); ax.set_yticklabels([n for _, n in yt], fontsize=8, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 66); ax.set_xticks(range(0, 61, 10)); ax.set_xlabel("Time (s)")
    ax.grid(axis="x", color="#eeeeea", lw=0.6); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], color=c, lw=4.5, label=lab) for _, lab, c in MODELS]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7.5)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
