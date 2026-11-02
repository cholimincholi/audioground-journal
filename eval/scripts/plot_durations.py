"""Target duration on Clotho-Moment (R3-4, R4-2): (a) R1@0.5 by target duration, (b) mean predicted vs. ground-truth length
by target duration, (c) distribution of predicted lengths. AudioGround (rho0=22%, ATE_h) and AudioGround (rho0=0%, ATE_l),
three seeds pooled, Original setting.
usage: python eval/scripts/plot_durations.py [--out paper_v2/Figures/pred_durations.pdf]
"""
import argparse, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "experiments/eval")

R = "runs"
M = [("E0_main", r"AudioGround ($\rho_0=22\%$, ATE$_h$)", "#b5461a"), ("E1L_fi_ts_learned", r"AudioGround ($\rho_0=0\%$, ATE$_l$)", "#2a78d6")]
BINS = [(0, 5), (5, 10), (10, 20), (20, 30)]


def parse(t):
    m = re.findall(r"(\d+(?:\.\d+)?)", t)
    return (float(m[0]), float(m[1])) if len(m) >= 2 else None


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_v2/Figures/pred_durations.pdf")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 1.55), gridspec_kw={"wspace": 0.32})
    x = np.arange(len(BINS)); w = 0.38
    gt_all = None
    for m, (exp, lab, c) in enumerate(M):
        r1 = {b: [] for b in BINS}; pl = {b: [] for b in BINS}; gl = {b: [] for b in BINS}; d = []
        for s in (0, 1, 2):
            for l in open(f"{R}/{exp}/{s}/preds/default/clotho_moment_original.jsonl"):
                r = json.loads(l); g = r["gt"][0]; glen = g[1] - g[0]
                b = next((bb for bb in BINS if bb[0] <= glen < bb[1]), None)
                p = parse(r["raw_output"])
                if p and p[1] > p[0]:
                    d.append(p[1] - p[0])
                if b is None:
                    continue
                r1[b].append(bool(p) and iou(p, g) >= 0.5 - 1e-6); gl[b].append(glen)
                if p:
                    pl[b].append(abs(p[1] - p[0]))
        if gt_all is None:
            gt_all = [json.loads(l)["gt"][0] for l in open(f"{R}/{exp}/0/preds/default/clotho_moment_original.jsonl")]
        v = [100 * np.mean(r1[b]) for b in BINS]
        ax1.bar(x + (m - 0.5) * w, v, w, color=c, label=lab)
        ax2.plot([np.mean(gl[b]) for b in BINS], [np.mean(pl[b]) for b in BINS], marker="o", ms=3.5, color=c, lw=1.2)
        ax3.hist(d, bins=np.arange(0, 62, 2), weights=np.ones(len(d)) * 100 / len(d), histtype="step", lw=1.5, color=c)
        print(exp, "R1 by bin", [round(t, 1) for t in v], "pred mean", [round(np.mean(pl[b]), 1) for b in BINS])
    gl_all = [g[1] - g[0] for g in gt_all]
    ax3.hist(gl_all, bins=np.arange(0, 62, 2), weights=np.ones(len(gl_all)) * 100 / len(gl_all), color="#2e8b57", alpha=0.25, label="Ground truth")
    ax1.set_xticks(x); ax1.set_xticklabels(["<5", "5-10", "10-20", "20-30"]); ax1.set_xlabel("Target length (s)"); ax1.set_ylabel("R1@0.5 (%)")
    ax1.set_title("(a) Recall by target length", fontsize=7)
    ax2.plot([0, 30], [0, 30], ls=":", color="#888", lw=1); ax2.set_xlim(0, 30); ax2.set_ylim(0, 30)
    ax2.set_xlabel("Mean target length (s)"); ax2.set_ylabel("Mean predicted length (s)"); ax2.set_title("(b) Predicted vs. target length", fontsize=7)
    ax3.set_xlabel("Interval length (s)"); ax3.set_ylabel("Share of answers (%)"); ax3.set_title("(c) Distribution of lengths", fontsize=7)
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in M] + [plt.Rectangle((0, 0), 1, 1, color="#2e8b57", alpha=0.25)]
    fig.legend(h, [lab for _, lab, _ in M] + ["Ground truth"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=6.5)
    fig.savefig(a.out, bbox_inches="tight"); fig.savefig(a.out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
