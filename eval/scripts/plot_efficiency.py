"""Efficiency figure: end-to-end latency and peak GPU memory vs. input duration (same server and protocol for all three).
usage: envs/eval/bin/python eval/scripts/plot_efficiency.py"""
import json
import numpy as np
D = "eval/results/latency"
OUT = "results/figures/efficiency_analysis"
S = [("SALMONN baseline", "salmonn_baseline", "#8a8a85", "o"), ("AudioGround w/ ATE$_h$", "ate_h", "#eb6834", "s"),
     ("AudioGround (ATE$_l$)", "ate_l", "#2a78d6", "^")]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": "#8a8a85"})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"wspace": 0.35})
w = 7
for k, (lab, f, c, m) in enumerate(S):
    rows = json.load(open(f"{D}/{f}.json"))["rows"]
    x = np.array([r["duration"] for r in rows]) + (k - 1) * w
    a1.bar(x, [r["latency_s"] for r in rows], width=w - 1, color=c, label=lab)
    a2.bar(x, [r["peak_mb"] / 1024 for r in rows], width=w - 1, color=c)
for a, yl in ((a1, "Latency (s)"), (a2, "Peak GPU memory (GB)")):
    a.set_xticks([60, 90, 120]); a.set_xticklabels(["60 s", "90 s", "120 s"]); a.set_xlabel("Input duration")
    a.set_ylabel(yl); a.spines[["top", "right"]].set_visible(False); a.grid(axis="y", color="#e6e6e1", lw=0.6); a.set_axisbelow(True)
a1.set_ylim(0, 1.2); a2.set_ylim(0, 19)
h, l = a1.get_legend_handles_labels()
fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=3)
fig.savefig(OUT + ".pdf", bbox_inches="tight"); fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight"); print("wrote", OUT)
