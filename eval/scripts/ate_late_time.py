"""R5-M5: how the absolute time embedding behaves at later times, next to per-position recall.
(a) cosine similarity of ATE_l(t) vs ATE_l(t') over 0-120 s (seed 0)
(b) beta*||TE(t)||, (c) local resolution ||TE(t+1)-TE(t)|| / mean_t ||TE(t)||, for ATE_s / ATE_l / ATE_h (3 seeds)
(d) Clotho-Moment (Original) R1@0.5 by GT-centre bin, 3 seeds, with the training-target centre distribution (rho0 = 0%)
usage: <audioground env python> eval/scripts/ate_late_time.py --compute ; <eval env python> eval/scripts/ate_late_time.py
"""
import json, sys
import numpy as np
sys.path.insert(0, "code/audioground")
sys.path.insert(0, "eval")
from agl_eval.parser import parse_first  # noqa: E402
from agl_eval.metrics import max_iou  # noqa: E402

R = "runs"
OUT = "results/figures/ate_late_time"
VAR = [("ATE$_s$ (sinusoidal)", "sin", "E1L_fi_ts_sin", "#8a8a85", "--"),
       ("ATE$_l$ (learned, ours)", "learned", "E1L_fi_ts_learned", "#2a78d6", "-"),
       ("ATE$_h$ (hybrid)", "hybrid", "E1L_fi_ts_hybrid", "#eb6834", "-.")]
NPZ = "eval/results/ate_late_time.npz"
T = np.arange(0, 120.01, 1 / 3)


def compute():
    """step 1 (audioground env, torch): embeddings -> npz"""
    import torch
    out = {}
    with torch.no_grad():
        t = torch.tensor(T, dtype=torch.float32)
        for _, var, exp, _, _ in VAR:
            for s in (0, 1, 2):
                m = load_ate(var, f"{R}/{exp}/{s}")
                te = m.embed(t)
                n = te.norm(dim=-1)
                out[f"{exp}_{s}_norm"] = (m.beta.abs() * n).numpy()
                out[f"{exp}_{s}_res"] = ((te[3:] - te[:-3]).norm(dim=-1) / n.mean()).numpy()   # 1-s step = 3 grid points
                if var == "learned" and s == 0:
                    tn = te / te.norm(dim=-1, keepdim=True); out["cos_l0"] = (tn @ tn.T).numpy()
    np.savez(NPZ, **out)


def load_ate(variant, run):
    import torch
    from audioground.time_embedding import AbsoluteTimeEmbedding
    m = AbsoluteTimeEmbedding(variant, 4096)
    sd = torch.load(f"{run}/best.pt", map_location="cpu"); sd = sd.get("model", sd)
    m.load_state_dict({k[4:]: v.float() for k, v in sd.items() if k.startswith("ate.")})
    return m.eval()


def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": "#8a8a85"})
    fig = plt.figure(figsize=(7.2, 5.6)); gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32)
    axA, axB, axC, axD = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    report = {}
    Z = np.load(NPZ)
    cs = Z["cos_l0"]
    im = axA.imshow(cs, origin="lower", extent=[0, 120, 0, 120], cmap="Blues", vmin=cs.min(), vmax=1)
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.03)
    report["ATE_l cos(t, t+10s) at t=5,25,45,65,95"] = [round(float(cs[i, i + 30]), 4) for i in (15, 75, 135, 195, 285)]
    report["ATE_l cos min"] = round(float(cs.min()), 4)
    for label, var, exp, c, ls in VAR:
        norms = np.array([Z[f"{exp}_{s}_norm"] for s in (0, 1, 2)]); res = np.array([Z[f"{exp}_{s}_res"] for s in (0, 1, 2)])
        axB.plot(T, norms.mean(0), color=c, ls=ls, lw=1.5, label=label)
        axB.fill_between(T, norms.min(0), norms.max(0), color=c, alpha=.15, lw=0)
        tt = T[:-3] + 0.5
        axC.plot(tt, res.mean(0), color=c, ls=ls, lw=1.5)
        axC.fill_between(tt, res.min(0), res.max(0), color=c, alpha=.15, lw=0)
        report[f"{exp} beta*||TE|| at 5/30/60/100 s"] = [round(float(norms.mean(0)[i]), 3) for i in (15, 90, 180, 300)]
        report[f"{exp} 1-s resolution at 5/30/60/100 s"] = [round(float(res.mean(0)[i]), 4) for i in (15, 90, 180, 300)]
    axA.set_title("(a) ATE$_l$ cosine similarity", fontsize=8.5); axA.set_xlabel("t (s)"); axA.set_ylabel("t' (s)")
    axB.set_title(r"(b) Magnitude $\beta\,\|\mathrm{TE}(t)\|$", fontsize=8.5); axB.set_xlabel("t (s)")
    axC.set_title(r"(c) 1-s resolution $\|\mathrm{TE}(t{+}1)-\mathrm{TE}(t)\|\,/\,\overline{\|\mathrm{TE}\|}$", fontsize=8.5); axC.set_xlabel("t (s)")
    axC.set_yscale("log")
    # (d) recall by GT-centre bin
    bins = np.arange(0, 61, 10)
    for label, var, exp, c, ls in VAR:
        r = []
        for s in (0, 1, 2):
            rows = [json.loads(l) for l in open(f"{R}/{exp}/{s}/preds/default/clotho_moment_original.jsonl")]
            cen = np.array([(x["gt"][0][0] + x["gt"][0][1]) / 2 for x in rows])
            hit = np.array([max_iou(parse_first(x["raw_output"], "paper"), x["gt"]) >= 0.5 - 1e-6 if parse_first(x["raw_output"], "paper") else False for x in rows])
            r.append([100 * hit[(cen >= a) & (cen < b)].mean() for a, b in zip(bins[:-1], bins[1:])])
        r = np.array(r); x = (bins[:-1] + bins[1:]) / 2
        axD.errorbar(x, r.mean(0), yerr=r.std(0, ddof=1), color=c, ls=ls, marker="o", ms=3.5, lw=1.5, capsize=2)
        report[f"{exp} R1@0.5 by centre bin"] = [f"{m:.1f}±{s_:.1f}" for m, s_ in zip(r.mean(0), r.std(0, ddof=1))]
    cen_tr = [np.mean(g) for g in (json.loads(l)["gt_intervals"][0] for l in open("data/variants/lead/train.jsonl")
                                 if '"task": "grounding"' in l)]
    h, e = np.histogram(cen_tr, bins=np.arange(0, 121, 10)); h = 100 * h / h.sum()
    report["train grounding target centre share per 10 s (0-120)"] = [round(float(v), 1) for v in h]
    axD2 = axD.inset_axes([0.0, 0.0, 1.0, 0.22]); axD2.bar(x, h[:6], width=8, color="#c3c2b7"); axD2.axis("off")
    axD.set_title("(d) R1@0.5 by target center (bars: training share)", fontsize=8.5); axD.set_xlabel("target center (s)")
    axD.set_xticks(bins); axD.set_ylim(-28, 100)
    axD.set_yticks([0, 25, 50, 75, 100])
    for ax in (axB, axC, axD):
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e6e1", lw=0.6)
    axB.legend(fontsize=7, frameon=False, loc="upper left")
    fig.savefig(OUT + ".pdf", bbox_inches="tight"); fig.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
    json.dump(report, open("eval/results/ate_late_time.json", "w"), indent=1)
    for k, v in report.items():
        print(k, v)


if __name__ == "__main__":
    compute() if "--compute" in sys.argv else main()
