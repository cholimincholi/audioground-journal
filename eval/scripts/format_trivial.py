"""results/trivial_baselines_tol*.json -> results/trivial_baselines.md"""
import json, os
R = "eval/results"
rows = [("valtuned_abs_obj0.5", "Best fixed, Clotho-val tuned (abs s, obj R1@.5)"),
        ("valtuned_norm_obj0.5", "Best fixed, Clotho-val tuned (normalized by L, obj R1@.5)"),
        ("valtuned_abs_obj0.7", "Best fixed, Clotho-val tuned (abs s, obj R1@.7)"),
        ("loo_abs_obj0.5", "Best fixed, leave-one-out (abs s, obj R1@.5)"),
        ("loo_norm_obj0.5", "Best fixed, leave-one-out (normalized, obj R1@.5)"),
        ("oracle_insample_abs", "Best fixed, in-sample oracle (upper bound)"),
        ("centered_10s", "Centered 10 s"), ("centered_20s", "Centered 20 s"), ("centered_26s", "Centered 26 s"),
        ("centered_30s", "Centered 30 s"), ("whole_window", "Whole 30-s window (clipped)"), ("whole_recording", "Whole recording"),
        ("random_gtdur", "Random onset, GT-duration prior (200 draws)"), ("random_dur26.5", "Random onset, 26.5 s"),
        ("random_dur10", "Random onset, 10 s")]
cols = [(b, s) for b in ("clotho_moment", "unav100", "tut2017") for s in ("original", "clipped")]
out = []
for tol in ("1e-06", "0"):
    d = json.load(open(f"{R}/trivial_baselines_tol{tol}.json"))
    out += [f"## IoU tolerance {tol}", "", "| Baseline | " + " | ".join(f"{b} {s}" for b, s in cols) + " |", "|---" * (1 + len(cols)) + "|"]
    for k, name in rows:
        cells = []
        for b, s in cols:
            v = d[f"{b}/{s}"].get(k)
            if v is None: cells.append("--"); continue
            r = v.get("r1", v)
            c = f"{r['0.5'] if '0.5' in r else r[0.5]:.2f} / {r['0.7'] if '0.7' in r else r[0.7]:.2f}"
            if "ci" in v and tol == "1e-06":
                c += f" [{v['ci']['0.5'][1]:.1f}-{v['ci']['0.5'][2]:.1f} / {v['ci']['0.7'][1]:.1f}-{v['ci']['0.7'][2]:.1f}]"
            if "interval" in v and k.startswith("valtuned_abs"): c += f" @{v['interval']}"
            cells.append(c)
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Clotho-val tuned intervals: " + "; ".join(f"{k.split('/',1)[1]}: {v['interval']} (val R1 {v['val_r1']:.2f}), norm {[round(x,3) for x in v['norm_interval']]}" for k, v in d.items() if k.startswith("clotho_val")))
    out.append("")
open(f"{R}/trivial_baselines.md", "w").write("# Trivial baselines (R1@0.5 / R1@0.7, %, [95% bootstrap CI, 10k])\n\n" + "\n".join(out) + "\n")
print("\n".join(out[:20]))
