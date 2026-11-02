"""Metrics: IoU / R1, two protocols, multi-interval scoring, paired bootstrap, diagnostics.

Per-query records are dicts with at least
    qid, gt: [[s,e],...], pred: [s,e] or None, (optional) preds_all: [[s,e],...], L
Protocols
    old   : R1 over parsed queries only (paper protocol; unparsable responses excluded)
    miss  : parse failure counted as a miss; R1 over ALL queries
Multi-GT (single prediction): IoU = max over GT intervals (paper).
Multi-interval (TUT; uses preds_all from parse_all):
    set_f1@θ : one-to-one greedy matching of predicted and GT intervals at IoU>=θ,
               F1 = 2·TP/(|P|+|G|); averaged over queries (0 if no parse)
    mean_gt@θ: fraction of GT intervals covered by some prediction with IoU>=θ (recall),
               averaged over queries
    union_iou: IoU between union(pred intervals) and union(GT intervals) on the time axis
"""
from __future__ import annotations

import numpy as np

THRS = (0.5, 0.7)
# IoU tolerance: many GT/prediction pairs have IoU *analytically* equal to 0.5 or 0.7
# (e.g. a 15-s or 21-s target inside a 30-s prediction). Without a tolerance the hit
# decision depends on floating-point noise (Clotho clipped whole-window R1@0.5 moves
# between 83.4 and 84.6 depending on how coordinates are rounded). We count IoU >= θ - TOL.
import os as _os
TOL = float(_os.environ.get("AGL_IOU_TOL", "1e-6"))


def iou(p, g):
    if p is None:
        return 0.0
    inter = max(0.0, min(p[1], g[1]) - max(p[0], g[0]))
    union = max(p[1], g[1]) - min(p[0], g[0])
    return inter / union if union > 0 else 0.0


def max_iou(p, gts):
    return max((iou(p, g) for g in gts), default=0.0) if p is not None else 0.0


def per_query_hits(records, thrs=THRS):
    """Returns arrays: parsed (bool), iou (float; 0 when unparsed), hit[θ] (bool)."""
    parsed = np.array([r.get("pred") is not None for r in records], dtype=bool)
    ious = np.array([max_iou(r.get("pred"), r["gt"]) for r in records], dtype=float)
    hits = {t: (ious >= t - TOL) & parsed for t in thrs}
    return parsed, ious, hits


def r1(records, thrs=THRS):
    parsed, ious, hits = per_query_hits(records, thrs)
    n, npar = len(records), int(parsed.sum())
    out = dict(n=n, n_parsed=npar, parse_rate=100.0 * npar / n if n else float("nan"),
               miou_miss=100.0 * ious.mean() if n else float("nan"),
               miou_old=100.0 * ious[parsed].mean() if npar else float("nan"))
    for t in thrs:
        out[f"old@{t}"] = 100.0 * hits[t].sum() / npar if npar else float("nan")
        out[f"miss@{t}"] = 100.0 * hits[t].sum() / n if n else float("nan")
    return out


# ---------------------------------------------------------------- multi-interval
def _union_len(iv):
    iv = sorted([list(x) for x in iv if x[1] > x[0]])
    tot, cur = 0.0, None
    for a, b in iv:
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        tot += cur[1] - cur[0]
    return tot


def union_iou(P, G):
    if not P:
        return 0.0
    inter = 0.0
    for a, b in _merge(P):
        for c, d in _merge(G):
            inter += max(0.0, min(b, d) - max(a, c))
    u = _union_len(list(P) + list(G))
    return inter / u if u > 0 else 0.0


def _merge(iv):
    iv = sorted([list(x) for x in iv if x[1] > x[0]])
    out = []
    for a, b in iv:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def greedy_match(P, G, thr):
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(P) for j, g in enumerate(G)), reverse=True)
    up, ug, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < thr - TOL:
            break
        if i in up or j in ug:
            continue
        up.add(i); ug.add(j); tp += 1
    return tp


def multi_interval_scores(records, thrs=THRS):
    """Uses r['preds_all'] (list of intervals, possibly empty)."""
    res = {f"set_f1@{t}": [] for t in thrs}
    res.update({f"mean_gt@{t}": [] for t in thrs})
    res["union_iou"] = []
    for r in records:
        P, G = r.get("preds_all") or [], r["gt"]
        for t in thrs:
            tp = greedy_match(P, G, t) if P else 0
            res[f"set_f1@{t}"].append(2 * tp / (len(P) + len(G)) if P else 0.0)
            cov = sum(1 for g in G if any(iou(p, g) >= t - TOL for p in P))
            res[f"mean_gt@{t}"].append(cov / len(G))
        res["union_iou"].append(union_iou(P, G))
    return {k: np.array(v) for k, v in res.items()}


# ---------------------------------------------------------------- bootstrap
def _rng(seed):
    return np.random.default_rng(seed)


def _boot_idx(n, B, seed, chunk=1000):
    """Yield resample index blocks; identical stream for identical (n, B, seed) -> paired."""
    rng = _rng(seed)
    done = 0
    while done < B:
        k = min(chunk, B - done)
        yield rng.integers(0, n, size=(k, n))
        done += k


def _boot_stats(v, m, B, seed):
    out = []
    for idx in _boot_idx(len(v), B, seed):
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(100.0 * (v * m)[idx].sum(1) / m[idx].sum(1))
    return np.concatenate(out)


def bootstrap_ci(values, mask=None, B=10000, seed=0, alpha=0.05):
    """values: per-query 0/1 (or float) scores. mask: per-query inclusion (old protocol:
    parsed). The statistic sum(values*mask)/sum(mask) is recomputed on each resample of the
    n queries (so for the old protocol the denominator varies). Percentile 95% CI."""
    v = np.asarray(values, dtype=float)
    m = np.ones_like(v, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if len(v) == 0 or m.sum() == 0:
        return (float("nan"),) * 3
    stats = _boot_stats(v, m, B, seed)
    point = 100.0 * (v * m).sum() / m.sum()
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def paired_bootstrap_p(vA, vB, mA=None, mB=None, B=10000, seed=0):
    """Two-sided paired bootstrap test of metric(A) - metric(B) over the same queries
    (same resampled query indices for A and B). Returns (delta, ci_lo, ci_hi, p) with
    p = 2*min(P*(d<=0), P*(d>=0)) from the bootstrap distribution, floored at 1/B."""
    vA, vB = np.asarray(vA, float), np.asarray(vB, float)
    mA = np.ones_like(vA, bool) if mA is None else np.asarray(mA, bool)
    mB = np.ones_like(vB, bool) if mB is None else np.asarray(mB, bool)
    a = _boot_stats(vA, mA, B, seed)
    b = _boot_stats(vB, mB, B, seed)
    d = a - b
    d = d[~np.isnan(d)]
    delta = 100.0 * ((vA * mA).sum() / max(mA.sum(), 1) - (vB * mB).sum() / max(mB.sum(), 1))
    if len(d) == 0:
        return float(delta), float("nan"), float("nan"), float("nan")
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    p = float(min(1.0, max(p, 1.0 / B)))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(delta), float(lo), float(hi), p


def entry_with_ci(records, B=10000, seed=0, thrs=THRS):
    """All numbers for one (model, benchmark, setting) entry, with 95% CIs."""
    parsed, ious, hits = per_query_hits(records, thrs)
    out = r1(records, thrs)
    for t in thrs:
        h = hits[t].astype(float)
        out[f"old@{t}_ci"] = bootstrap_ci(h, parsed, B, seed)[1:]
        out[f"miss@{t}_ci"] = bootstrap_ci(h, None, B, seed)[1:]
    out["parse_rate_ci"] = bootstrap_ci(parsed.astype(float), None, B, seed)[1:]
    out["miou_miss_ci"] = bootstrap_ci(ious, None, B, seed)[1:]          # mIoU over all queries (unparsed = IoU 0)
    out["miou_old_ci"] = bootstrap_ci(ious, parsed, B, seed)[1:]
    return out


# ---------------------------------------------------------------- diagnostics
def prediction_diagnostics(records, bins_dur=(0, 5, 10, 20, 30, 1e9), bins_on=None):
    P = [r["pred"] for r in records if r.get("pred") is not None]
    if not P:
        return dict(n_parsed=0)
    P = np.array(P, float)
    dur, on = P[:, 1] - P[:, 0], P[:, 0]
    Ls = np.array([r.get("L", np.nan) for r in records if r.get("pred") is not None], float)
    uniq = len({(round(a, 1), round(b, 1)) for a, b in P})
    dh, _ = np.histogram(dur, bins=bins_dur)
    if bins_on is None:
        bins_on = (0, 10, 20, 30, 40, 50, 60, 1e9)
    oh, _ = np.histogram(on, bins=bins_on)
    out = dict(
        n_parsed=len(P),
        unique_intervals=uniq,
        unique_frac=uniq / len(P),
        dur_mean=float(dur.mean()), dur_std=float(dur.std()), dur_median=float(np.median(dur)),
        dur_p10=float(np.percentile(dur, 10)), dur_p90=float(np.percentile(dur, 90)),
        onset_mean=float(on.mean()), onset_std=float(on.std()),
        norm_onset_std=float(np.nanstd(on / Ls)) if np.isfinite(Ls).any() else float("nan"),
        frac_zero_len=float((dur <= 0).mean()),
        frac_out_of_range=float(np.nanmean(P[:, 1] > Ls + 1e-6)) if np.isfinite(Ls).any() else float("nan"),
        dur_hist={f"{bins_dur[i]}-{bins_dur[i+1]}": int(dh[i]) for i in range(len(dh))},
        onset_hist={f"{bins_on[i]}-{bins_on[i+1]}": int(oh[i]) for i in range(len(oh))},
    )
    top = {}
    for a, b in P:
        k = (round(a, 1), round(b, 1))
        top[k] = top.get(k, 0) + 1
    out["top5_intervals"] = sorted(top.items(), key=lambda x: -x[1])[:5]
    return out
