"""Trivial (model-free) baselines.

All functions take lists of unified records ({gt, L, ...}) and return per-query predictions
or R1 numbers. IoU for multi-GT queries = max over GT intervals.
"""
from __future__ import annotations

import numpy as np

from .clipping import clip_query
from .metrics import TOL


def _gt_arrays(records):
    """Pad GT intervals into arrays [n, G]."""
    G = max(len(r["gt"]) for r in records)
    s = np.full((len(records), G), np.nan)
    e = np.full((len(records), G), np.nan)
    for i, r in enumerate(records):
        for j, (a, b) in enumerate(r["gt"]):
            s[i, j], e[i, j] = a, b
    L = np.array([r["L"] for r in records], float)
    return s, e, L


def iou_matrix(ps, pe, gs, ge):
    """ps,pe: [C, n] candidate intervals per query; gs,ge: [n, G]. returns max-IoU [C, n]."""
    ps, pe = ps[..., None], pe[..., None]
    inter = np.clip(np.minimum(pe, ge[None]) - np.maximum(ps, gs[None]), 0, None)
    union = np.maximum(pe, ge[None]) - np.minimum(ps, gs[None])
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(union > 0, inter / union, 0.0)
    v = np.nan_to_num(v, nan=0.0)
    return v.max(-1)


def candidate_grid(maxL=60.0, step=1.0, min_len=None):
    pts = np.arange(0, maxL + 1e-9, step)
    c = [(a, b) for a in pts for b in pts if b > a and (min_len is None or b - a >= min_len)]
    return np.array(c)


def eval_fixed(records, cands, normalized=False):
    """Max-IoU [C, n] for each fixed candidate interval (clipped to [0, L])."""
    gs, ge, L = _gt_arrays(records)
    if normalized:
        ps, pe = cands[:, :1] * L[None], cands[:, 1:] * L[None]
    else:
        ps = np.minimum(cands[:, :1], L[None]) * np.ones_like(L)[None]
        pe = np.minimum(cands[:, 1:], L[None]) * np.ones_like(L)[None]
    return iou_matrix(ps, pe, gs, ge)


def tune_best_fixed(tune_records, cands, thr_obj=0.5, normalized=False, chunk=2000):
    best, best_v = None, -1
    for k in range(0, len(cands), chunk):
        M = eval_fixed(tune_records, cands[k:k + chunk], normalized)
        sc = (M >= thr_obj - TOL).mean(1)
        j = int(np.argmax(sc))
        if sc[j] > best_v:
            best_v, best = sc[j], cands[k + j]
    return best, 100 * best_v


def loo_best_fixed(records, cands, thr_obj=0.5, normalized=False, thrs=(0.5, 0.7)):
    """Leave-one-out: for query i choose the candidate maximising R1@thr_obj on the other
    n-1 queries (ties -> first in grid order), evaluate on i."""
    M = eval_fixed(records, cands, normalized)  # [C, n]
    H = (M >= thr_obj - TOL).astype(np.int32)
    tot = H.sum(1)
    hits = {t: [] for t in thrs}
    chosen = []
    for i in range(M.shape[1]):
        j = int(np.argmax(tot - H[:, i]))
        chosen.append(tuple(cands[j]))
        for t in thrs:
            hits[t].append(M[j, i] >= t - TOL)
    return {t: 100 * np.mean(v) for t, v in hits.items()}, chosen


def r1_of_interval(records, interval, normalized=False, thrs=(0.5, 0.7)):
    M = eval_fixed(records, np.array([interval], float), normalized)[0]
    return {t: 100 * float((M >= t - TOL).mean()) for t in thrs}, M


def centered(records, length, thrs=(0.5, 0.7)):
    gs, ge, L = _gt_arrays(records)
    c = L / 2
    ps, pe = np.clip(c - length / 2, 0, None), np.minimum(c + length / 2, L)
    M = iou_matrix(ps[None], pe[None], gs, ge)[0]
    return {t: 100 * float((M >= t - TOL).mean()) for t in thrs}, M


def whole(records, thrs=(0.5, 0.7)):
    gs, ge, L = _gt_arrays(records)
    M = iou_matrix(np.zeros_like(L)[None], L[None], gs, ge)[0]
    return {t: 100 * float((M >= t - TOL).mean()) for t in thrs}, M


def random_with_durations(records, durations, n_draws=200, seed=0, thrs=(0.5, 0.7)):
    """Random interval: duration sampled from the given empirical distribution (e.g. a
    model's parsed predicted durations), truncated to L, onset ~ U[0, L-d]. Returns expected
    R1 (mean over draws) and the std over draws."""
    rng = np.random.default_rng(seed)
    durations = np.asarray(durations, float)
    durations = durations[durations > 0]
    gs, ge, L = _gt_arrays(records)
    n = len(L)
    d = np.minimum(rng.choice(durations, size=(n_draws, n)), L[None])
    on = rng.uniform(0, 1, size=(n_draws, n)) * (L[None] - d)
    M = iou_matrix(on, on + d, gs, ge)
    out = {}
    for t in thrs:
        per_draw = 100 * (M >= t - TOL).mean(1)
        out[t] = float(per_draw.mean())
        out[f"{t}_std"] = float(per_draw.std())
    return out


def clip_records(records, eps=0.0, anchor="first"):
    """Apply the clipping procedure to annotation-only records (used for Clotho val)."""
    out = []
    for r in records:
        c = clip_query(r["gt"], r["L"], eps=eps, anchor=anchor)
        out.append(dict(r, gt=c["windows"], L=c["L_new"]))
    return out
