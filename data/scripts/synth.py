#!/usr/bin/env python
"""AudioGround-IT synthesis (reconstruction).

Pipeline per variant (all deterministic given cfg.seed):
  1. layout  : per-sample RNG = default_rng([seed, task_id, idx]).  Decides target/filler clips, crops, order.
               A *separate* junction stream default_rng([seed, task_id, idx, 7]) draws one u in [0,1) per
               junction, and a background stream ([.., 11]) draws background-bed parameters.  Hence two
               variants that differ only in junction mode share the exact same clip layout.
  2. timing  : junction mode maps (layout, u) -> absolute segment times and GT intervals.
  3. render  : mix segments (+ optional background bed) -> 16 kHz mono int16 wav.  Audio files are
               content-addressed (hash of layout+junction+bg), so identical samples across variants are
               rendered once and shared.
  4. write   : JSONL rows + stats.json.

Usage:  python synth.py --config ../configs/default.yaml [--override key=value ...] [--layout_only]
"""
import argparse, copy, hashlib, json, math, os, re, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

TASKS = ["grounding", "duration", "ordering", "frequency"]
TASK_ID = {t: i for i, t in enumerate(TASKS)}
SR = 16000

DEFAULT_CFG = dict(
    name="default",
    root="data",
    out_dir=None,                       # default: <root>/variants/<name>
    audio_store=None,                   # default: <root>/audio_store
    seed=20260101,
    n_per_task=dict(grounding=16230, duration=15230, ordering=9310, frequency=9130),   # = 49,900
    n_val=1000,
    n_val_reserved_clips=1500,          # clips used only for validation targets (never in train samples)
    tau=0.33,                           # filler/target and filler/filler cosine must be < tau
    compose_tau=None,                   # long-span composition: sim(anchor, extra) > compose_tau (None -> tau)
    len_bins=[20, 40, 60, 80, 100, 120],
    len_weights=dict(grounding=[.245, .245, .245, .245, .02], duration=[.245, .245, .245, .245, .02],
                     ordering=[.15, .25, .27, .27, .06], frequency=[.12, .24, .29, .29, .06]),
    n_onset_bins=5,
    span_bins=[1, 3, 5, 10, 20, 30],     # grounding / duration answer-span bins (s)
    span_weights=[.2, .2, .2, .2, .2],
    min_crop=1.0,
    freq_k=[2, 3, 4],
    nominal_gap=[0.3, 0.5],             # silence-gap range used for layout planning / bias balancing
    junction=dict(
        mode="silence",                 # silence | nogap | crossfade | overlap | bgmix
        gap=[0.3, 0.5],                 # silence (and bgmix foreground) gap range
        xfade=[0.5, 1.0],               # crossfade length range (equal-power)
        overlap=[0.5, 1.5],             # overlap length range (plain sum, 10 ms edge ramps)
        max_frac=0.4,                   # crossfade/overlap capped at max_frac * shorter neighbour
        label_boundary=None,            # midpoint | full ; None -> crossfade:midpoint, overlap:full
        bg_snr=[10.0, 20.0],            # bgmix: foreground-to-background loudness ratio (dB)
        bg_xfade=1.0,
    ),
    lead_prob=1.0,                      # fraction of samples that get the leading crop (v2: 1.0, v2.1: 0.9)
    lead=None,                          # [min, max] s: random-length leading filler crop (no answer starts at 0.0)
    strong_tighten=False,               # tighten target intervals with AudioSet-Strong matched events
    strong_min_len=0.3,
    response_style="unified",           # unified (intervals only) | conventional | both
    workers=64,
    max_tries=60,
)

# ----------------------------------------------------------------------------------------- templates
TPL = {
    "grounding": ['When does the sound "{q}" occur?', 'At what time in the audio can you hear "{q}"?',
                  'Locate the segment where the following happens: "{q}".', 'Give the start and end times of "{q}".',
                  'In which interval does "{q}" happen?', 'Identify when "{q}" is heard in the recording.',
                  'Ground the event "{q}" in time.', 'Find the timestamps for "{q}".'],
    "duration": ['How long does "{q}" last?', 'What is the duration of "{q}"? Give its time span.',
                 'For how long can "{q}" be heard?', 'Estimate how long "{q}" lasts in the audio.',
                 'How long is the event "{q}"?', 'Measure the duration of "{q}".',
                 'How much time does "{q}" take?', 'What is the time span of "{q}"?'],
    "ordering": ['Which happens first: "{a}" or "{b}"?', 'Does "{a}" occur before or after "{b}"?',
                 'Sort in time: "{a}"; "{b}".', 'Which comes earlier, "{a}" or "{b}"?',
                 'Is "{a}" heard before "{b}"?', 'Determine the temporal order of "{a}" and "{b}".'],
    "frequency": ['How many times does "{q}" occur in the audio?', 'Count the occurrences of "{q}" and list when each happens.',
                  'How often is "{q}" heard? Give every interval.', 'How many times can you hear "{q}"?',
                  'Find every occurrence of "{q}" in the recording.', 'Count how many times "{q}" happens in the clip.'],
}
RTPL = {
    "grounding": ["The sound occurs from {s} to {e} seconds.", "From {s} to {e} seconds.",
                  "It can be heard from {s} to {e} seconds.", "The event spans {s} to {e} seconds."],
    "duration": ["It lasts from {s} to {e} seconds.", "The event spans {s} to {e} seconds.",
                 "It is heard from {s} to {e} seconds.", "From {s} to {e} seconds."],
}
NUMW = {2: "two", 3: "three", 4: "four"}


def fmt(x):
    return f"{x:.1f}"


def iv_text(iv):
    return f"{fmt(iv[0])} to {fmt(iv[1])}"


def norm_caption(c):
    c = c.strip().rstrip(".").strip()
    if c and c[0].isupper() and (len(c) == 1 or not c[1].isupper()):     # keep acronyms (TV, ...)
        c = c[0].lower() + c[1:]
    return c


# ----------------------------------------------------------------------------------------- globals (fork)
G = {}


def load_globals(cfg):
    R = cfg["root"]
    pools = pd.read_parquet(f"{R}/pools/pools.parquet")
    E = np.load(f"{R}/emb/emb.npy")
    nbr = np.load(f"{R}/emb/nbr_idx.npy"); nbs = np.load(f"{R}/emb/nbr_sim.npy").astype(np.float32)
    dur = pools.dur.values.astype(np.float64)
    is_bg = pools.is_background.values
    fg = np.where(~is_bg)[0]
    rng = np.random.default_rng([cfg["seed"], 999])            # val reservation depends on seed only
    perm_fg = rng.permutation(fg)
    val_res = np.sort(perm_fg[:cfg["n_val_reserved_clips"]])
    train_pool = np.sort(perm_fg[cfg["n_val_reserved_clips"]:])
    allowed_train = np.zeros(len(pools), bool); allowed_train[train_pool] = True
    allowed_val = np.zeros(len(pools), bool); allowed_val[fg] = True        # val fillers may come from anywhere fg
    G.update(cfg=cfg, pools=pools, E=E, nbr=nbr, nbs=nbs, dur=dur, caps=pools.caption.values,
             bg_pool=np.where(is_bg)[0], train_pool=train_pool, val_res=val_res,
             allowed={"train": allowed_train, "val": allowed_val},
             fill_pool={"train": train_pool, "val": fg},
             target_perm={t: np.random.default_rng([cfg["seed"], 555, TASK_ID[t]]).permutation(train_pool) for t in TASKS},
             val_perm={t: np.random.default_rng([cfg["seed"], 556, TASK_ID[t]]).permutation(val_res) for t in TASKS})
    if cfg["strong_tighten"]:
        st = pd.read_parquet(f"{R}/pools/strong_events.parquet")
        st = st[st.matched]
        hull = st.groupby("row").agg(on=("on", "min"), off=("off", "max"))
        G["strong_hull"] = {int(r): (float(a), float(b)) for r, a, b in zip(hull.index, hull.on, hull.off)}


# ----------------------------------------------------------------------------------------- layout helpers
def pick_fillers(rng, split, forbid_rows, n, tau, max_draw=4000):
    """Pick n filler rows whose cosine with every forbidden row AND with each other is < tau."""
    E, pool = G["E"], G["fill_pool"][split]
    chosen, F = [], [E[r] for r in forbid_rows]
    used = set(int(r) for r in forbid_rows)
    drawn = 0
    while len(chosen) < n and drawn < max_draw:
        cand = pool[rng.integers(0, len(pool), 64)]
        drawn += 64
        M = np.stack(F) if F else None
        for c in cand:
            c = int(c)
            if c in used:
                continue
            if M is not None and np.max(M @ E[c]) >= tau:
                continue
            chosen.append(c); used.add(c); F.append(E[c]); M = np.stack(F)
            if len(chosen) == n:
                break
    return chosen if len(chosen) == n else None


def target_row(task, idx, split, attempt, rng):
    perm = G["val_perm"][task] if split == "val" else G["target_perm"][task]
    if attempt == 0:
        return int(perm[idx % len(perm)])
    return int(perm[rng.integers(0, len(perm))])


MIN_CROP_DBFS = -45.0      # clips are normalized to -23 LUFS; a target crop quieter than this is rejected


def audible(row, a, b):
    x = clip_audio(row)[int(a * SR):int(b * SR)]
    return len(x) > 0 and 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12) >= MIN_CROP_DBFS


def compose_span(rng, anchor, s, split, ctau, gap):
    """Build a contiguous target span of length ~s from anchor + similar clips (sim > ctau).
    Returns list of (row, a, b) crops, in playback order."""
    dur = G["dur"]
    if s <= dur[anchor] + 1e-6:
        a = rng.uniform(0, dur[anchor] - s)
        for _ in range(10):                      # re-draw (only on failure) crops that are near-silent
            if audible(anchor, a, a + s):
                return [(anchor, a, a + s)]
            a = rng.uniform(0, dur[anchor] - s)
        return None
    nb, ns = G["nbr"][anchor], G["nbs"][anchor]
    ok = (ns > ctau) & G["allowed"][split][nb] & (nb != anchor)
    cand = nb[ok][:50]
    # need k clips with sum(dur)+(k-1)*gap >= s
    order = list(rng.permutation(cand))
    rows, tot = [anchor], dur[anchor]
    while tot < s and order:
        r = int(order.pop()); rows.append(r); tot += gap + dur[r]
    if tot < s:
        return None
    rows = [int(r) for r in rng.permutation(rows)]               # anchor at random position inside the span
    # crop: the last clip in playback order is shortened so that the span is exactly s
    full = sum(dur[r] for r in rows[:-1]) + gap * (len(rows) - 1)
    piece = s - full
    if piece < 2.0:
        piece = min(2.0, dur[rows[-1]])
    a = 0.0 if rng.random() < 0.5 else dur[rows[-1]] - piece
    if not audible(rows[-1], a, a + piece):
        a = dur[rows[-1]] - piece - a            # try the other end of the clip
        if not audible(rows[-1], a, a + piece):
            return None
    segs = [(int(r), 0.0, float(dur[r])) for r in rows[:-1]] + [(int(rows[-1]), float(a), float(a + piece))]
    return segs


def onset_bin(x, n):
    return min(int(x * n), n - 1)


def len_bin(L, bins):
    for i in range(len(bins) - 1):
        if L < bins[i + 1]:
            return i
    return len(bins) - 2


def plan_times(lengths, gaps):
    """Start times with nominal gaps (layout planning)."""
    t, out = 0.0, []
    for i, l in enumerate(lengths):
        out.append(t); t += l + (gaps[i] if i < len(lengths) - 1 else 0)
    return out, t


def make_layout(task, idx, split):
    cfg = G["cfg"]
    tau = cfg["tau"]; ctau = cfg["compose_tau"] if cfg["compose_tau"] is not None else tau
    tid = TASK_ID[task]
    rng = np.random.default_rng([cfg["seed"], tid, idx, 1 if split == "val" else 0])
    ju = np.random.default_rng([cfg["seed"], tid, idx, 7, 1 if split == "val" else 0]).random(32)
    g0, g1 = cfg["nominal_gap"]
    ngaps = g0 + (g1 - g0) * ju
    bins = cfg["len_bins"]; nb = cfg["n_onset_bins"]; dur = G["dur"]
    lw = np.array(cfg["len_weights"][task], float); lw /= lw.sum()
    want_ob = int(rng.integers(0, nb))
    want_lb = int(rng.choice(len(lw), p=lw))
    for attempt in range(cfg["max_tries"]):
        if attempt and attempt % 15 == 0:            # cell looks infeasible: keep onset bin, resample length bin
            want_lb = int(rng.choice(len(lw), p=lw))
        lo, hi = bins[want_lb], bins[want_lb + 1]
        Lt = rng.uniform(lo, hi)
        X = target_row(task, idx, split, attempt, rng)
        # ---------------- target items (each item = list of segments played contiguously)
        if task in ("grounding", "duration"):
            sw = np.array(cfg["span_weights"], float); sw /= sw.sum()
            sb = int(rng.choice(len(sw), p=sw))
            s = rng.uniform(cfg["span_bins"][sb], cfg["span_bins"][sb + 1])
            s = max(s, cfg["min_crop"])
            if s > Lt - 8:                         # leave room for >= 1 filler
                continue
            span = compose_span(rng, X, s, split, ctau, 0.4)
            if span is None:
                continue
            targets = [span]; target_rows = [r for r, _, _ in span]; k = 1
        elif task == "ordering":
            Bs = pick_fillers(rng, split, [X], 1, tau)
            if Bs is None:
                continue
            Y = Bs[0]
            targets = [[(X, 0.0, dur[X])], [(Y, 0.0, dur[Y])]]; target_rows = [X, Y]; k = 2
        else:  # frequency
            ks = [kk for kk in cfg["freq_k"] if kk * dur[X] + (kk - 1) * 10.4 <= hi + 1e-6]
            if not ks:
                continue
            k = int(rng.choice(ks))
            targets = [[(X, 0.0, dur[X])] for _ in range(k)]; target_rows = [X]
        tlen = sum(sum(b - a for _, a, b in t) for t in targets)
        lead_len = float(rng.uniform(*cfg["lead"])) if cfg.get("lead") else 0.0
        if lead_len > 0 and rng.random() >= cfg.get("lead_prob", 1.0):   # v2.1: keep a few lead-free samples
            lead_len = 0.0
        # ---------------- number of fillers to reach Lt
        n_f_est = max(0, int(round((Lt - tlen - lead_len - 0.4 * (len(targets) - 1)) / 10.3)))
        if task == "frequency":
            n_f_est = max(n_f_est, k - 1)
        n_f_est = max(n_f_est, 1)
        fill = pick_fillers(rng, split, target_rows, n_f_est + 1, tau)
        if fill is None:
            continue
        lead_seg = None
        if lead_len > 0:                                # spare filler (drawn above, previously unused) -> lead crop
            lr_ = int(fill[n_f_est]); ln = min(lead_len, float(dur[lr_]))
            la = float(rng.uniform(0.0, float(dur[lr_]) - ln))
            lead_seg = dict(row=lr_, a=la, b=la + ln, role="filler", tgt=-1)
        fill = fill[:n_f_est]
        n_items = len(targets) + len(fill)
        # ---------------- enumerate positions; designated interval for onset balancing
        if task in ("grounding", "duration"):
            combos = [(p,) for p in range(n_items)]; desig = 0
        elif task == "ordering":
            combos = [(p, q) for p in range(n_items) for q in range(n_items) if p != q]; desig = 0   # first-mentioned
        else:
            from itertools import combinations
            combos = [c for c in combinations(range(n_items), k) if all(c[i + 1] - c[i] >= 2 for i in range(k - 1))]
            desig = int(rng.integers(0, k))
        rng.shuffle(combos)
        found = None
        for c in combos:
            items, fi = [], iter(fill)
            pos = {p: j for j, p in enumerate(c)}
            for p in range(n_items):
                items.append(("T", pos[p]) if p in pos else ("F", next(fi)))
            # flatten to segments
            segs = [dict(lead_seg)] if lead_seg else []
            for kind, v in items:
                if kind == "T":
                    for (r, a, b) in targets[v]:
                        segs.append(dict(row=int(r), a=float(a), b=float(b), role="target", tgt=int(v)))
                else:
                    segs.append(dict(row=int(v), a=0.0, b=float(dur[v]), role="filler", tgt=-1))
            starts, L = plan_times([s_["b"] - s_["a"] for s_ in segs], ngaps)
            if not (lo <= L < hi or (want_lb == len(bins) - 2 and lo <= L <= hi)):
                break                                   # length independent of order -> resample fillers
            ons = [starts[i] for i, s_ in enumerate(segs) if s_["role"] == "target" and s_["tgt"] == desig]
            if onset_bin(ons[0] / L, nb) == want_ob:
                found = (segs, L); break
        if found is None:
            continue
        segs, L = found
        return dict(task=task, idx=idx, split=split, segs=segs, ju=ju[:len(segs) - 1].tolist(), k=k,
                    designated=desig, want_onset_bin=want_ob, want_len_bin=want_lb, attempts=attempt + 1,
                    mention_swap=bool(rng.random() < 0.5), tpl=int(rng.integers(0, 1000)),
                    rtpl=int(rng.integers(0, 1000)))
    return None


# ----------------------------------------------------------------------------------------- timing per junction
def mixed_plan(lay, jcfg):
    """mode 'mixed': per recording a background bed with probability bg_prob; per junction one of nogap / crossfade /
    overlap, plus 'bggap' (a silence gap filled by the bed, never digital silence) when the recording has a bed."""
    if "jmodes" in lay:
        return lay["jmodes"], lay["has_bg"]
    cfg = G["cfg"]
    rng = np.random.default_rng([cfg["seed"], TASK_ID[lay["task"]], lay["idx"], 13, 1 if lay["split"] == "val" else 0])
    has_bg = bool(rng.random() < jcfg.get("bg_prob", 0.5))
    choices = list(jcfg.get("mix_modes", ["nogap", "crossfade", "overlap"])) + (["bggap"] if has_bg and jcfg.get("mix_bggap", False) else [])
    lay["jmodes"] = [str(rng.choice(choices)) for _ in range(len(lay["segs"]) - 1)]
    lay["has_bg"] = has_bg
    return lay["jmodes"], has_bg


def timing(lay, jcfg):
    """Return (starts, total_len, label_spans per segment [ls, le], per-junction length)."""
    segs = lay["segs"]; u = lay["ju"]; mode = jcfg["mode"]
    jm = mixed_plan(lay, jcfg)[0] if mode == "mixed" else [mode] * (len(segs) - 1)
    lens = [s["b"] - s["a"] for s in segs]
    lb = jcfg.get("label_boundary") or ("midpoint" if mode == "crossfade" else "full")
    starts, jl = [0.0], []
    for i in range(len(segs) - 1):
        m = jm[i]
        if m in ("silence", "bgmix", "bggap"):
            g0, g1 = jcfg["gap"]; d = g0 + (g1 - g0) * u[i]
        elif m == "nogap":
            d = 0.0
        elif m in ("crossfade", "overlap"):
            r0, r1 = jcfg["xfade"] if m == "crossfade" else jcfg["overlap"]
            d = -min(r0 + (r1 - r0) * u[i], jcfg["max_frac"] * min(lens[i], lens[i + 1]))
        else:
            raise ValueError(m)
        jl.append(d)
        starts.append(starts[-1] + lens[i] + d)
    L = starts[-1] + lens[-1]
    lab = []
    for i in range(len(segs)):
        s, e = starts[i], starts[i] + lens[i]
        if mode == "mixed":     # crossfade junctions: midpoint boundary; overlap junctions: full extent
            if i > 0 and jm[i - 1] == "crossfade": s += -jl[i - 1] / 2
            if i < len(segs) - 1 and jm[i] == "crossfade": e -= -jl[i] / 2
        elif mode in ("crossfade", "overlap") and lb == "midpoint":
            if i > 0: s += -jl[i - 1] / 2
            if i < len(segs) - 1: e -= -jl[i] / 2
        lab.append([s, e])
    return starts, L, lab, jl


def gt_intervals(lay, lab):
    """Exact (unrounded) GT intervals per target index, in playback order of targets."""
    segs = lay["segs"]
    per = {}
    for i, s in enumerate(segs):
        if s["role"] != "target":
            continue
        per.setdefault(s["tgt"], []).append((i, lab[i]))
    out = {}
    for t, lst in per.items():          # a target = contiguous run of segments -> outer edges
        out[t] = [lst[0][1][0], lst[-1][1][1]]
    return out


def strong_tighten(lay, starts, lab):
    """Tighten each target interval's outer edges using AudioSet-Strong matched-event hulls."""
    hull, segs = G["strong_hull"], lay["segs"]
    res, status = {}, {}
    for t in sorted(set(s["tgt"] for s in segs if s["role"] == "target")):
        idxs = [i for i, s in enumerate(segs) if s["role"] == "target" and s["tgt"] == t]
        i0, i1 = idxs[0], idxs[-1]
        S, Eend = lab[i0][0], lab[i1][1]
        st = "no_strong"
        for which, i in (("start", i0), ("end", i1)):
            s = segs[i]
            if s["row"] not in hull:
                continue
            h0, h1 = hull[s["row"]]
            a, b = max(s["a"], h0), min(s["b"], h1)
            if b - a < G["cfg"]["strong_min_len"]:
                st = "no_overlap"; continue
            if which == "start":
                S = max(S, starts[i] + (a - s["a"]))
            else:
                Eend = min(Eend, starts[i] + (b - s["a"]))
            st = "tightened"
        if Eend - S < G["cfg"]["strong_min_len"]:
            S, Eend, st = lab[i0][0], lab[i1][1], "collapsed_revert"
        res[t] = [S, Eend]; status[t] = st
    return res, status


# ----------------------------------------------------------------------------------------- render
def bg_plan(lay, L, jcfg):
    """Background bed: clips from the ambience pool with cosine < tau to every caption in the sample."""
    cfg = G["cfg"]
    rng = np.random.default_rng([cfg["seed"], TASK_ID[lay["task"]], lay["idx"], 11, 1 if lay["split"] == "val" else 0])
    E, pool, dur = G["E"], G["bg_pool"], G["dur"]
    rows = list({s["row"] for s in lay["segs"]})
    M = E[rows]
    ok = pool[np.max(E[pool] @ M.T, axis=1) < cfg["tau"]]
    if len(ok) == 0:
        ok = pool
    snr = rng.uniform(*jcfg["bg_snr"])
    need, picks, tot = L + 1.0, [], 0.0
    while tot < need:
        r = int(ok[rng.integers(0, len(ok))]); picks.append(r); tot += dur[r] - jcfg["bg_xfade"]
    off = float(rng.uniform(0, max(0.0, tot - need)))
    return dict(rows=picks, snr=float(snr), offset=off)


def clip_audio(row):
    mm = G.get("mm")
    if mm is None:
        mm = G["mm"] = np.memmap(f"{G['cfg']['root']}/clips/clips_16k.int16", dtype=np.int16, mode="r")
    p = G["pools"]
    o, n = int(p.offset.values[row]), int(p.n_samples.values[row])
    return mm[o:o + n].astype(np.float32) / 32768.0


def ramp(n):
    return np.sin(0.5 * np.pi * (np.arange(n) + 0.5) / max(n, 1))


def render(lay, starts, L, jl, jcfg, bg):
    mode = jcfg["mode"]
    jm = mixed_plan(lay, jcfg)[0] if mode == "mixed" else None
    N = int(round(L * SR)) + 1
    y = np.zeros(N, np.float32)
    segs = lay["segs"]
    for i, s in enumerate(segs):
        x = clip_audio(s["row"])
        a, b = int(round(s["a"] * SR)), int(round(s["b"] * SR))
        x = x[a:b].copy()
        if mode == "mixed":
            for side, j in (("in", i - 1), ("out", i)):
                if j < 0 or j >= len(jm):
                    continue
                n = int(round(-jl[j] * SR)) if jm[j] == "crossfade" else int(0.01 * SR) if jm[j] == "overlap" else 0
                n = min(n, len(x))
                if n > 0:
                    if side == "in": x[:n] *= ramp(n)
                    else: x[len(x) - n:] *= ramp(n)[::-1]
        elif mode == "crossfade":
            if i > 0:
                n = int(round(-jl[i - 1] * SR)); x[:n] *= ramp(n)
            if i < len(segs) - 1:
                n = int(round(-jl[i] * SR)); x[len(x) - n:] *= ramp(n)[::-1]
        elif mode == "overlap":
            n = int(0.01 * SR)
            if i > 0: x[:n] *= ramp(n)
            if i < len(segs) - 1: x[len(x) - n:] *= ramp(n)[::-1]
        st = int(round(starts[i] * SR))
        e = min(N, st + len(x))
        y[st:e] += x[:e - st]
    if mode in ("bgmix", "mixed") and bg is not None:
        import pyloudnorm as pyln
        xf = int(jcfg["bg_xfade"] * SR)
        bed = np.zeros(0, np.float32)
        for r in bg["rows"]:
            x = clip_audio(r).copy()
            if len(bed) == 0:
                bed = x; continue
            n = min(xf, len(x) // 2, len(bed) // 2)
            w = ramp(n).astype(np.float32)
            bed[-n:] = bed[-n:] * w[::-1] + x[:n] * w
            bed = np.concatenate([bed, x[n:]])
        o = int(bg["offset"] * SR)
        bed = bed[o:o + N]
        if len(bed) < N:
            bed = np.pad(bed, (0, N - len(bed)))
        meter = pyln.Meter(SR)
        lf, lb = meter.integrated_loudness(y.astype(np.float64)), meter.integrated_loudness(bed.astype(np.float64))
        if np.isfinite(lf) and np.isfinite(lb):
            bed *= 10 ** ((lf - bg["snr"] - lb) / 20)
        y = y + bed
    pk = float(np.max(np.abs(y))) if len(y) else 0.0
    if pk > 0.99:
        y *= 0.99 / pk
    return np.clip(np.round(y * 32767), -32768, 32767).astype(np.int16)


# ----------------------------------------------------------------------------------------- text
def build_text(lay, ivs):
    task, segs = lay["task"], lay["segs"]
    style = G["cfg"]["response_style"]
    caps = G["caps"]
    def qcap(t):
        rows = [s["row"] for s in segs if s["role"] == "target" and s["tgt"] == t]
        return norm_caption(caps[lay["anchor_rows"][t] if "anchor_rows" in lay else rows[0]])
    Lmax = math.floor(lay["_L"] * 10 + 1e-6) / 10          # rounded end never exceeds the audio length
    r = lambda iv: [round(iv[0], 1), min(round(iv[1], 1), Lmax)]
    if task in ("grounding", "duration"):
        q = qcap(0); iv = r(ivs[0])
        ins = TPL[task][lay["tpl"] % len(TPL[task])].format(q=q)
        if task == "duration" and style == "conventional":
            resp = f"It lasts about {fmt(iv[1] - iv[0])} seconds."
        elif task == "duration" and style == "both":
            resp = f"It lasts about {fmt(iv[1] - iv[0])} seconds, from {fmt(iv[0])} to {fmt(iv[1])} seconds."
        else:
            resp = RTPL[task][lay["rtpl"] % len(RTPL[task])].format(s=fmt(iv[0]), e=fmt(iv[1]))
        return ins, resp, [iv], [q]
    if task == "ordering":
        order = [0, 1] if not lay["mention_swap"] else [1, 0]      # mention order of the two targets
        qa, qb = qcap(order[0]), qcap(order[1])
        ia, ib = r(ivs[order[0]]), r(ivs[order[1]])
        ins = TPL[task][lay["tpl"] % len(TPL[task])].format(a=qa, b=qb)
        first = qa if ia[0] <= ib[0] else qb
        unified = f'"{qa}" occurs from {fmt(ia[0])} to {fmt(ia[1])} seconds, and "{qb}" occurs from {fmt(ib[0])} to {fmt(ib[1])} seconds.'
        if style == "conventional":
            resp = f'"{first}" happens first.'
        elif style == "both":
            resp = unified + f' So "{first}" happens first.'
        else:
            resp = unified
        return ins, resp, [ia, ib], [qa, qb]
    # frequency
    q = qcap(0)
    lst = [r(ivs[t]) for t in sorted(ivs)]
    ins = TPL[task][lay["tpl"] % len(TPL[task])].format(q=q)
    parts = [f"{fmt(a)} to {fmt(b)}" for a, b in lst]
    joined = ", ".join(parts[:-1]) + ", and " + parts[-1] if len(parts) > 2 else " and ".join(parts)
    unified = f"It occurs at {joined} seconds."
    if style == "conventional":
        resp = f"It occurs {NUMW[len(lst)]} times."
    elif style == "both":
        resp = f"It occurs {NUMW[len(lst)]} times: at {joined} seconds."
    else:
        resp = unified
    return ins, resp, lst, [q]


# ----------------------------------------------------------------------------------------- workers
def w_layout(args):
    task, idx, split = args
    lay = make_layout(task, idx, split)
    if lay is not None:
        lay["anchor_rows"] = anchor_rows(lay)
    return (task, idx, split, lay)


def anchor_rows(lay):
    """The caption used as the query for each target: for composed spans, the anchor clip is the one whose
    caption all other span clips are similar to (= the most central one)."""
    out = {}
    E = G["E"]
    for t in sorted(set(s["tgt"] for s in lay["segs"] if s["role"] == "target")):
        rows = [s["row"] for s in lay["segs"] if s["role"] == "target" and s["tgt"] == t]
        rows_u = list(dict.fromkeys(rows))
        if len(rows_u) == 1:
            out[t] = rows_u[0]
        else:
            M = E[rows_u] @ E[rows_u].T
            out[t] = rows_u[int(np.argmax(M.sum(1)))]
    return out


def layout_hash(lay, jcfg, bg):
    key = json.dumps(dict(segs=[(s["row"], round(s["a"], 4), round(s["b"], 4)) for s in lay["segs"]],
                          ju=[round(x, 6) for x in lay["ju"]], j=jcfg_key(jcfg), bg=bg), sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()


def jcfg_key(j):
    m = j["mode"]
    k = {"mode": m}
    if m in ("silence", "bgmix"): k["gap"] = j["gap"]
    if m == "crossfade": k.update(xfade=j["xfade"], max_frac=j["max_frac"])
    if m == "overlap": k.update(overlap=j["overlap"], max_frac=j["max_frac"])
    if m == "bgmix": k.update(bg_snr=j["bg_snr"], bg_xfade=j["bg_xfade"])
    if m == "mixed": k.update(gap=j["gap"], xfade=j["xfade"], overlap=j["overlap"], max_frac=j["max_frac"],
                              bg_snr=j["bg_snr"], bg_xfade=j["bg_xfade"], bg_prob=j.get("bg_prob", 0.5))
    return k


def w_finalize(lay):
    cfg = G["cfg"]; jcfg = cfg["junction"]
    starts, L, lab, jl = timing(lay, jcfg)
    ivs = gt_intervals(lay, lab)
    meta_strong = None
    if cfg["strong_tighten"]:
        ivs2, status = strong_tighten(lay, starts, lab)
        meta_strong = dict(status=status, untightened={t: [round(a, 3), round(b, 3)] for t, (a, b) in ivs.items()})
        ivs = ivs2
    bg = bg_plan(lay, L, jcfg) if (jcfg["mode"] == "bgmix" or (jcfg["mode"] == "mixed" and mixed_plan(lay, jcfg)[1])) else None
    h = layout_hash(lay, jcfg, bg)
    tag = jcfg["mode"]
    rel = f"{tag}/{h[:2]}/{h}.wav"
    path = os.path.join(cfg["audio_store"], rel)
    if not os.path.exists(path):
        y = render(lay, starts, L, jl, jcfg, bg)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".tmp{os.getpid()}.wav"
        sf.write(tmp, y, SR, subtype="PCM_16")
        os.replace(tmp, path)
    lay["_L"] = L
    ins, resp, gt, queries = build_text(lay, ivs)
    caps = G["caps"]; pools = G["pools"]
    nb = cfg["n_onset_bins"]
    desig_iv = ivs[lay["designated"]] if lay["task"] != "ordering" else ivs[1 if lay["mention_swap"] else 0]
    meta = dict(
        id=f"{lay['task'][:4]}_{lay['split']}_{lay['idx']:06d}", split=lay["split"], duration=round(L, 3),
        onset_norm=round(desig_iv[0] / L, 4), onset_bin=onset_bin(desig_iv[0] / L, nb),
        min_onset_norm=round(min(v[0] for v in ivs.values()) / L, 4),
        len_bin=len_bin(L, cfg["len_bins"]), want_onset_bin=lay["want_onset_bin"], want_len_bin=lay["want_len_bin"],
        span=round(sum(v[1] - v[0] for v in ivs.values()) / len(ivs), 3), k=lay["k"],
        junction=jcfg_key(jcfg), junction_len=[round(x, 3) for x in jl],
        gt_exact=[[round(ivs[t][0], 4), round(ivs[t][1], 4)] for t in
                  (([1, 0] if lay["mention_swap"] else [0, 1]) if lay["task"] == "ordering" else sorted(ivs))],
        segments=[dict(audiocap_id=int(pools.audiocap_id.values[s["row"]]), youtube_id=pools.youtube_id.values[s["row"]],
                       caption=caps[s["row"]], crop=[round(s["a"], 3), round(s["b"], 3)], role=s["role"], tgt=s["tgt"],
                       start=round(starts[i], 4), end=round(starts[i] + s["b"] - s["a"], 4))
                  for i, s in enumerate(lay["segs"])],
        mention_swap=lay["mention_swap"] if lay["task"] == "ordering" else None,
        background=dict(audiocap_ids=[int(pools.audiocap_id.values[r]) for r in bg["rows"]], snr_db=round(bg["snr"], 2),
                        offset=round(bg["offset"], 3)) if bg else None,
        strong=meta_strong, audio_sha1=h,
        junction_modes=mixed_plan(lay, jcfg)[0] if jcfg["mode"] == "mixed" else None,
    )
    return dict(audio=path, instruction=ins, response=resp, task=lay["task"], gt_intervals=gt, meta=meta)


def init_worker(cfg):
    if not G:
        load_globals(cfg)


# ----------------------------------------------------------------------------------------- stats
def compute_stats(rows, cfg):
    df = pd.DataFrame([dict(task=r["task"], split=r["meta"]["split"], L=r["meta"]["duration"],
                            words=len(r["instruction"].split()), ob=r["meta"]["onset_bin"],
                            mob=onset_bin(r["meta"]["min_onset_norm"], cfg["n_onset_bins"]),
                            lb=r["meta"]["len_bin"], span=r["meta"]["span"], k=r["meta"]["k"],
                            rows=[s["audiocap_id"] for s in r["meta"]["segments"]],
                            tg=[s["audiocap_id"] for s in r["meta"]["segments"] if s["role"] == "target"],
                            strong=(list(r["meta"]["strong"]["status"].values()) if r["meta"]["strong"] else []))
                       for r in rows])
    out = {}
    sb = cfg["span_bins"]
    for t in TASKS + ["total"]:
        d = df if t == "total" else df[df.task == t]
        o = dict(samples=int(len(d)), avg_len_s=round(d.L.mean(), 2), avg_query_words=round(d.words.mean(), 2),
                 total_hours=round(d.L.sum() / 3600, 2),
                 onset_bin_hist=np.bincount(d.ob, minlength=cfg["n_onset_bins"]).tolist(),
                 min_onset_bin_hist=np.bincount(d.mob, minlength=cfg["n_onset_bins"]).tolist(),
                 len_bin_hist=np.bincount(d.lb, minlength=len(cfg["len_bins"]) - 1).tolist(),
                 unique_clips=int(len(set(x for l in d.rows for x in l))),
                 unique_target_clips=int(len(set(x for l in d.tg for x in l))),
                 train=int((d.split == "train").sum()), val=int((d.split == "val").sum()))
        if t in ("grounding", "duration"):
            o["span_bin_hist"] = np.histogram(d.span.clip(sb[0], sb[-1]), bins=sb)[0].tolist()
            o["avg_span_s"] = round(d.span.mean(), 2)
        if t == "frequency":
            o["k_hist"] = {int(k): int(v) for k, v in d.k.value_counts().sort_index().items()}
        if cfg["strong_tighten"]:
            from collections import Counter
            o["strong_status"] = dict(Counter(x for l in d.strong for x in l))
        out[t] = o
    out["bins"] = dict(len_bins=cfg["len_bins"], n_onset_bins=cfg["n_onset_bins"], span_bins=sb)
    return out


# ----------------------------------------------------------------------------------------- main
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--override", nargs="*", default=[], help="dotted.key=yaml_value")
    ap.add_argument("--layout_only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="debug: samples per task")
    a = ap.parse_args()
    cfg = copy.deepcopy(DEFAULT_CFG)
    if a.config:
        deep_update(cfg, yaml.safe_load(open(a.config)) or {})
    for ov in a.override:
        k, v = ov.split("=", 1)
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d[kk]
        d[ks[-1]] = yaml.safe_load(v)
    cfg["out_dir"] = cfg["out_dir"] or f"{cfg['root']}/variants/{cfg['name']}"
    cfg["audio_store"] = cfg["audio_store"] or f"{cfg['root']}/audio_store"
    os.makedirs(cfg["out_dir"], exist_ok=True)
    yaml.safe_dump(cfg, open(f"{cfg['out_dir']}/config.resolved.yaml", "w"), sort_keys=False)

    # job list: first n_val_t indices of each task are validation (val targets from reserved clips)
    tot = sum(cfg["n_per_task"].values())
    nval = {t: int(round(cfg["n_val"] * n / tot)) for t, n in cfg["n_per_task"].items()}
    nval[TASKS[0]] += cfg["n_val"] - sum(nval.values())
    jobs = []
    for t in TASKS:
        n = cfg["n_per_task"][t] if a.limit is None else a.limit
        nv = nval[t] if a.limit is None else max(1, a.limit // 50)
        jobs += [(t, i, "val") for i in range(nv)] + [(t, i, "train") for i in range(n - nv)]
    t0 = time.time()
    load_globals(cfg)      # fork-shared
    with Pool(cfg["workers"], initializer=init_worker, initargs=(cfg,)) as pool:
        lays = pool.map(w_layout, jobs, chunksize=16)
        failed = [(t, i, s) for t, i, s, l in lays if l is None]
        print(f"layouts: {len(lays) - len(failed)} ok, {len(failed)} failed, {time.time() - t0:.0f}s", flush=True)
        # deterministic replacement for failures: append extra indices
        extra_idx = {t: (cfg["n_per_task"][t] if a.limit is None else a.limit) for t in TASKS}
        good = [l for _, _, _, l in lays if l is not None]
        for (t, i, s) in failed:
            while True:
                ni = 10_000_000 + extra_idx[t]; extra_idx[t] += 1
                l = pool.apply(w_layout, ((t, ni, s),))[3]
                if l is not None:
                    good.append(l); break
        if a.layout_only:
            rows = pool.map(w_stats_only, good, chunksize=32)
        else:
            rows = []
            for i, r in enumerate(pool.imap(w_finalize, good, chunksize=8)):
                rows.append(r)
                if i % 5000 == 0:
                    print(f"rendered {i}/{len(good)} {time.time() - t0:.0f}s", flush=True)
    rows.sort(key=lambda r: (r["meta"]["split"], TASK_ID[r["task"]], r["meta"]["id"]))
    rng = np.random.default_rng([cfg["seed"], 4242])
    for split in ("train", "val"):
        rs = [r for r in rows if r["meta"]["split"] == split]
        perm = rng.permutation(len(rs))
        if not a.layout_only:
            with open(f"{cfg['out_dir']}/{split}.jsonl", "w") as f:
                for j in perm:
                    f.write(json.dumps(rs[j], ensure_ascii=False) + "\n")
    st = compute_stats(rows, cfg)
    st["elapsed_s"] = round(time.time() - t0, 1); st["layout_failures_replaced"] = len(failed)
    json.dump(st, open(f"{cfg['out_dir']}/stats{'_layout' if a.layout_only else ''}.json", "w"), indent=1)
    print_table(st)


def w_stats_only(lay):
    """Layout-only dry run: timing + text, no rendering (for calibration)."""
    cfg = G["cfg"]; jcfg = cfg["junction"]
    starts, L, lab, jl = timing(lay, jcfg)
    ivs = gt_intervals(lay, lab)
    st = None
    if cfg["strong_tighten"]:
        ivs, status = strong_tighten(lay, starts, lab); st = dict(status=status)
    lay["_L"] = L
    ins, resp, gt, q = build_text(lay, ivs)
    nb = cfg["n_onset_bins"]
    d = ivs[lay["designated"]] if lay["task"] != "ordering" else ivs[1 if lay["mention_swap"] else 0]
    return dict(task=lay["task"], instruction=ins, response=resp, gt_intervals=gt,
                meta=dict(split=lay["split"], duration=L, onset_bin=onset_bin(d[0] / L, nb),
                          min_onset_norm=min(v[0] for v in ivs.values()) / L, len_bin=len_bin(L, cfg["len_bins"]),
                          span=sum(v[1] - v[0] for v in ivs.values()) / len(ivs), k=lay["k"], id=f"{lay['idx']}",
                          segments=[dict(audiocap_id=s["row"], role=s["role"]) for s in lay["segs"]], strong=st))


def print_table(st):
    cols = TASKS + ["total"]
    print(f"{'':22s}" + "".join(f"{c[:5]:>10s}" for c in cols))
    for k, lab in [("samples", "# samples"), ("avg_len_s", "avg len (s)"), ("avg_query_words", "avg query words"),
                   ("total_hours", "total hours")]:
        print(f"{lab:22s}" + "".join(f"{st[c][k]:>10}" for c in cols))
    for k in ["onset_bin_hist", "min_onset_bin_hist", "len_bin_hist"]:
        for c in cols:
            print(f"{k:20s} {c:10s} {st[c][k]}")
    for c in ("grounding", "duration"):
        print("span_bin_hist", c, st[c]["span_bin_hist"], "avg span", st[c]["avg_span_s"])
    print("freq k", st["frequency"]["k_hist"])


if __name__ == "__main__":
    main()
