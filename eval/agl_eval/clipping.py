"""Clipped-setting window selection (paper, Sec. Experiment / Evaluation).

For a recording of length L with target [ts, te]:
    c = (ts + te) / 2,  f = c / L
    if te - ts <= 30:
        s_min = max(0, te - 30);  s_max = min(ts, L - 30)
        s = clip(c - 30 f, s_min, s_max)
    else (target longer than 30 s):
        s = clip(c - 30 f, 0, L - 30)
    window = [s, s + 30]; target -> [max(0, ts - s), min(30, te - s)]
Recordings with L <= 30 are used unchanged.

NOTE on float precision: in Clotho-Moment test, two windows ([2.2,32.2], [28.2,58.2]) have
te - ts == 30.000000000000004 in IEEE double. Applied literally (eps=0) the paper's rule
sends them to the ">30 s" branch, which is what the paper reports ("two targets longer than
30 s"). `eps` > 0 treats them as 30-s targets instead. Default eps=0 reproduces the paper.

Multi-interval queries (TUT): the paper defines the procedure for one target. We anchor the
window on a single "anchor" interval (default: the span/hull [min ts, max te] of all GT
intervals, `anchor="span"`; alternative `anchor="first"`) and then remap *all* GT intervals,
truncating them to [0, 30] and dropping those that fall completely outside the window.
"""
from __future__ import annotations

WIN = 30.0


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def select_window(ts: float, te: float, L: float, win: float = WIN, eps: float = 0.0):
    """Return (s, branch) where branch in {'unchanged','fit','long'}."""
    if L <= win + eps:
        return 0.0, "unchanged"
    c = (ts + te) / 2.0
    f = c / L
    if (te - ts) <= win + eps:
        s_min = max(0.0, te - win)
        s_max = min(ts, L - win)
        # s_min <= s_max always holds when te-ts <= win and te <= L (up to eps)
        s = _clip(c - win * f, s_min, s_max) if s_min <= s_max else _clip(c - win * f, 0.0, L - win)
        return s, "fit"
    s = _clip(c - win * f, 0.0, L - win)
    return s, "long"


def remap(windows, s: float, win: float = WIN, keep_empty: bool = False, ndigits: int = 6):
    out = []
    for a, b in windows:
        na, nb = max(0.0, a - s), min(win, b - s)
        if nb - na <= 1e-9 and not keep_empty:
            continue
        out.append([round(na, ndigits), round(nb, ndigits)])
    return out


def clip_query(windows, L: float, win: float = WIN, eps: float = 0.0, anchor: str = "span"):
    """windows: list of [ts, te] GT intervals for one query, L: recording length (s).

    Returns dict(start=s, end=s+win (or L), branch, windows=remapped, L_new)."""
    if anchor == "span":
        ts, te = min(w[0] for w in windows), max(w[1] for w in windows)
    elif anchor == "first":
        ts, te = windows[0]
    else:
        raise ValueError(anchor)
    te = min(te, L)  # guard against annotation slightly beyond audio end
    s, branch = select_window(ts, te, L, win, eps)
    if branch == "unchanged":
        return dict(start=0.0, end=L, branch=branch, windows=[list(w) for w in windows], L_new=L)
    return dict(start=s, end=s + win, branch=branch, windows=remap(windows, s, win), L_new=win)
