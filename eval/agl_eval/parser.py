"""Heuristic timestamp parser for free-form LALM responses.

Paper rule: "extract the first timestamp pair expressed using 'to', a hyphen, or 'and'.
When the predicted end precedes the start, the two values are reversed. Responses from
which no valid interval can be extracted are excluded" (old protocol).

Implementation details (the "handle units/variants" part):
  * a timestamp is   hh:mm:ss(.f) | mm:ss(.f) | number   optionally followed by a unit
    (ms, s, sec(s), second(s), min(s), minute(s)); minutes / ms are converted to seconds.
  * optional wrappers around a timestamp: [ ( < { " ' (e.g. "<3.0>", "[3.0s]") and a
    leading "t=" are tolerated.
  * connectors (mode="paper"): "to", "and", hyphen "-" plus its typographic variants
    en dash / em dash / minus sign (all are "hyphens" in model output).
    mode="extended" additionally accepts "starts at X ... ends at Y" phrasing, "~", "until", "till", "through", "," / ";" between
    two timestamps (e.g. "[3.0, 7.0]") -- reported separately, NOT used for paper numbers.
  * between the first timestamp and the connector we allow its unit and filler such as
    "seconds", "s", "sec", "the", "mark".
  * end < start -> swapped. start == end is kept as a (degenerate, IoU=0) parse.
  * parse_all(): multi-interval mode, returns every non-overlapping pair in order.
"""
from __future__ import annotations

import re

_UNIT = r"(?:milliseconds?|msecs?|ms|seconds?|secs?|s|minutes?|mins?|m(?![a-z]))"
_NUM = r"(?:\d{1,2}:)?\d{1,3}:\d{1,2}(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"
_OPEN = r"[\[\(<{\"'`]?\s*(?:t\s*=\s*)?"
_CLOSE = r"\s*[\]\)>}\"'`]?"
_TIME = rf"{_OPEN}(?P<NN>{_NUM})\s*(?P<NNu>{_UNIT})?\b{_CLOSE}"

_CONN_PAPER = r"(?:to|and|-|–|—|−)"
_CONN_EXT = r"(?:to|and|-|–|—|−|~|until|till|through|thru|,|;)"
_FILLER = r"(?:\s*(?:the|mark|seconds?|secs?|timestamp)\b)*"


def _build(conn):
    t1 = _TIME.replace("NN", "a")
    t2 = _TIME.replace("NN", "b")
    return re.compile(rf"{t1}{_FILLER}\s*{conn}\s*(?:the\s+)?{t2}", re.IGNORECASE)


_RE = {"paper": _build(_CONN_PAPER), "extended": _build(_CONN_EXT)}
# extended only: "starts at X seconds and ends at Y", "start time is X ... end time is Y"
_RE_SE = re.compile(rf"\b(?:start|begin|onset)\w*[^0-9]{{0,30}}?(?P<a>{_NUM})\s*(?P<au>{_UNIT})?\b.{{0,40}}?"
                    rf"\b(?:end|finish|stop|offset)\w*[^0-9]{{0,30}}?(?P<b>{_NUM})\s*(?P<bu>{_UNIT})?\b", re.IGNORECASE | re.S)


def _to_sec(num: str, unit: str | None) -> float:
    if ":" in num:
        parts = [float(p) for p in num.split(":")]
        v = 0.0
        for p in parts:
            v = v * 60 + p
        return v
    v = float(num)
    if unit:
        u = unit.lower()
        if u.startswith("ms") or u.startswith("msec") or u.startswith("milli"):
            v /= 1000.0
        elif u.startswith("min") or u == "m":
            v *= 60.0
    return v


def _match_to_pair(m):
    a = _to_sec(m.group("a"), m.group("au"))
    au, bu = m.group("au"), m.group("bu")
    # "1 to 2 minutes": unit stated only on the end applies to both
    if au is None and bu is not None and ":" not in m.group("a"):
        a = _to_sec(m.group("a"), bu)
    b = _to_sec(m.group("b"), bu)
    if b < a:
        a, b = b, a
    return [a, b]


def _clean(text: str) -> str:
    if text is None:
        return ""
    t = str(text)
    # strip special tokens of the form <|...|> but keep the number inside (e.g. <|3.50|>)
    t = re.sub(r"<\|\s*([0-9.:]+)\s*\|>", r" \1 ", t)
    t = t.replace(" ", " ")
    return t


def parse_first(text: str, mode: str = "paper"):
    """Return [start, end] in seconds or None."""
    t = _clean(text)
    m = _RE[mode].search(t)
    if m is None and mode == "extended":
        m = _RE_SE.search(t)
    return _match_to_pair(m) if m else None


def parse_all(text: str, mode: str = "paper"):
    """Multi-interval mode: all non-overlapping pairs, in textual order."""
    return [_match_to_pair(m) for m in _RE[mode].finditer(_clean(text))]


if __name__ == "__main__":
    tests = [
        "The sound occurs from 12.5 to 20.3 seconds.",
        "12.5s - 20.3s",
        "between 3 and 7 seconds",
        "From 00:05 to 00:12.",
        "<3.0> to <7.0>",
        "It ends at 30 and starts at 10",  # no pair -> "30 and starts" fails; returns None
        "The event happens 20 to 10 seconds",  # swapped
        "1:05-1:10",
        "[3.0, 7.0]",
        "No timestamps here.",
        "from 1 to 2 minutes",
        "10.0 – 15.5 s; and again 30.2 to 35.0 s",
    ]
    for t in tests:
        print(repr(t), parse_first(t), parse_first(t, "extended"), parse_all(t))
