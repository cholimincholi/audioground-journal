"""Unified benchmark loaders.

Each benchmark/setting has a JSONL at benchmarks/<bench>/<setting>.jsonl with records
    {qid, bench, setting, query, vid, audio (abs path), L (s), gt: [[s,e],...],
     clip_start, clip_branch}
built by scripts/build_benchmarks.py.
"""
from __future__ import annotations

import json
import os

from . import BENCH_ROOT

BENCHES = ("clotho_moment", "unav100", "tut2017")
SETTINGS = ("original", "clipped")
BENCH_PRETTY = {"clotho_moment": "Clotho-Moment", "unav100": "UnAV-100 subset", "tut2017": "TUT-SED 2017"}


def ann_path(bench, setting):
    return os.path.join(BENCH_ROOT, bench, f"{setting}.jsonl")


def load(bench, setting):
    with open(ann_path(bench, setting)) as f:
        return [json.loads(l) for l in f]


def load_index(bench, setting):
    return {str(r["qid"]): r for r in load(bench, setting)}
