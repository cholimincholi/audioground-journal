"""JSONL dataset for AudioGround.

Row format: {"audio": path, "instruction": str, "response": str, ...optional}
Optional fields used if present: "id", "gt_intervals": [[s, e], ...] or "span": [s, e] (seconds; E6 options).
Benchmark rows for inference may use "query"/"question" instead of "instruction" (see infer.py).
"""
import json
import math
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from transformers import WhisperFeatureExtractor

from .audio import FPS, load_audio, plan_chunks, chunk_waveform, cache_paths

_NUM = r"(\d+(?:\.\d+)?)"
SPAN_RE = re.compile(_NUM + r"\s*(?:s|sec|secs|seconds?)?\s*(?:to|-|–|~|and)\s*" + _NUM, re.I)


def parse_first_span(text):
    """First 'X to Y' / 'X - Y' / 'X and Y' pair (same rule as the paper's evaluation parser). None if absent."""
    m = SPAN_RE.search(text or "")
    if not m:
        return None
    s, e = float(m.group(1)), float(m.group(2))
    return (e, s) if e < s else (s, e)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def fmt_num(x):
    return f"{x:.1f}" if abs(x - round(x)) > 1e-9 else f"{int(round(x))}"


class AudioGroundDataset(Dataset):
    def __init__(self, rows, mcfg, dcfg, whisper_path, train=True):
        self.rows, self.m, self.d, self.train = rows, mcfg, dcfg, train
        self.fe = WhisperFeatureExtractor.from_pretrained(whisper_path)
        self.sr = dcfg.sample_rate
        # pre-compute target spans (for E6 reweighting / oversampling / aux loss)
        # span = first target interval: "span" field > "gt_intervals"[0] (AudioGround-IT) > parsed from response.
        # single[i]: exactly one target interval (only those feed the auxiliary span loss).
        self.spans, self.single = [], []
        for r in rows:
            gts = r.get("gt_intervals")
            sp = r.get("span") or (gts[0] if gts else None)
            if sp is None and train:
                sp = parse_first_span(r.get("response", ""))
            self.spans.append(tuple(map(float, sp)) if sp is not None else None)
            self.single.append(sp is not None and (gts is None or len(gts) == 1))

    def __len__(self):
        return len(self.rows)

    def span_duration(self, i):
        sp = self.spans[i]
        return None if sp is None else max(0.0, sp[1] - sp[0])

    def _path(self, p):
        return p if os.path.isabs(p) or not self.d.audio_root else os.path.join(self.d.audio_root, p)

    def _response(self, i):
        r = self.rows[i].get("response", "")
        if self.d.response_format == "duration_first" and self.spans[i] is not None and self.single[i]:
            s, e = self.spans[i]
            r = self.d.duration_first_template.format(dur=fmt_num(e - s), response=r, start=fmt_num(s), end=fmt_num(e))
        return r

    def _loss_weight(self, i):
        if self.d.loss_reweight == "inv_duration":
            dur = self.span_duration(i)
            if dur is None:
                return 1.0
            w = (self.d.reweight_ref_sec / max(dur, 0.1)) ** self.d.reweight_gamma
            return float(min(max(w, 1.0), self.d.reweight_max))
        return 1.0

    def _load_cached(self, path):
        _, base = cache_paths(self.d.feature_cache, path, self.m, self.sr)
        if not os.path.exists(base + ".json"):
            raise FileNotFoundError(f"feature cache miss for {path} ({base}); build it with tools/build_feature_cache.py "
                                    f"or set data.feature_cache=''")
        meta = json.load(open(base + ".json"))
        w = torch.from_numpy(np.ascontiguousarray(np.load(base + ".w.npy", mmap_mode="r")).view(np.int16)).view(torch.bfloat16)
        b = torch.from_numpy(np.ascontiguousarray(np.load(base + ".b.npy", mmap_mode="r")).view(np.int16)).view(torch.bfloat16) \
            if os.path.exists(base + ".b.npy") else None
        return meta["duration"], meta["chunks"], w, b

    def __getitem__(self, i):
        row = self.rows[i]
        path = self._path(row["audio"])
        feats = None
        if self.d.get("feature_cache"):
            dur, chunks, fw, fb = self._load_cached(path)
            feats = {"feat_w": fw, "feat_b": fb}
        else:
            wav = load_audio(path, self.sr, self.m.max_audio_sec)
            dur = len(wav) / self.sr
            chunks = plan_chunks(dur, self.m.chunk_sec, self.m.chunk_hop_sec, self.m.chunk_offset_sec,
                                 self.m.keep_final_padding)
            cw, cpad = chunk_waveform(wav, chunks, self.sr)
            mel = self.fe(list(cw), sampling_rate=self.sr, return_tensors="np")["input_features"]  # (N, 80, 3000)
        n_frames = max(c["start_frame"] + c["n_keep"] for c in chunks)
        sp = self.spans[i]
        span_norm = ((sp[0] / dur, sp[1] / dur) if (sp is not None and self.single[i] and dur > 0)
                     else (float("nan"), float("nan")))
        instr = row.get("instruction") or row.get("query") or row.get("question") or ""
        audio = feats if feats is not None else {"mel": torch.from_numpy(mel), "wav": torch.from_numpy(cw),
                                                  "wav_pad": torch.from_numpy(cpad)}
        return {
            "id": row.get("id", str(i)), **audio, "chunks": chunks, "n_frames": n_frames, "duration": dur,
            "instruction": instr, "response": self._response(i) if self.train else row.get("response", ""),
            "loss_weight": self._loss_weight(i), "span_norm": span_norm, "row_index": i,
        }


def collate(samples):
    cs, cst, ck = [], [], []
    for b, s in enumerate(samples):
        for c in s["chunks"]:
            cs.append(b); cst.append(c["start_frame"]); ck.append(c["n_keep"])
    if "feat_w" in samples[0]:   # cached encoder features
        audio = {"feat_w": torch.cat([s["feat_w"] for s in samples])}
        if samples[0]["feat_b"] is not None:
            audio["feat_b"] = torch.cat([s["feat_b"] for s in samples])
    else:
        audio = {"mel": torch.cat([s["mel"] for s in samples]), "wav": torch.cat([s["wav"] for s in samples]),
                 "wav_pad": torch.cat([s["wav_pad"] for s in samples])}
    return {
        **audio,
        "chunk_sample": torch.tensor(cs), "chunk_start": torch.tensor(cst), "chunk_keep": torch.tensor(ck),
        "n_frames": torch.tensor([s["n_frames"] for s in samples]),
        "duration": torch.tensor([s["duration"] for s in samples]),
        "instruction": [s["instruction"] for s in samples], "response": [s["response"] for s in samples],
        "loss_weight": torch.tensor([s["loss_weight"] for s in samples]),
        "span_norm": torch.tensor([s["span_norm"] for s in samples]),
        "id": [s["id"] for s in samples], "row_index": [s["row_index"] for s in samples],
    }


class InfiniteDistributedSampler(Sampler):
    """Infinite per-rank index stream. Epoch-wise reshuffle (seed + epoch); optional short-span oversampling
    (E6c) with step-dependent weights (schedule constant / decay / warmup)."""

    def __init__(self, dataset, rank, world, seed, micro_bs, oversample_factor=1.0, short_thr=5.0,
                 schedule="constant", max_steps=1, accum=1, target="short"):
        self.ds, self.rank, self.world, self.seed = dataset, rank, world, seed
        self.micro_bs, self.accum, self.max_steps = micro_bs, accum, max(1, max_steps)
        self.factor, self.schedule = float(oversample_factor), schedule
        n = len(dataset)
        if target == "strongcrop":   # oversample the instructions whose target is an AudioSet-Strong event crop
            self.is_short = np.array(["strongcrop" in (dataset.rows[i].get("meta") or {}) for i in range(n)], dtype=bool)
        else:                        # oversample targets shorter than short_thr seconds
            self.is_short = np.array([(d is not None and d < short_thr) for d in
                                      (dataset.span_duration(i) for i in range(n))], dtype=bool)
        self.epoch = 0

    def _factor_at(self, step):
        p = min(1.0, step / self.max_steps)
        if self.schedule == "decay":
            return self.factor + (1.0 - self.factor) * p
        if self.schedule == "warmup":
            return 1.0 + (self.factor - 1.0) * p
        return self.factor

    def __iter__(self):
        n = len(self.ds)
        drawn = 0
        while True:
            g = np.random.default_rng(self.seed + 1000003 * self.epoch)
            if self.factor == 1.0:
                perm = g.permutation(n)
                perm = perm[: (n // self.world) * self.world] if n >= self.world else np.resize(perm, self.world)
                for idx in perm[self.rank::self.world]:
                    drawn += 1
                    yield int(idx)
                self.epoch += 1
            else:  # weighted sampling with replacement, re-weighted every ~100 optimizer steps
                block = 100 * self.micro_bs * self.accum
                step = drawn // (self.micro_bs * self.accum)
                w = np.where(self.is_short, self._factor_at(step), 1.0); w = w / w.sum()
                idx = g.choice(n, size=block * self.world, replace=True, p=w)
                for j in idx[self.rank::self.world]:
                    drawn += 1
                    yield int(j)
                self.epoch += 1

    def __len__(self):
        return 2 ** 31
