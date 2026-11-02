#!/usr/bin/env python
"""Stage 1: decode OpenSound/AudioCaps train parquets -> mono 16 kHz, LUFS-normalized clips.

Output (in <out>/):
  clips_16k.int16   flat int16 memmap of all kept clips, concatenated
  clips.parquet     one row per AudioCaps train clip (incl. ones without audio):
                    audiocap_id, youtube_id, start_time, caption, has_audio, offset, n_samples,
                    dur, lufs_in, gain_db, peak, keep, drop_reason
"""
import argparse, glob, io, os, sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

SR = 16000
TARGET_LUFS = -23.0
PEAK_CEIL = 0.97          # sample-peak ceiling after gain (avoid clipping)
MIN_DUR = 3.0             # drop clips shorter than this (s)
MIN_LUFS = -60.0          # drop near-silent clips


def process_shard(path):
    meter = pyln.Meter(SR)
    t = pq.read_table(path)
    out = []
    for r in t.to_pylist():
        rec = dict(audiocap_id=r["audiocap_id"], youtube_id=r["youtube_id"], start_time=r["start_time"],
                   caption=r["caption"], audio=None, lufs_in=np.nan, gain_db=np.nan, peak=np.nan,
                   dur=0.0, keep=False, drop_reason="")
        b = (r.get("audio") or {}).get("bytes")
        if not b:
            rec["drop_reason"] = "no_audio_bytes"; out.append(rec); continue
        try:
            y, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        except Exception as e:  # noqa
            rec["drop_reason"] = f"decode_error"; out.append(rec); continue
        y = y.mean(axis=1)
        if sr != SR:
            g = gcd(sr, SR)
            y = resample_poly(y, SR // g, sr // g).astype(np.float32)
        rec["dur"] = len(y) / SR
        if rec["dur"] < MIN_DUR:
            rec["drop_reason"] = "too_short"; out.append(rec); continue
        try:
            L = meter.integrated_loudness(y.astype(np.float64))
        except Exception:
            L = -np.inf
        rec["lufs_in"] = L
        if not np.isfinite(L) or L < MIN_LUFS:
            rec["drop_reason"] = "silent"; out.append(rec); continue
        g_db = TARGET_LUFS - L
        z = y * (10 ** (g_db / 20))
        pk = float(np.max(np.abs(z)))
        if pk > PEAK_CEIL:            # peak-limit by simple scaling (loudness ends slightly below target)
            z *= PEAK_CEIL / pk
            g_db += 20 * np.log10(PEAK_CEIL / pk)
            pk = PEAK_CEIL
        rec.update(gain_db=g_db, peak=pk, keep=True)
        rec["audio"] = np.clip(np.round(z * 32767), -32768, 32767).astype(np.int16)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_glob", required=True)
    ap.add_argument("--captions", required=True, help="d0rj train parquet (full official caption list)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(a.parquet_glob))
    print("shards", len(files), flush=True)
    raw = os.path.join(a.out, "clips_16k.int16")
    rows, off = [], 0
    with open(raw, "wb") as fh, Pool(a.workers) as pool:
        for i, recs in enumerate(pool.imap(process_shard, files)):
            for r in recs:
                au = r.pop("audio")
                if au is not None:
                    fh.write(au.tobytes()); r["offset"] = off; r["n_samples"] = len(au); off += len(au)
                else:
                    r["offset"] = -1; r["n_samples"] = 0
                r["has_audio"] = True
                rows.append(r)
            if i % 20 == 0:
                print(f"{i+1}/{len(files)} shards, {len(rows)} clips", flush=True)
    df = pd.DataFrame(rows)
    # merge with the official caption list to count clips that have no audio at all
    cap = pd.read_parquet(a.captions)
    miss = cap[~cap.audiocap_id.isin(df.audiocap_id)].copy()
    miss = miss.assign(has_audio=False, keep=False, drop_reason="not_in_opensound", offset=-1, n_samples=0,
                       dur=0.0, lufs_in=np.nan, gain_db=np.nan, peak=np.nan)
    df = pd.concat([df, miss[df.columns]], ignore_index=True)
    df = df.drop_duplicates("audiocap_id").sort_values("audiocap_id").reset_index(drop=True)
    df.to_parquet(os.path.join(a.out, "clips.parquet"))
    print("official train captions:", len(cap), " in OpenSound:", int(df.has_audio.sum()),
          " kept:", int(df.keep.sum()))
    print(df.drop_reason.value_counts().to_string())
    print("kept dur stats:", df[df.keep].dur.describe().to_string())


if __name__ == "__main__":
    main()
