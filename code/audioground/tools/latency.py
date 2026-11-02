"""End-to-end inference latency / peak GPU memory (paper Section V, R5 latency comment).

Protocol (as in the paper): batch size 1, FP16, one warm-up run, median over 8 runs, generation fixed to exactly
30 new tokens (min_new_tokens = max_new_tokens = 30), 4 beams as in evaluation, frozen encoders run live (no feature
cache), peak memory = torch.cuda.max_memory_allocated after reset_peak_memory_stats before each run.
Both systems use the same 30-s chunking of the encoders; the SALMONN baseline is the vanilla SALMONN connector
(no timestamp text, no ATE, zero-padded BEATs) applied to the same chunked input, i.e. E7_salmonn_zeroshot.

  CUDA_VISIBLE_DEVICES=4 python tools/latency.py --cfg <exp.yaml> [--ckpt best.pt] --wav <src.wav> --out <json>
"""
import argparse
import json
import os
import statistics
import time

import numpy as np
import soundfile as sf
import torch

from audioground import gpu_policy
from audioground.config import load_config
from audioground.data import AudioGroundDataset, collate
from audioground.model import AudioGround


def make_wav(src, dur, path):
    x, sr = sf.read(src, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    n = int(dur * sr)
    x = np.tile(x, int(np.ceil(n / len(x))))[:n]
    sf.write(path, x, sr)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--durations", default="60,90,120")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=30)
    ap.add_argument("--tmp", default="/tmp")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config([a.cfg], ["data.feature_cache=''"])
    gpu_policy.enforce("cuda")
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    model = AudioGround(cfg.model, compute_dtype=torch.float16)
    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"], strict=False)
    model.to(dev).eval().to(torch.float16)
    if model.use_beats:
        model.beats.float()
    tok = model.llama_tokenizer
    res = {"cfg": a.cfg, "ckpt": a.ckpt, "protocol": "bs1 fp16 beams4 30 tokens, 1 warmup + median of %d" % a.runs, "rows": []}
    for d in map(float, a.durations.split(",")):
        wav = make_wav(a.wav, d, os.path.join(a.tmp, f"lat_{int(d)}.wav"))
        row = dict(id="lat", audio=wav, instruction='Identify when "a dog barks" is heard in the recording.')
        ds = AudioGroundDataset([row], cfg.model, cfg.data, cfg.model.whisper_path, train=False)

        def once():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.perf_counter()
            batch = collate([ds[0]])   # includes audio loading / feature extraction, as in end-to-end use
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                at, _ = model.encode_audio(batch)
                emb, att, _, _ = model.build_inputs(at, batch["instruction"], None, left_pad=True)
                with model.autocast():
                    out = model.llama_model.generate(inputs_embeds=emb, attention_mask=att, num_beams=4, do_sample=False,
                                                     min_new_tokens=a.tokens, max_new_tokens=a.tokens,
                                                     pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id)
            torch.cuda.synchronize()
            return time.perf_counter() - t0, torch.cuda.max_memory_allocated(dev) / 2**20, emb.shape[1], out.shape[-1]   # input length incl. audio tokens

        once()
        ts, ms = [], []
        for _ in range(a.runs):
            t, m, ntok, nout = once()
            ts.append(t); ms.append(m)
        r = dict(duration=d, latency_s=statistics.median(ts), peak_mb=statistics.median(ms), input_tokens=int(ntok),
                 gen_len=int(nout), all_s=ts)
        print(json.dumps(r), flush=True)
        res["rows"].append(r)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
