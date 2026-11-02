"""Builds the offline frozen-encoder feature cache (Whisper + BEATs raw outputs, exact bf16) for every unique audio
file in the given JSONL files, for one chunking config. Run with torchrun.

  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 tools/build_feature_cache.py \
      --jsonl a.jsonl,b.jsonl [--set model.chunk_offset_sec=15] [--root cache/feats]

Files: <root>/<chunk_tag>/<hh>/<sha1>.w.npy (uint16 view of bf16, (N,1500,1280)), .b.npy ((N,~1496,768)), .json
(duration, chunks). Existing entries are skipped, so the build is resumable. A per-source index of
audio -> key is written to <root>/<chunk_tag>/index/<name>.tsv.
"""
import argparse
import json
import os
import sys
import threading
import time
import queue as Q

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audioground import gpu_policy  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import WhisperFeatureExtractor, WhisperModel  # noqa: E402

from audioground.audio import load_audio, plan_chunks, chunk_waveform, cache_paths, chunk_tag  # noqa: E402
from audioground.beats.BEATs import BEATs, BEATsConfig  # noqa: E402
from audioground.config import load_config  # noqa: E402


class AudioOnly(Dataset):
    def __init__(self, paths, m, sr, whisper_path):
        self.paths, self.m, self.sr = paths, m, sr
        self.fe = WhisperFeatureExtractor.from_pretrained(whisper_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        torch.set_num_threads(1)
        p = self.paths[i]
        try:
            wav = load_audio(p, self.sr, self.m.max_audio_sec)
        except Exception as e:  # unreadable file: report, skip
            return {"path": p, "error": repr(e)}
        dur = len(wav) / self.sr
        chunks = plan_chunks(dur, self.m.chunk_sec, self.m.chunk_hop_sec, self.m.chunk_offset_sec, self.m.keep_final_padding)
        cw, cpad = chunk_waveform(wav, chunks, self.sr)
        mel = self.fe(list(cw), sampling_rate=self.sr, return_tensors="np")["input_features"]
        return {"path": p, "mel": torch.from_numpy(mel), "wav": torch.from_numpy(cw), "pad": torch.from_numpy(cpad),
                "dur": dur, "chunks": chunks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="comma-separated JSONL files (rows with 'audio')")
    ap.add_argument("--cfg", action="append", default=[])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--root", default="")
    ap.add_argument("--name", default="", help="index name (default: derived from first jsonl)")
    ap.add_argument("--chunks_per_batch", type=int, default=48)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(a.cfg or [os.path.join(here, "configs/audioground_7b.yaml")], a.set)
    m, sr = cfg.model, cfg.data.sample_rate
    root = a.root or cfg.data.feature_cache
    gpu_policy.enforce("cuda")
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    dev = torch.device(f"cuda:{local}"); torch.cuda.set_device(dev)

    paths = []
    for f in a.jsonl.split(","):
        for l in open(f):
            if l.strip():
                p = json.loads(l)["audio"]
                paths.append(p if os.path.isabs(p) or not cfg.data.audio_root else os.path.join(cfg.data.audio_root, p))
    uniq = sorted(set(paths))
    tag = chunk_tag(m, sr)
    if rank == 0:
        name = a.name or os.path.basename(os.path.dirname(a.jsonl.split(",")[0])) + "_" + os.path.basename(a.jsonl.split(",")[0]).replace(".jsonl", "")
        os.makedirs(os.path.join(root, tag, "index"), exist_ok=True)
        with open(os.path.join(root, tag, "index", name + ".tsv"), "w") as f:
            for p in uniq:
                f.write(f"{p}\t{cache_paths(root, p, m, sr)[1]}\n")
    todo = [p for p in uniq[rank::world] if not os.path.exists(cache_paths(root, p, m, sr)[1] + ".json")]
    print(f"[rank {rank}] {tag}: {len(uniq)} unique audio, {len(todo)} to encode on this rank", flush=True)

    enc = WhisperModel.from_pretrained(m.whisper_path, torch_dtype=torch.bfloat16).encoder.to(dev).eval()
    beats = None
    if m.use_beats and m.beats_path:
        bck = torch.load(m.beats_path, map_location="cpu")
        beats = BEATs(BEATsConfig(bck["cfg"])); beats.load_state_dict(bck["model"]); beats.to(dev).eval()  # fp32 weights

    wq = Q.Queue(maxsize=256)

    def writer():
        while True:
            item = wq.get()
            if item is None:
                return
            p, w, b, dur, chunks = item
            d, base = cache_paths(root, p, m, sr)
            os.makedirs(d, exist_ok=True)
            np.save(base + ".w.npy.tmp.npy", w); os.replace(base + ".w.npy.tmp.npy", base + ".w.npy")
            if b is not None:
                np.save(base + ".b.npy.tmp.npy", b); os.replace(base + ".b.npy.tmp.npy", base + ".b.npy")
            with open(base + ".json.tmp", "w") as f:
                json.dump({"audio": p, "duration": dur, "chunks": chunks, "tag": tag}, f)
            os.replace(base + ".json.tmp", base + ".json")   # .json last = entry complete
    threads = [threading.Thread(target=writer, daemon=True) for _ in range(4)]
    [t.start() for t in threads]

    dl = DataLoader(AudioOnly(todo, m, sr, m.whisper_path), batch_size=None, shuffle=False, num_workers=a.workers,
                    prefetch_factor=8 if a.workers else None)
    buf, nbuf = [], 0
    tic = time.time(); done = 0; nch = 0; errors = []

    @torch.no_grad()
    def flush(buf):
        mel = torch.cat([x["mel"] for x in buf]).to(dev, torch.bfloat16, non_blocking=True)
        wav = torch.cat([x["wav"] for x in buf]).to(dev, non_blocking=True)
        pad = torch.cat([x["pad"] for x in buf]).to(dev, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sp = enc(mel, return_dict=True).last_hidden_state.to(torch.bfloat16)
            au = beats.extract_features(wav, padding_mask=pad, feature_only=True)[0].to(torch.bfloat16) if beats else None
        sp = sp.view(torch.int16).cpu().numpy().view(np.uint16)
        au = au.view(torch.int16).cpu().numpy().view(np.uint16) if au is not None else None
        o = 0
        for x in buf:
            n = x["mel"].shape[0]
            wq.put((x["path"], sp[o:o + n].copy(), au[o:o + n].copy() if au is not None else None, x["dur"], x["chunks"]))
            o += n

    for x in dl:
        if "error" in x:
            errors.append(x); continue
        buf.append(x); nbuf += x["mel"].shape[0]
        if nbuf >= a.chunks_per_batch:
            nflush = len(buf); flush(buf); done += nflush; nch += nbuf; buf, nbuf = [], 0
            if rank == 0 and (done // 500) != ((done - nflush) // 500):
                el = time.time() - tic
                print(f"[rank 0] {done}/{len(todo)} files, {nch / el:.1f} chunks/s, eta {(len(todo) - done) * el / max(done, 1) / 60:.1f} min",
                      flush=True)
    if buf:
        flush(buf); done += len(buf); nch += nbuf
    for _ in threads:
        wq.put(None)
    [t.join() for t in threads]
    print(f"[rank {rank}] done {done} files / {nch} chunks in {(time.time() - tic) / 60:.1f} min; {len(errors)} errors", flush=True)
    for e in errors[:20]:
        print("  ERROR", e["path"], e["error"], flush=True)
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
