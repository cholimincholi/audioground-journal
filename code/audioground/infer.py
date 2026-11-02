"""AudioGround inference: raw generations per query -> JSONL (all input fields kept + "raw_output").

  torchrun --nproc_per_node 8 infer.py --run_dir runs/E0_main/0 --ckpt best.pt \
      --input a.jsonl,b.jsonl --output pa.jsonl,pb.jsonl [--set model.chunk_offset_sec=15]

Input rows: {"id", "audio", "instruction"} or {"id", "audio", "query"} (query is wrapped with
--instruction_template). Ground-truth fields are passed through untouched.
Defaults: beam search 4 beams, no sampling, max 100 new tokens (paper evaluation protocol).
"""
import argparse
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audioground import gpu_policy  # noqa: E402  (sets CUDA_DEVICE_ORDER before torch)
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audioground.config import load_config
from audioground.data import AudioGroundDataset, collate, read_jsonl
from audioground.model import AudioGround

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
# matches the AudioGround-IT grounding instruction style (data/variants/*/train.jsonl, task=grounding)
DEFAULT_TEMPLATE = 'Identify when "{query}" is heard in the recording.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="", help="training output dir (config.yaml + ckpt)")
    ap.add_argument("--ckpt", default="best.pt", help="ckpt path or name inside run_dir; '' = no AudioGround ckpt")
    ap.add_argument("--cfg", action="append", default=[], help="extra/alternative config yaml(s)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. model.chunk_offset_sec=15")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--instruction_template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--precision", default="bf16", help="bf16 matches training; paper latency profiling used fp16")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfgs = ([os.path.join(args.run_dir, "config.yaml")] if args.run_dir else []) + args.cfg
    cfg = load_config(cfgs, args.set)
    gpu_policy.enforce(cfg.train.device)
    # inference must not re-merge SALMONN LoRA on top of an AudioGround ckpt trained from the merged base:
    # merge is deterministic, so building the model the same way and loading trainable weights reproduces training.
    use_cuda = cfg.train.device == "cuda" and torch.cuda.is_available()
    if "RANK" in os.environ:
        dist.init_process_group("nccl" if use_cuda else "gloo")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    dev = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(dev)
    dtype = DTYPES[args.precision] if use_cuda else torch.float32
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, format="%(asctime)s %(message)s")
    log = logging.getLogger("infer")

    model = AudioGround(cfg.model, compute_dtype=dtype)
    if args.ckpt:
        ck = args.ckpt if os.path.isabs(args.ckpt) or not args.run_dir else os.path.join(args.run_dir, args.ckpt)
        sd = torch.load(ck, map_location="cpu")["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys in ckpt: {unexpected[:5]}"
        log.info(f"loaded {len(sd)} tensors from {ck}")
    model.to(dev).eval()
    if use_cuda:
        model.to(dtype)  # inference: everything in half precision
        if model.use_beats:
            model.beats.float()  # weight_norm needs fp32 weights (autocast handles compute)

    inputs, outputs = args.input.split(","), args.output.split(",")
    assert len(inputs) == len(outputs), "--input and --output must list the same number of files"
    for inp, outp in zip(inputs, outputs):   # model is loaded once for all benchmark files
        run_file(inp, outp, args, cfg, model, dev, rank, world, log)
    if dist.is_initialized():
        dist.destroy_process_group()


def run_file(inp, outp, args, cfg, model, dev, rank, world, log):
    rows = read_jsonl(inp)
    if args.limit:
        rows = rows[: args.limit]
    for i, r in enumerate(rows):
        r.setdefault("id", str(r.get("qid", i)))
        r["_order"] = i
        if not r.get("instruction"):
            r["instruction"] = args.instruction_template.format(query=(r.get("query") or r.get("question") or "").strip(),
                                                                L=float(r.get("L") or 0.0))
    shard = rows[rank::world]
    ds = AudioGroundDataset(shard, cfg.model, cfg.data, cfg.model.whisper_path, train=False)
    dl = DataLoader(ds, batch_size=cfg.generate.batch_size, shuffle=False, num_workers=min(4, cfg.data.num_workers),
                    collate_fn=collate)
    part = f"{outp}.part{rank}"
    os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
    tic = time.time(); n = 0
    with open(part, "w") as f:
        for batch in dl:
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            outs = model.generate(batch, cfg.generate)
            for j, (ri, o) in enumerate(zip(batch["row_index"], outs)):
                rec = dict(shard[ri]); rec["raw_output"] = o
                rec["audio_duration"] = float(batch["duration"][j])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += len(outs)
            if rank == 0 and n % (cfg.generate.batch_size * 10) == 0:
                log.info(f"{n}/{len(shard)} ({(time.time() - tic) / n:.2f}s/query/rank)")
    if world > 1:
        dist.barrier()
    if rank == 0:
        recs = []
        for r_ in range(world):
            p = f"{outp}.part{r_}"
            recs += read_jsonl(p)
            os.remove(p)
        recs.sort(key=lambda r: r["_order"])
        with open(outp, "w") as f:
            for r in recs:
                r.pop("_order", None)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta = {"input": inp, "ckpt": args.ckpt, "run_dir": args.run_dir, "overrides": args.set,
                "instruction_template": args.instruction_template, "precision": args.precision,
                "generate": OmegaConf.to_container(cfg.generate, resolve=True),
                "model": OmegaConf.to_container(cfg.model, resolve=True), "n": len(recs),
                "sec_total": time.time() - tic}
        json.dump(meta, open(outp + ".meta.json", "w"), indent=1)
        log.info(f"wrote {len(recs)} predictions to {outp} ({time.time() - tic:.0f}s)")
    if world > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
