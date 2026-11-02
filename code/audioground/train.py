"""AudioGround training (single process or torchrun DDP).

  torchrun --nproc_per_node 8 train.py --cfg configs/audioground_7b.yaml --set train.seed=0 train.output_dir=...
"""
import argparse
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import logging
import math
import os
import random
import sys
import time

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audioground import gpu_policy  # noqa: E402  (sets CUDA_DEVICE_ORDER before torch)
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audioground.config import load_config
from audioground.data import AudioGroundDataset, InfiniteDistributedSampler, collate, read_jsonl
from audioground.model import AudioGround

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def setup_dist(use_cuda):
    if "RANK" in os.environ:
        dist.init_process_group("nccl" if use_cuda else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        return rank, world, local
    return 0, 1, 0


def lr_at(step, t):
    if step < t.warmup_steps:
        return t.warmup_start_lr + (t.lr - t.warmup_start_lr) * step / max(1, t.warmup_steps)
    if t.lr_schedule == "constant":
        return t.lr
    p = (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps)
    if t.lr_schedule == "linear":
        return t.min_lr + (t.lr - t.min_lr) * (1 - p)
    return t.min_lr + 0.5 * (t.lr - t.min_lr) * (1 + math.cos(math.pi * p))


def _worker_init(_):
    torch.set_num_threads(1)  # dataloader workers: 1 thread each (avoid CPU oversubscription on a shared box)


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, dev, max_batches=None):
    model.eval()
    tot, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        out = model(to_dev(batch, dev))
        bs = len(batch["id"])
        tot += out["lm_loss"].item() * bs; n += bs
    t = torch.tensor([tot, n], device=dev, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(t)
    model.train()
    return (t[0] / t[1].clamp(min=1)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", action="append", default=[])
    ap.add_argument("--set", nargs="*", default=[], help="dotlist overrides, e.g. model.ate=sin train.seed=1")
    args = ap.parse_args()
    cfg = load_config(args.cfg, args.set)
    t, m, d = cfg.train, cfg.model, cfg.data
    gpu_policy.enforce(t.device)

    use_cuda = t.device == "cuda" and torch.cuda.is_available()
    rank, world, local = setup_dist(use_cuda)
    dev = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(dev)
    dtype = DTYPES[t.precision] if use_cuda else torch.float32
    os.makedirs(t.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler()] + ([logging.FileHandler(os.path.join(t.output_dir, "train.log"))] if rank == 0 else []))
    log = logging.getLogger("train")
    if rank == 0:
        OmegaConf.save(cfg, os.path.join(t.output_dir, "config.yaml"), resolve=True)
    seed = int(t.seed)
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)

    # ---- data
    task_ok = (lambda r: r.get("task") in set(d.tasks)) if d.tasks else (lambda r: True)
    train_rows = [r for r in read_jsonl(d.train_jsonl) if task_ok(r)]
    train_ds = AudioGroundDataset(train_rows, m, d, m.whisper_path, train=True)
    assert t.global_batch_size % (t.micro_batch_size * world) == 0, "global batch must be divisible by micro*world"
    accum = t.global_batch_size // (t.micro_batch_size * world)
    max_steps = int(t.max_steps)
    if t.num_epochs:
        max_steps = int(math.ceil(len(train_ds) * float(t.num_epochs) / t.global_batch_size))
        t.max_steps = max_steps
    sampler = InfiniteDistributedSampler(train_ds, rank, world, seed, t.micro_batch_size, d.short_oversample_factor,
                                         d.short_threshold_sec, d.short_oversample_schedule, max_steps, accum,
                                         d.get("oversample_target", "short"))
    train_loader = DataLoader(train_ds, batch_size=t.micro_batch_size, sampler=sampler, num_workers=d.num_workers,
                              collate_fn=collate, pin_memory=use_cuda, drop_last=True, worker_init_fn=_worker_init,
                              persistent_workers=d.num_workers > 0, prefetch_factor=4 if d.num_workers > 0 else None)
    val_loader = None
    if d.val_jsonl and os.path.exists(d.val_jsonl):
        vrows = [r for r in read_jsonl(d.val_jsonl) if task_ok(r)][: d.max_val_samples or None]
        vrows = vrows[rank::world]
        val_ds = AudioGroundDataset(vrows, m, d, m.whisper_path, train=True)
        val_loader = DataLoader(val_ds, batch_size=t.micro_batch_size, shuffle=False, num_workers=min(4, d.num_workers),
                                collate_fn=collate, worker_init_fn=_worker_init)
    log.info(f"train rows {len(train_ds)}  world {world}  micro {t.micro_batch_size}  accum {accum}  "
             f"global {t.global_batch_size}  steps {max_steps}  (~{max_steps * t.global_batch_size / len(train_ds):.2f} epochs)")

    # ---- model
    model = AudioGround(m, compute_dtype=dtype)
    if t.init_from:
        sd = torch.load(t.init_from, map_location="cpu")["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.info(f"init_from {t.init_from}: {len(sd)} tensors, unexpected={len(unexpected)}")
    model.to(dev)
    for n_, p in model.named_parameters():        # trainable params in fp32
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params); n_all = sum(p.numel() for p in model.parameters())
    log.info(f"trainable params {n_train/1e6:.2f}M / {n_all/1e6:.1f}M")
    decay = [p for n_, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for n_, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": t.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=t.lr, betas=(t.beta1, t.beta2))
    scaler = torch.cuda.amp.GradScaler(enabled=(use_cuda and t.precision == "fp16"))
    step = 0
    best = float("inf")
    if t.resume and os.path.exists(t.resume):
        ck = torch.load(t.resume, map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        step = ck["step"]; best = ck.get("best_val", best)
        sampler.epoch = ck.get("sampler_epoch", 0)
        log.info(f"resumed from {t.resume} at step {step}")
    raw_model = model
    if world > 1:
        model = DDP(model, device_ids=[local] if use_cuda else None, find_unused_parameters=False)

    def save(name, val=None, with_opt=False):
        if rank != 0:
            return
        ck = {"model": raw_model.trainable_state_dict(), "optimizer": opt.state_dict() if with_opt else None, "step": step,
              "best_val": best, "val_loss": val, "sampler_epoch": sampler.epoch,
              "config": OmegaConf.to_container(cfg, resolve=True)}
        torch.save(ck, os.path.join(t.output_dir, name + ".tmp"))
        os.replace(os.path.join(t.output_dir, name + ".tmp"), os.path.join(t.output_dir, name))

    # ---- loop
    model.train()
    it = iter(train_loader)
    tic = time.time(); hist = []; step0 = step; nonfinite = 0
    while step < max_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, t)
        opt.zero_grad(set_to_none=True)
        agg = {"loss": 0.0, "lm_loss": 0.0, "span_loss": 0.0}
        for a in range(accum):
            batch = to_dev(next(it), dev)
            ctx = model.no_sync() if (world > 1 and a < accum - 1) else torch.enable_grad()
            with ctx:
                out = model(batch)
                scaler.scale(out["loss"] / accum).backward()
            for k in agg:
                if k in out:
                    agg[k] += float(out[k]) / accum
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, t.grad_clip if t.grad_clip > 0 else float("inf"))
        if not torch.isfinite(gn) and not scaler.is_enabled():
            # non-finite gradients (identical on all ranks after all-reduce): skip the update instead of
            # poisoning every parameter; abort after max_nonfinite consecutive bad steps (no DONE -> queue sees FAILED)
            nonfinite += 1
            log.warning(f"step {step + 1}: non-finite grad norm, update skipped ({nonfinite} consecutive)")
            if nonfinite >= t.get("max_nonfinite", 20):
                log.error("too many consecutive non-finite steps; aborting run")
                sys.exit(3)
        else:
            scaler.step(opt); scaler.update()
            nonfinite = 0
        step += 1
        hist.append(agg["loss"])
        if step % t.log_every == 0 or step == 1:
            el = time.time() - tic
            done = max(1, step - step0)
            beta = float(raw_model.ate.beta) if hasattr(raw_model.ate, "beta") else float("nan")
            mem = torch.cuda.max_memory_allocated(dev) / 2**30 if use_cuda else 0.0
            log.info(f"step {step}/{max_steps} loss {np.mean(hist[-t.log_every:]):.4f} lm {agg['lm_loss']:.4f} "
                     f"span {agg['span_loss']:.4f} lr {opt.param_groups[0]['lr']:.2e} gnorm {float(gn):.3f} beta {beta:.4f} "
                     f"{el / done:.2f}s/step mem {mem:.1f}GB eta {(max_steps - step) * el / done / 3600:.2f}h")
            if rank == 0:
                with open(os.path.join(t.output_dir, "metrics.jsonl"), "a") as f:
                    f.write(json.dumps({"step": step, "loss": agg["loss"], "lm_loss": agg["lm_loss"],
                                        "span_loss": agg["span_loss"], "lr": opt.param_groups[0]["lr"],
                                        "beta": beta, "sec_per_step": el / done, "mem_gb": mem}) + "\n")
        if val_loader is not None and (step % t.eval_every == 0 or step == max_steps):
            vl = evaluate(raw_model, val_loader, dev)
            if not math.isfinite(vl):
                log.error(f"step {step}: non-finite val loss ({vl}); aborting run")
                sys.exit(3)
            log.info(f"step {step} val_loss {vl:.4f} (best {best:.4f})")
            if rank == 0:
                with open(os.path.join(t.output_dir, "metrics.jsonl"), "a") as f:
                    f.write(json.dumps({"step": step, "val_loss": vl}) + "\n")
            if vl < best:
                best = vl
                save("best.pt", vl)
        if step % t.save_every == 0 or step == max_steps:
            save("last.pt", with_opt=True)
        if t.get("keep_ckpt_every", 0) and step % t.keep_ckpt_every == 0:   # training-dynamics analysis
            save(f"step{step:05d}.pt")
    if rank == 0:
        if val_loader is None:
            save("best.pt")  # no validation set: best == last
        open(os.path.join(t.output_dir, "DONE"), "w").write(f"{step}\n")
    if dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
