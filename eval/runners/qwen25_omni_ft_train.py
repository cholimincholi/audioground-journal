"""LoRA fine-tuning of the Qwen2.5-Omni-7B *thinker* on AudioGround-IT (Reviewer 3: competing LALM adapted to our
instruction/output format). Single process, ONE GPU (or CPU for smoke tests).

  CUDA_VISIBLE_DEVICES=4 PYTHONNOUSERSITE=1 python qwen25_omni_ft_train.py \
      --output-dir runs/E8L_qwen25omni_ft/0 --seed 0
  (normally launched by experiments/train/qwen25omni_ft_run.sh)

Model / input path (identical to the zero-shot runner eval/runners/qwen25_omni.py):
  Qwen2_5OmniForConditionalGeneration.from_pretrained(..., enable_audio_output=False)  -> talker / token2wav are never
  instantiated; only `.thinker` is used. Conversation = [system <official Qwen system prompt>, user [audio, instruction]],
  prompt = processor.apply_chat_template(conv, add_generation_prompt=True), audio via qwen_omni_utils.process_mm_info
  (librosa @16 kHz, FULL length: the feature extractor accepts up to 300 s, our audio is 20-120 s, no truncation),
  processor(..., use_audio_in_video=False). Training target = tokenizer(response + "<|im_end|>") appended directly after
  the generation prompt ("<|im_start|>assistant\n"), i.e. exactly the tokens the model must emit at inference time.
  Loss: only on these response tokens (+ <|im_end|>, the analogue of our end_sym "</s>"); per-sample token-mean, then
  mean over the global batch (= our train.py / model.py normalisation).

Trainable parameters (decision):
  * LoRA r=32, alpha=64, dropout 0.1 on the LLM (thinker.model.layers.*.self_attn.{q_proj,v_proj}) only. The regex
    excludes the audio encoder's own q_proj/v_proj.  28 layers x 32 x [(3584+3584) + (3584+512)] = 10,092,544 params.
  * audio_tower.proj (Linear 1280 -> 3584 = 4,591,104 params incl. bias): the audio-to-LLM projection. In Qwen2.5-Omni it
    is the last op of the audio encoder (after avg-pool + ln_post) but it is a single small Linear that maps encoder
    features into the LLM embedding space - the counterpart of the connector/projection our model trains - so it is
    trained (peft modules_to_save, fp32 copy). Disable with --train-proj 0. Everything else (audio encoder incl. ln_post,
    vision tower, embeddings, lm_head, LLM base weights) is frozen.
  * Total trainable: 14,683,648 (14.68 M) of ~8.93 B thinker params (printed at start-up as well).

Recipe (code/audioground/configs/audioground_7b.yaml + train.py defaults, E0L_main data):
  AdamW(beta 0.9/0.999) lr 3e-5, linear warm-up from 1e-6 over 100 steps, cosine decay to 1e-5, weight decay 0.05 on
  >=2-D trainable tensors (0 on biases), grad-clip 1.0, bf16 autocast with base weights in bf16 and trainable params in
  fp32, 6000 steps x global batch 24 on ONE GPU (default micro-batch 3 x grad-accumulation 8 = our yaml), val loss on the first 1000 rows of
  the val split every 1000 steps (+ final step), keep the adapter with the lowest val loss (trainable weights only:
  <out>/best_adapter/ = peft adapter dir), non-finite grad -> skip update (abort after 20 in a row).
  Data order: same infinite sampler as ours (np.random.default_rng(seed + 1000003*epoch).permutation, world=1).
  Seeds: random/np/torch seeded with --seed (rank 0).

Deviations (all numerically neutral w.r.t. the objective unless stated):
  * Length-grouped micro-batching: each optimizer step draws the 24 samples of the global batch first, sorts them by
    token length and splits them into micro-batches (less padding). The gradient is the same mean over the same 24
    samples (each micro-batch loss = sum of its per-sample losses / 24).
  * Audio features are encoded one sample at a time (the HF encoder builds a dense block-diagonal mask over all frames of
    the batch, so batching audio is quadratic in batch size); then merged into the token embeddings exactly as in
    Qwen2_5OmniThinkerForConditionalGeneration.forward, positions from thinker.get_rope_index (TMRoPE). lm_head is only
    applied to the response positions (left padding -> responses are right-aligned), saving the 152k-vocab logits.
  * Audio encoder kept in eval mode (frozen). Sequences are left-padded as at inference.
  * --preflight (default on for cuda): before training, one forward/backward on the micro-batch of the longest
    training samples; if it OOMs, gradient checkpointing is switched on automatically (logged). Grads are discarded.

Memory (measured on CPU with saved-tensor hooks, bf16 autocast, no grad-ckpt): 8.07 MiB of saved activations per
LLM token; 120 s audio = 3000 audio tokens (+~60-120 text) -> worst micro-batch of 3 ~ 74 GiB + 18 GiB bf16 weights.

Resumable: <out>/last.pt (trainable weights + optimizer + step + best) every --save-every steps; rerunning with the same
--output-dir resumes (sample stream is replayed deterministically up to the saved step).
CUDA_DEVICE_ORDER=PCI_BUS_ID is set by code/audioground/audioground/gpu_policy.py.
"""
import argparse
import importlib.util
import json
import logging
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HOME", "ckpt/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_spec = importlib.util.spec_from_file_location(
    "ag_gpu_policy", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../code/audioground/audioground/gpu_policy.py"))
gpu_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpu_policy)   # sets CUDA_DEVICE_ORDER=PCI_BUS_ID before torch is imported

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Sampler  # noqa: E402

DEFAULT_CKPT = "ckpt/baselines/qwen25_omni"
DATA = "data/variants/lead"
LLM_LORA_REGEX = r"model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"   # fullmatch on thinker module names
PROJ_MODULE = "audio_tower.proj"
END = "<|im_end|>"


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------------------------------------ data
def build_prompt(processor, system_prompt, audio, instruction):
    conv = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio}, {"type": "text", "text": instruction}]}]
    return conv, processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


class ITDataset(Dataset):
    """One AudioGround-IT row -> prompt ids (audio placeholder expanded), response ids, cropped mel features."""

    def __init__(self, rows, processor, system_prompt, max_txt_len=300):
        self.rows, self.proc, self.sp, self.max_txt_len = rows, processor, system_prompt, max_txt_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from qwen_omni_utils import process_mm_info
        r = self.rows[i]
        conv, text = build_prompt(self.proc, self.sp, r["audio"], r["instruction"])
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        x = self.proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
                      use_audio_in_video=False)
        n_feat = int(x["feature_attention_mask"][0].sum())
        resp = self.proc.tokenizer(r["response"] + END, add_special_tokens=False, return_tensors="pt").input_ids[0]
        resp = resp[: self.max_txt_len]
        return {"idx": i, "prompt_ids": x["input_ids"][0], "resp_ids": resp,
                "feats": x["input_features"][0, :, :n_feat].clone(), "n_feat": n_feat}


def collate_list(items):
    return items


def make_micro(items, pad_id):
    """Left-pad a list of samples into one micro-batch (responses right-aligned, as at inference)."""
    seqs = [torch.cat([it["prompt_ids"], it["resp_ids"]]) for it in items]
    Lmax = max(len(s) for s in seqs)
    B = len(items)
    ids = torch.full((B, Lmax), pad_id, dtype=torch.long)
    att = torch.zeros((B, Lmax), dtype=torch.long)
    lab = torch.full((B, Lmax), -100, dtype=torch.long)
    for b, (it, s) in enumerate(zip(items, seqs)):
        ids[b, Lmax - len(s):] = s
        att[b, Lmax - len(s):] = 1
        lab[b, Lmax - len(it["resp_ids"]):] = it["resp_ids"]
    Fmax = max(it["n_feat"] for it in items)
    feats = torch.zeros((B, items[0]["feats"].shape[0], Fmax))
    fmask = torch.zeros((B, Fmax), dtype=torch.long)
    for b, it in enumerate(items):
        feats[b, :, : it["n_feat"]] = it["feats"]
        fmask[b, : it["n_feat"]] = 1
    return {"input_ids": ids, "attention_mask": att, "labels": lab, "input_features": feats,
            "feature_attention_mask": fmask, "r_max": max(len(it["resp_ids"]) for it in items)}


class InfiniteSampler(Sampler):
    """Same stream as audioground.data.InfiniteDistributedSampler for world=1, oversample factor 1."""

    def __init__(self, n, seed, skip=0):
        self.n, self.seed, self.skip = n, seed, skip

    def __iter__(self):
        epoch, drawn = 0, 0
        while True:
            perm = np.random.default_rng(self.seed + 1000003 * epoch).permutation(self.n)
            for idx in perm:
                drawn += 1
                if drawn > self.skip:
                    yield int(idx)
            epoch += 1


# ------------------------------------------------------------------------------------------------ model
def load_model(ckpt, dtype, attn):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    processor = Qwen2_5OmniProcessor.from_pretrained(ckpt)
    full = Qwen2_5OmniForConditionalGeneration.from_pretrained(ckpt, dtype=dtype, attn_implementation=attn,
                                                               enable_audio_output=False)
    full.disable_talker()
    return processor, full


def add_lora(thinker, a):
    from peft import LoraConfig, get_peft_model
    for p in thinker.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, bias="none",
                     target_modules=LLM_LORA_REGEX, modules_to_save=[PROJ_MODULE] if a.train_proj else None)
    pm = get_peft_model(thinker, cfg)   # modules are replaced in place inside `thinker`
    for n, p in pm.named_parameters():  # trainable params in fp32 (as in our train.py)
        if p.requires_grad:
            p.data = p.data.float()
    return pm


def set_grad_ckpt(thinker, on):
    if on:
        thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        thinker.gradient_checkpointing_disable()


def per_sample_loss(thinker, mb, dev, compute_dtype, autocast):
    """Mirror of Qwen2_5OmniThinkerForConditionalGeneration.forward (audio-only path) + response-only CE.
    Returns per-sample mean token loss (B,)."""
    ids = mb["input_ids"].to(dev)
    att = mb["attention_mask"].to(dev)
    lab = mb["labels"].to(dev)
    fmask = mb["feature_attention_mask"].to(dev)
    feats = mb["input_features"].to(dev, compute_dtype)
    with autocast():
        emb = thinker.get_input_embeddings()(ids)
        af = []
        for b in range(ids.shape[0]):   # one audio at a time (dense block-diag encoder mask is quadratic in batch)
            nf = int(fmask[b].sum())
            af.append(thinker.get_audio_features(feats[b:b + 1, :, :nf], feature_attention_mask=fmask[b:b + 1, :nf]))
        af = torch.cat(af).to(emb.dtype)
        amask = (ids == thinker.config.audio_token_index).unsqueeze(-1).expand_as(emb)
        assert int(amask[..., 0].sum()) == af.shape[0], "audio token count mismatch"
        emb = emb.masked_scatter(amask, af)
        pos, _ = thinker.get_rope_index(ids, None, None, att, False, fmask.sum(-1), None)
        out = thinker.model(attention_mask=att, position_ids=pos, inputs_embeds=emb, use_cache=False, return_dict=True)
        R = mb["r_max"]
        h = out.last_hidden_state[:, -R - 1:-1]          # positions predicting the last R tokens
        logits = thinker.lm_head(h)
    tgt = lab[:, -R:]
    tl = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100,
                         reduction="none").view(tgt.shape)
    m = (tgt != -100).float()
    return (tl * m).sum(1) / m.sum(1).clamp(min=1)


def lr_at(step, a):
    if step < a.warmup_steps:
        return a.warmup_start_lr + (a.lr - a.warmup_start_lr) * step / max(1, a.warmup_steps)
    p = (step - a.warmup_steps) / max(1, a.max_steps - a.warmup_steps)
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * p))


def split_micro(items, micro, pad_id):
    items = sorted(items, key=lambda it: len(it["prompt_ids"]) + len(it["resp_ids"]))
    return [make_micro(items[i:i + micro], pad_id) for i in range(0, len(items), micro)]


@torch.no_grad()
def evaluate(thinker, loader, micro, pad_id, dev, cdt, autocast, max_batches=None, rank=0, world=1):
    thinker.eval()
    tot, n = 0.0, 0
    for i, items in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        for mb in split_micro(items, micro, pad_id)[rank::world]:   # micro-batches sharded over ranks
            ps = per_sample_loss(thinker, mb, dev, cdt, autocast)
            tot += float(ps.sum()); n += ps.numel()
    if world > 1:
        t = torch.tensor([tot, float(n)], device=dev, dtype=torch.float64)
        torch.distributed.all_reduce(t); tot, n = float(t[0]), int(t[1])
    thinker.train(); thinker.audio_tower.eval()
    return tot / max(1, n)


def show_example(processor, ds, log, i=0):
    it = ds[i]
    mb = make_micro([it], processor.tokenizer.pad_token_id)
    ids, lab = mb["input_ids"][0], mb["labels"][0]
    tok = processor.tokenizer
    n_audio = int((ids == tok.convert_tokens_to_ids("<|AUDIO|>")).sum())
    txt = tok.decode(ids, skip_special_tokens=False)
    txt = txt.replace("<|AUDIO|>" * n_audio, f"<|AUDIO|> x {n_audio}")
    sup = lab[lab != -100]
    log.info("=== decoded training sample (row %d, %d tokens, %d audio tokens, %d mel frames) ===\n%s\n=== end ===",
             ds[i]["idx"], len(ids), n_audio, it["n_feat"], txt)
    log.info("supervised tokens (%d): %s", len(sup), [tok.decode([t]) for t in sup.tolist()])
    log.info("supervised text: %r", tok.decode(sup))
    log.info("last 6 unsupervised tokens: %s", [tok.decode([t]) for t in ids[lab == -100][-6:].tolist()])


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--train-jsonl", default=f"{DATA}/train.jsonl")
    ap.add_argument("--val-jsonl", default=f"{DATA}/val.jsonl")
    ap.add_argument("--max-val-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--global-batch", type=int, default=24)
    ap.add_argument("--micro-batch", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--warmup-start-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--train-proj", type=int, default=1)
    ap.add_argument("--grad-ckpt", type=int, default=0)
    ap.add_argument("--preflight", type=int, default=-1, help="-1: on for cuda, off for cpu")
    ap.add_argument("--system-prompt", default=None, help="default: official Qwen prompt (= zero-shot runner)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--max-nonfinite", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--val-max-batches", type=int, default=0, help="debug: cap validation batches")
    ap.add_argument("--dump-example", type=int, default=1, help="log one decoded sample + label mask at start")
    a = ap.parse_args()

    use_cuda = a.device.startswith("cuda")
    gpu_policy.enforce("cuda" if use_cuda else "cpu")
    if use_cuda and not torch.cuda.is_available():
        sys.exit("cuda requested but no visible GPU")
    # multi-GPU (torchrun): every rank draws the same global batch of 24, takes every world-th micro-batch, and the
    # gradients of the trainable parameters are summed over ranks before clipping -> the same update as on one GPU.
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
    dev = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if a.preflight < 0:
        a.preflight = int(use_cuda)
    os.makedirs(a.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler()] + ([logging.FileHandler(os.path.join(a.output_dir, "train.log"))] if rank == 0 else []))
    log = logging.getLogger("train")
    import qwen25_omni  # zero-shot runner: same checkpoint / system prompt
    a.system_prompt = a.system_prompt or qwen25_omni.SYSTEM_PROMPT
    assert a.global_batch % a.micro_batch == 0, "global batch must be divisible by micro-batch"
    accum = a.global_batch // a.micro_batch
    if rank == 0:
      json.dump(dict(vars(a), accum=accum, world=world, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   torch=torch.__version__), open(os.path.join(a.output_dir, "args.json"), "w"), indent=1)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    cdt = torch.bfloat16 if a.precision == "bf16" else torch.float32
    autocast = (lambda: torch.autocast(dev.type, dtype=torch.bfloat16)) if a.precision == "bf16" else \
        (lambda: torch.autocast(dev.type, enabled=False))
    processor, full = load_model(a.ckpt, cdt, a.attn)
    thinker = full.thinker
    pad_id = processor.tokenizer.pad_token_id

    train_rows = read_jsonl(a.train_jsonl)
    val_rows = read_jsonl(a.val_jsonl)[: a.max_val_samples or None] if a.val_jsonl and os.path.exists(a.val_jsonl) else []
    train_ds = ITDataset(train_rows, processor, a.system_prompt)
    val_ds = ITDataset(val_rows, processor, a.system_prompt) if val_rows else None
    log.info(f"train rows {len(train_ds)}  val rows {len(val_rows)}  micro {a.micro_batch}  accum {accum}  global "
             f"{a.global_batch}  steps {a.max_steps}  (~{a.max_steps * a.global_batch / len(train_ds):.2f} epochs)")
    if a.dump_example and rank == 0:
        show_example(processor, train_ds, log, 0)

    pm = add_lora(thinker, a)
    thinker.to(dev)
    set_grad_ckpt(thinker, a.grad_ckpt)
    thinker.train(); thinker.audio_tower.eval()
    named = [(n, p) for n, p in pm.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in thinker.parameters()) - (sum(p.numel() for n, p in named if "modules_to_save" in n))
    n_lora = sum(p.numel() for n, p in named if "lora_" in n)
    log.info(f"trainable params {n_train:,} ({n_train / 1e6:.2f}M: LoRA {n_lora:,}, proj {n_train - n_lora:,}) / "
             f"{n_all / 1e6:.1f}M thinker params; dtype base {cdt}, trainable {params[0].dtype}")
    opt = torch.optim.AdamW([{"params": [p for n, p in named if p.ndim >= 2], "weight_decay": a.weight_decay},
                             {"params": [p for n, p in named if p.ndim < 2], "weight_decay": 0.0}],
                            lr=a.lr, betas=(a.beta1, a.beta2))

    step, best = 0, float("inf")
    last = os.path.join(a.output_dir, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)   # own file (contains numpy RNG state)
        sd = dict(named)
        for n, t in ck["trainable"].items():
            sd[n].data.copy_(t)
        opt.load_state_dict(ck["optimizer"])
        step, best = ck["step"], ck["best_val"]
        if ck.get("grad_ckpt"):
            a.grad_ckpt = 1; set_grad_ckpt(thinker, True)
        torch.set_rng_state(ck["torch_rng"]); np.random.set_state(ck["np_rng"]); random.setstate(ck["py_rng"])
        log.info(f"resumed from {last} at step {step} (best val {best:.4f})")

    def save_last():
        ck = {"trainable": {n: p.detach().cpu() for n, p in named}, "optimizer": opt.state_dict(), "step": step,
              "best_val": best, "grad_ckpt": a.grad_ckpt, "torch_rng": torch.get_rng_state(),
              "np_rng": np.random.get_state(), "py_rng": random.getstate()}
        torch.save(ck, last + ".tmp"); os.replace(last + ".tmp", last)

    def save_best(vl):
        d = os.path.join(a.output_dir, "best_adapter")
        tmp = d + ".tmp"
        pm.save_pretrained(tmp)
        json.dump({"step": step, "val_loss": vl}, open(os.path.join(tmp, "best_info.json"), "w"))
        if os.path.exists(d):
            import shutil; shutil.rmtree(d)
        os.replace(tmp, d)

    # ---- pre-flight: worst-case micro-batch (longest training samples) fwd/bwd; OOM -> gradient checkpointing
    if a.preflight and step == 0:
        durs = np.array([float(r.get("meta", {}).get("duration") or 0) for r in train_rows])
        lng = [train_ds[int(i)] for i in np.argsort(-durs)[: a.micro_batch]]
        import gc
        for attempt in range(2):
            oom = False
            try:
                torch.cuda.reset_peak_memory_stats(dev)
                t0 = time.time()
                (per_sample_loss(thinker, make_micro(lng, pad_id), dev, cdt, autocast).sum() / a.global_batch).backward()
                torch.cuda.synchronize(dev)
                log.info(f"preflight OK: micro {a.micro_batch} x {max(it['n_feat'] for it in lng) / 100:.0f}s audio, "
                         f"grad_ckpt={a.grad_ckpt}, peak mem {torch.cuda.max_memory_allocated(dev) / 2**30:.1f}GB, "
                         f"{time.time() - t0:.2f}s")
            except torch.cuda.OutOfMemoryError:
                oom = True          # handled outside the except block so the traceback frames are released first
            if not oom:
                break
            opt.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
            if a.grad_ckpt:
                sys.exit(f"preflight OOM even with gradient checkpointing at micro-batch {a.micro_batch}; lower MICRO")
            log.warning("preflight OOM without gradient checkpointing -> enabling it")
            a.grad_ckpt = 1; set_grad_ckpt(thinker, True)
        opt.zero_grad(set_to_none=True)
        if world > 1:   # all ranks must agree on gradient checkpointing
            f = torch.tensor([float(a.grad_ckpt)], device=dev); torch.distributed.all_reduce(f, op=torch.distributed.ReduceOp.MAX)
            if f.item() > 0 and not a.grad_ckpt:
                a.grad_ckpt = 1; set_grad_ckpt(thinker, True)
        if use_cuda:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)

    sampler = InfiniteSampler(len(train_ds), a.seed, skip=step * a.global_batch)
    nw = max(2, a.num_workers // world)
    train_loader = DataLoader(train_ds, batch_size=a.global_batch, sampler=sampler, num_workers=nw,
                              collate_fn=collate_list, drop_last=True, persistent_workers=nw > 0,
                              prefetch_factor=2 if nw > 0 else None,
                              worker_init_fn=lambda _: torch.set_num_threads(1))
    val_loader = DataLoader(val_ds, batch_size=a.global_batch, shuffle=False, num_workers=min(8, nw),
                            collate_fn=collate_list, worker_init_fn=lambda _: torch.set_num_threads(1)) if val_ds else None

    it = iter(train_loader)
    tic = time.time(); hist = []; step0 = step; nonfinite = 0
    metrics = open(os.path.join(a.output_dir, "metrics.jsonl"), "a") if rank == 0 else open(os.devnull, "w")
    while step < a.max_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        opt.zero_grad(set_to_none=True)
        items = next(it)
        loss_sum, ntok = 0.0, 0
        for mb in split_micro(items, a.micro_batch, pad_id)[rank::world]:
            ps = per_sample_loss(thinker, mb, dev, cdt, autocast)
            (ps.sum() / a.global_batch).backward()
            loss_sum += float(ps.detach().sum()); ntok += int(mb["attention_mask"].numel())
        if world > 1:
            for p_ in params:
                if p_.grad is None:
                    p_.grad = torch.zeros_like(p_)
                torch.distributed.all_reduce(p_.grad)
            t = torch.tensor([loss_sum, float(ntok)], device=dev, dtype=torch.float64)
            torch.distributed.all_reduce(t); loss_sum, ntok = float(t[0]), int(t[1])
        gn = torch.nn.utils.clip_grad_norm_(params, a.grad_clip if a.grad_clip > 0 else float("inf"))
        if not torch.isfinite(gn):
            nonfinite += 1
            log.warning(f"step {step + 1}: non-finite grad norm, update skipped ({nonfinite} consecutive)")
            if nonfinite >= a.max_nonfinite:
                log.error("too many consecutive non-finite steps; aborting run"); sys.exit(3)
        else:
            opt.step(); nonfinite = 0
        step += 1
        loss = loss_sum / len(items)
        hist.append(loss)
        if step % a.log_every == 0 or step == 1:
            el = time.time() - tic; done = max(1, step - step0)
            mem = torch.cuda.max_memory_allocated(dev) / 2**30 if use_cuda else 0.0
            log.info(f"step {step}/{a.max_steps} loss {np.mean(hist[-a.log_every:]):.4f} lm {loss:.4f} "
                     f"lr {opt.param_groups[0]['lr']:.2e} gnorm {float(gn):.3f} padded_tok {ntok} "
                     f"{el / done:.2f}s/step mem {mem:.1f}GB eta {(a.max_steps - step) * el / done / 3600:.2f}h")
            metrics.write(json.dumps({"step": step, "loss": loss, "lm_loss": loss, "lr": opt.param_groups[0]["lr"],
                                      "gnorm": float(gn), "sec_per_step": el / done, "mem_gb": mem}) + "\n")
            metrics.flush()
        if val_loader is not None and (step % a.eval_every == 0 or step == a.max_steps):
            t0 = time.time()
            vl = evaluate(thinker, val_loader, a.micro_batch, pad_id, dev, cdt, autocast, a.val_max_batches or None, rank, world)
            if not math.isfinite(vl):
                log.error(f"step {step}: non-finite val loss ({vl}); aborting run"); sys.exit(3)
            log.info(f"step {step} val_loss {vl:.4f} (best {best:.4f}) [{time.time() - t0:.0f}s]")
            metrics.write(json.dumps({"step": step, "val_loss": vl}) + "\n"); metrics.flush()
            if vl < best:
                best = vl
                if rank == 0:
                    save_best(vl)
        if (step % a.save_every == 0 or step == a.max_steps) and rank == 0:
            save_last()
        if world > 1 and (step % a.save_every == 0 or step % a.eval_every == 0 or step == a.max_steps):
            torch.distributed.barrier()
    if val_loader is None and rank == 0:
        save_best(float("nan"))
    if rank == 0:
        open(os.path.join(a.output_dir, "DONE"), "w").write(f"{step}\n")
    log.info(f"done: {step} steps, best val {best:.4f}, total {(time.time() - tic) / 3600:.2f}h")


if __name__ == "__main__":
    main()
