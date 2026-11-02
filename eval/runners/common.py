"""Shared driver for baseline LALM runners (model-agnostic, no heavy deps).

Each model adapter module (runners/<model>.py) must define
    class Model:
        def __init__(self, ckpt_dir: str, device: str, **kw): ...
        def generate(self, audio_path: str, prompt: str, gen: dict) -> str: ...
and call `common.main(Model, default_name="<model>")` under __main__.

Output JSONL (one line per query, resumable / append-only):
    {qid, bench, setting, model, prompt_id, prompt, response, L, gen, time_s, error}
Default output: runs/baselines/<model>/<bench>_<setting>.jsonl
Sharding: --shard i --nshards k writes <bench>_<setting>.shard{i}of{k}.jsonl; merge with
runners/merge_shards.py (make_tables.py also reads shard files directly).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

BENCH_ROOT = "eval/benchmarks"
RUNS_ROOT = "runs/baselines"

# Paper: deterministic beam search, 4 beams, no sampling, up to 100 new tokens.
GEN_DEFAULT = dict(num_beams=4, do_sample=False, max_new_tokens=100)

# The paper does not print the grounding prompt. "v1" is the shared prompt used for ALL
# baselines (same prompt + same parser for every model). Change only via --prompt-id.
PROMPTS = {
    "v1": ("Given the audio, find the time interval in which the following sound event occurs: "
           "\"{query}\". The audio is {L:.1f} seconds long. Answer with the start and end time "
           "in seconds, in the format: from <start> to <end> seconds."),
    "v1_nodur": ("Given the audio, find the time interval in which the following sound event occurs: "
                 "\"{query}\". Answer with the start and end time in seconds, in the format: "
                 "from <start> to <end> seconds."),
    # AudioGround-IT style grounding question (all timestamps in seconds, 1 decimal)
    "agit": ("When does the sound described as \"{query}\" occur in the audio? "
             "Give the start and end timestamps in seconds."),
}


def build_prompt(prompt_id: str, rec: dict) -> str:
    q = rec["query"].strip()
    return PROMPTS[prompt_id].format(query=q, L=rec["L"])


def load_bench(bench: str, setting: str):
    with open(os.path.join(BENCH_ROOT, bench, f"{setting}.jsonl")) as f:
        return [json.loads(l) for l in f]


def parse_args(default_name):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default=default_name)
    ap.add_argument("--ckpt", default=None, help="checkpoint dir (adapter-specific default)")
    ap.add_argument("--bench", default="clotho_moment", choices=["clotho_moment", "unav100", "tut2017", "tacos"])
    ap.add_argument("--setting", default="original", choices=["original", "clipped"])
    ap.add_argument("--prompt-id", default="v1")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only first N queries (smoke test)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-beams", type=int, default=GEN_DEFAULT["num_beams"])
    ap.add_argument("--max-new-tokens", type=int, default=GEN_DEFAULT["max_new_tokens"])
    ap.add_argument("--extra", default="{}", help="JSON dict of adapter kwargs")
    return ap.parse_args()


def out_path(args):
    if args.out:
        return args.out
    d = os.path.join(RUNS_ROOT, args.model_name)
    os.makedirs(d, exist_ok=True)
    suf = "" if args.nshards == 1 else f".shard{args.shard}of{args.nshards}"
    pid = "" if args.prompt_id == "v1" else f".{args.prompt_id}"
    lim = "" if args.limit is None else f".limit{args.limit}"
    return os.path.join(d, f"{args.bench}_{args.setting}{pid}{suf}{lim}.jsonl")


def main(ModelCls, default_name):
    args = parse_args(default_name)
    recs = load_bench(args.bench, args.setting)
    if args.limit:
        recs = recs[: args.limit]
    recs = [r for i, r in enumerate(recs) if i % args.nshards == args.shard]
    op = out_path(args)
    done = set()
    if os.path.exists(op):
        with open(op) as f:
            for l in f:
                try:
                    j = json.loads(l)
                    if j.get("error") is None:
                        done.add(str(j["qid"]))
                except Exception:
                    pass
    todo = [r for r in recs if str(r["qid"]) not in done]
    print(f"[{args.model_name}] {args.bench}/{args.setting}: {len(recs)} queries, {len(todo)} to do -> {op}", flush=True)
    if not todo:
        return
    gen = dict(GEN_DEFAULT, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
    model = ModelCls(args.ckpt, args.device, **json.loads(args.extra))
    with open(op, "a") as f:
        for k, r in enumerate(todo):
            prompt = build_prompt(args.prompt_id, r)
            t0 = time.time()
            err, resp = None, None
            try:
                resp = model.generate(r["audio"], prompt, gen)
            except Exception as e:  # keep going; failures are retried on resume
                err = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            f.write(json.dumps(dict(qid=str(r["qid"]), bench=args.bench, setting=args.setting,
                                    model=args.model_name, prompt_id=args.prompt_id, prompt=prompt,
                                    response=resp, L=r["L"], gen=gen, time_s=round(time.time() - t0, 3),
                                    error=err)) + "\n")
            f.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(todo)} {time.time()-t0:.2f}s :: {str(resp)[:120]!r}", flush=True)
