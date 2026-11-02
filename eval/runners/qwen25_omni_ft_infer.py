"""Inference for the LoRA-fine-tuned Qwen2.5-Omni-7B thinker (E8L_qwen25omni_ft).

  CUDA_VISIBLE_DEVICES=4 PYTHONNOUSERSITE=1 python qwen25_omni_ft_infer.py \
      --adapter runs/E8L_qwen25omni_ft/0/best_adapter \
      --out-dir runs/E8L_qwen25omni_ft/0/preds/default --settings original

Code path = the zero-shot runner (eval/runners/qwen25_omni.py): its Model class loads the base checkpoint
(enable_audio_output=False, disable_talker(), sdpa, here dtype bf16), the trained peft adapter (LoRA on the LLM q/v +
the trained audio_tower.proj copy) is then injected in place into model.thinker (not merged), and generation goes through
Model.generate unchanged: same official system prompt, chat template, process_mm_info audio loading (full-length audio),
processor call and model.generate(thinker_num_beams=4, thinker_do_sample=False, thinker_max_new_tokens=100,
return_audio=False). Batch size is FIXED at 1 (default, = zero-shot runner; no padding effects, reproducible).
--batch-size N>1 uses a batched copy of the same steps (left padding) for speed; results may differ marginally
from bs=1 in bf16, so keep the batch size fixed across compared runs (it is recorded in <out>/infer_meta.json).

Instruction: the training-style template used for our AudioGround model (code/audioground/infer.py DEFAULT_TEMPLATE):
  'Identify when "{query}" is heard in the recording.'
Output: <out-dir>/<bench>_<setting>.jsonl, one JSON per benchmark row (all benchmark fields kept) + prompt, raw_output,
time_s; written to .partial (resumable by qid) and renamed when complete. <out-dir>/DONE after all files are written.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HOME", "ckpt/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
_spec = importlib.util.spec_from_file_location(
    "ag_gpu_policy", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../code/audioground/audioground/gpu_policy.py"))
gpu_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpu_policy)   # CUDA_DEVICE_ORDER=PCI_BUS_ID before torch

import common  # noqa: E402  (GEN_DEFAULT, load_bench; not modified)
import qwen25_omni  # noqa: E402

TEMPLATE = 'Identify when "{query}" is heard in the recording.'
BENCHES = ["clotho_moment", "unav100", "tut2017"]


class FTModel(qwen25_omni.Model):
    def __init__(self, ckpt_dir, device, adapter, dtype="bfloat16", **kw):
        super().__init__(ckpt_dir, device, dtype=dtype, **kw)
        torch = self.torch
        if adapter:
            from peft import PeftModel
            PeftModel.from_pretrained(self.model.thinker, adapter, is_trainable=False)  # in-place injection
            self.model.to(device=device, dtype=getattr(torch, dtype))
        self.model.eval()
        n_lora = sum(p.numel() for n, p in self.model.named_parameters() if "lora_" in n)
        n_mts = sum(p.numel() for n, p in self.model.named_parameters() if "modules_to_save" in n)
        print(f"adapter {adapter or '(none)'}: LoRA params {n_lora:,}, modules_to_save params {n_mts:,}", flush=True)

    def generate_batch(self, audio_paths, prompts, gen):
        """Batched copy of qwen25_omni.Model.generate (same steps; left padding)."""
        if len(audio_paths) == 1:
            return [self.generate(audio_paths[0], prompts[0], gen)]
        from qwen_omni_utils import process_mm_info
        convs = []
        for ap, p in zip(audio_paths, prompts):
            item = {"type": "audio", "audio": ap}
            if self.max_audio_s is not None:
                item.update(audio_start=0.0, audio_end=float(self.max_audio_s))
            convs.append([{"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                          {"role": "user", "content": [item, {"type": "text", "text": p}]}])
        texts = [self.processor.apply_chat_template(c, add_generation_prompt=True, tokenize=False) for c in convs]
        audios, images, videos = process_mm_info(convs, use_audio_in_video=False)
        inputs = self.processor(text=texts, audio=audios, images=images, videos=videos, return_tensors="pt",
                                padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.device)
        inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs, use_audio_in_video=False, return_audio=False,
                thinker_num_beams=gen["num_beams"], thinker_do_sample=gen["do_sample"],
                thinker_max_new_tokens=gen["max_new_tokens"],
            )
        out = out[:, inputs["input_ids"].size(1):]
        return [s.strip() for s in self.processor.batch_decode(out, skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="peft adapter dir (best_adapter); '' = base model")
    ap.add_argument("--ckpt", default=qwen25_omni.DEFAULT_CKPT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--benches", default=",".join(BENCHES))
    ap.add_argument("--settings", default="original", help="comma list of original,clipped")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N queries per file")
    ap.add_argument("--extra", default="{}", help="JSON kwargs for qwen25_omni.Model (system_prompt, max_audio_s, ...)")
    a = ap.parse_args()
    gpu_policy.enforce("cuda" if a.device.startswith("cuda") else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)
    done_flag = os.path.join(a.out_dir, "DONE")
    if os.path.exists(done_flag):
        os.remove(done_flag)
    gen = dict(common.GEN_DEFAULT)
    jobs = [(b, s) for s in a.settings.split(",") if s for b in a.benches.split(",") if b]
    model = None
    t_all = time.time()
    for bench, setting in jobs:
        recs = common.load_bench(bench, setting)
        if a.limit:
            recs = recs[: a.limit]
        final = os.path.join(a.out_dir, f"{bench}_{setting}.jsonl")
        if os.path.exists(final):
            print(f"{final} exists, skip", flush=True)
            continue
        part = final + ".partial"
        done = set()
        if os.path.exists(part):
            for l in open(part):
                try:
                    j = json.loads(l)
                    if j.get("error") is None:
                        done.add(str(j["qid"]))
                except Exception:
                    pass
        todo = [r for r in recs if str(r["qid"]) not in done]
        print(f"{bench}/{setting}: {len(recs)} queries, {len(todo)} to do, bs {a.batch_size} -> {final}", flush=True)
        if todo and model is None:
            model = FTModel(a.ckpt, a.device, a.adapter, dtype=a.dtype, **json.loads(a.extra))
        tic = time.time()
        with open(part, "a") as f:
            for k in range(0, len(todo), a.batch_size):
                chunk = todo[k:k + a.batch_size]
                prompts = [a.template.format(query=r["query"].strip(), L=float(r["L"])) for r in chunk]
                t0 = time.time()
                try:
                    outs, err = model.generate_batch([r["audio"] for r in chunk], prompts, gen), None
                except Exception as e:
                    import traceback; traceback.print_exc()
                    outs, err = [None] * len(chunk), f"{type(e).__name__}: {e}"
                dt = (time.time() - t0) / len(chunk)
                for r, p, o in zip(chunk, prompts, outs):
                    rec = dict(r); rec.update(prompt=p, raw_output=o, model="qwen25_omni_ft", gen=gen,
                                              batch_size=a.batch_size, time_s=round(dt, 3), error=err)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if (k // a.batch_size) % max(1, 50 // a.batch_size) == 0:
                    print(f"  {k + len(chunk)}/{len(todo)} {dt:.2f}s/q :: {str(outs[0])[:100]!r}", flush=True)
        # finalize: benchmark order, one row per qid, no errors allowed
        got = {}
        for l in open(part):
            j = json.loads(l)
            if j.get("error") is None:
                got[str(j["qid"])] = j
        missing = [r["qid"] for r in recs if str(r["qid"]) not in got]
        if missing:
            sys.exit(f"{bench}/{setting}: {len(missing)} queries failed (e.g. {missing[:5]}); rerun to retry")
        with open(final + ".tmp", "w") as f:
            for r in recs:
                f.write(json.dumps(got[str(r["qid"])], ensure_ascii=False) + "\n")
        os.replace(final + ".tmp", final)
        os.remove(part)
        print(f"wrote {len(recs)} -> {final} ({time.time() - tic:.0f}s)", flush=True)
    json.dump({"adapter": a.adapter, "ckpt": a.ckpt, "template": a.template, "gen": gen, "dtype": a.dtype,
               "batch_size": a.batch_size, "jobs": jobs, "limit": a.limit, "sec_total": time.time() - t_all},
              open(os.path.join(a.out_dir, "infer_meta.json"), "w"), indent=1)
    open(done_flag, "w").write("ok\n")


if __name__ == "__main__":
    main()
