"""Probe 2: answer-onset prior via teacher forcing.

Model built exactly like code/audioground/infer.py (<run_dir>/config.yaml + best.pt, bf16, live encoders).
Input = audio + instruction (as in generation), target = training response template
"The sound occurs from 0.0 ..." built by AudioGround.build_inputs (identical to the training collate: response +
</s>, tokenised without special tokens, appended right after "ASSISTANT:"). We read
  p0   = P(first onset digit = "0" | prefix "The sound occurs from")
  pdot = P("." | ... "0")      p00 = P("0" | ... "0.")
  P(onset<1s) = p0*pdot, P(onset==0.0) = p0*pdot*p00  (LLaMA tokenises numbers digit by digit).
Output: <run_dir>/preds/probe_prior/{clotho600,val325}.jsonl
  CUDA_VISIBLE_DEVICES=4 python onset_prior.py --run_dir runs/E0_main/1
"""
import argparse
import json
import os
import sys

CODE = "code/audioground"
sys.path.insert(0, CODE)
from audioground import gpu_policy  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from audioground import llama_sdpa  # noqa: E402
from audioground.config import load_config  # noqa: E402
from audioground.data import AudioGroundDataset, collate, read_jsonl  # noqa: E402
from audioground.model import AudioGround  # noqa: E402

TEMPLATE = 'Identify when "{query}" is heard in the recording.'
PREFIX = "The sound occurs from"


def load_sets():
    cl = read_jsonl("eval/benchmarks/clotho_moment/original.jsonl")
    idx = np.sort(np.random.default_rng(0).choice(len(cl), 600, replace=False))
    clotho = [dict(id=str(cl[i]["qid"]), audio=cl[i]["audio"], instruction=TEMPLATE.format(query=cl[i]["query"].strip()),
                   gt=cl[i]["gt"], L=cl[i]["L"]) for i in idx]
    va = [r for r in read_jsonl("data/variants/default/val.jsonl") if r["task"] == "grounding"]
    val = [dict(id=r["meta"]["id"], audio=r["audio"], instruction=r["instruction"], gt=r["gt_intervals"],
                L=r["meta"]["duration"]) for r in va]
    return {"clotho600": clotho, "val325": val}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config([os.path.join(args.run_dir, "config.yaml")], ["data.feature_cache=null"])
    gpu_policy.enforce(cfg.train.device)
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    dtype = torch.bfloat16
    model = AudioGround(cfg.model, compute_dtype=dtype)
    sd = torch.load(os.path.join(args.run_dir, "best.pt"), map_location="cpu")["model"]
    _, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected
    model.to(dev).eval(); model.to(dtype)
    if model.use_beats:
        model.beats.float()
    tok = model.llama_tokenizer

    # --- target tokenisation exactly as training (build_inputs: _tok(response + end_sym))
    a = model._tok("The sound occurs from 0.0 to 5.0 seconds." + cfg.model.end_sym).tolist()
    b = model._tok("The sound occurs from 23.4 to 35.0 seconds." + cfg.model.end_sym).tolist()
    n = 0
    while a[n] == b[n]:
        n += 1
    pre = a[:n]
    pieces = tok.convert_ids_to_tokens(a)
    assert pieces[n:n + 3] == ["0", ".", "0"], pieces
    assert tok.convert_ids_to_tokens(b)[n:n + 4] == ["2", "3", ".", "4"]
    digit_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    id0, iddot = digit_ids[0], tok.convert_tokens_to_ids(".")
    print("prefix tokens:", tok.convert_ids_to_tokens(pre), "| next:", pieces[n:n + 3], flush=True)
    resp = "The sound occurs from 0.0 to 5.0 seconds."   # teacher-forced; only positions n..n+2 are read

    out_dir = os.path.join(args.run_dir, "preds", "probe_prior"); os.makedirs(out_dir, exist_ok=True)
    meta = dict(prefix_tokens=tok.convert_ids_to_tokens(pre), run_dir=args.run_dir, precision="bf16")
    for name, rows in load_sets().items():
        ds = AudioGroundDataset(rows, cfg.model, cfg.data, cfg.model.whisper_path, train=False)
        dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=4, collate_fn=collate)
        recs = []
        with torch.no_grad():
            for batch in dl:
                batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
                at, _ = model.encode_audio(batch)
                emb, att, lab, pend = model.build_inputs(at, batch["instruction"], [resp] * len(at))
                with model.autocast(), llama_sdpa.causal_right_padded(cfg.model.get("llm_sdpa", True)):
                    logits = model.llama_model(inputs_embeds=emb, attention_mask=att, return_dict=True).logits
                for j in range(len(at)):
                    p = int(pend[j])          # logits[p] predicts the 1st response token
                    tgt = lab[j, p + 1:p + 1 + n + 3].tolist()
                    assert tgt == a[:n + 3], (tgt, a[:n + 3])
                    lp = torch.log_softmax(logits[j, p + n:p + n + 3].float(), -1)
                    pd_ = lp[0, digit_ids].exp().tolist()
                    p0, pdot, p00 = float(lp[0, id0].exp()), float(lp[1, iddot].exp()), float(lp[2, id0].exp())
                    r = rows[batch["row_index"][j]]
                    recs.append(dict(id=r["id"], gt_onset=r["gt"][0][0], L=r["L"], p0=p0, pdot=pdot, p00=p00,
                                     p_lt1=p0 * pdot, p_eq0=p0 * pdot * p00, p_digit=pd_,
                                     top_first=tok.convert_ids_to_tokens(int(lp[0].argmax()))))
        with open(os.path.join(out_dir, f"{name}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(name, len(recs), "mean P(<1s)=%.3f P(==0)=%.3f" % (np.mean([r["p_lt1"] for r in recs]),
                                                                  np.mean([r["p_eq0"] for r in recs])), flush=True)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
