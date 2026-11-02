"""CPU smoke tests on tiny random backbones (no GPU needed).
  python tests/smoke_test.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only check: never touch a GPU
import copy
import os
import sys
import tempfile

import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from audioground.config import load_config
from audioground.data import AudioGroundDataset, collate, read_jsonl, parse_first_span
from audioground.model import AudioGround
from audioground.time_embedding import sinusoidal

torch.set_num_threads(16)
TINY = os.path.join(ROOT, "configs/tiny_cpu.yaml")
OK = []


def check(name, cond, msg=""):
    assert cond, f"FAIL {name}: {msg}"
    OK.append(name); print(f"  ok  {name} {msg}")


def build(*ov):
    cfg = load_config([TINY], list(ov))
    torch.manual_seed(0)
    return cfg, AudioGround(cfg.model)


def batch_for(cfg, idx=(0, 1, 2)):
    rows = read_jsonl(cfg.data.train_jsonl)
    ds = AudioGroundDataset(rows, cfg.model, cfg.data, cfg.model.whisper_path, train=True)
    return collate([ds[i] for i in idx]), ds


print("== parsing")
check("parse_to", parse_first_span("It occurs from 3.2 to 7.5 seconds.") == (3.2, 7.5))
check("parse_hyphen_rev", parse_first_span("9s - 4s") == (4.0, 9.0))
check("parse_none", parse_first_span("no timestamps here") is None)

print("== frame-level interpolation / chunking / windows")
cfg, m = build()
b, ds = batch_for(cfg)
durs = b["duration"].tolist()
check("chunks_per_sample", b["chunk_sample"].tolist() == [0, 1, 1, 2, 2, 2], str(b["chunk_sample"].tolist()))
check("frames_ceil_D_30", b["n_frames"].tolist() == [1500, 3000, 4500], str(b["n_frames"].tolist()))
toks, Ls = m.encode_audio(b)
K = round(0.333333 * 50)
check("window_count", Ls == [(T - K) // K + 1 for T in [1500, 3000, 4500]], f"L={Ls} (K={K} frames)")
check("tokens_shape", all(t.shape == (L, m.d_llm) for t, L in zip(toks, Ls)))
ids, att = m._timestamp_ids(3)
txt = m.bert_tokenizer.batch_decode(ids, skip_special_tokens=True)
check("timestamp_text", txt[1] == "this segment is from 0. 34 to 0. 68 seconds.", repr(txt[1]))

cfg_o, m_o = build("model.chunk_offset_sec=15")
b_o, _ = batch_for(cfg_o)
check("offset15_chunks", b_o["chunk_start"].tolist() == [0, 0, 750, 0, 750, 2250, 3750], str(b_o["chunk_start"].tolist()))
cfg_v, m_v = build("model.chunk_hop_sec=20")
b_v, _ = batch_for(cfg_v)
t_v, L_v = m_v.encode_audio(b_v)
check("overlap_chunks", b_v["chunk_sample"].tolist().count(2) == 4 and b_v["n_frames"].tolist()[2] == 4500,
      f"{b_v['chunk_sample'].tolist()} frames={b_v['n_frames'].tolist()}")

print("== absolute time embedding")
cfg_h, m_h = build("model.ate=hybrid")
c = torch.tensor([0.17, 10.0, 59.8])
check("hybrid_zero_init_eq_sin", torch.allclose(m_h.ate.embed(c), sinusoidal(c, m_h.d_llm), atol=1e-6))
check("beta_init", abs(float(m_h.ate.beta) - 0.05) < 1e-7)
for v in ["none", "sin", "learned", "hybrid"]:
    for ts in ["true", "false"]:
        cfg_x, m_x = build(f"model.ate={v}", f"model.timestamp_cond={ts}")
        out = m_x(b)
        out["loss"].backward()
        g_q = m_x.speech_query_tokens.grad is not None
        g_b = (v == "none") or m_x.ate.beta.grad is not None
        frozen = all(p.grad is None for p in m_x.speech_encoder.parameters()) and all(p.grad is None for p in m_x.beats.parameters())
        lora_g = any(p.grad is not None for n, p in m_x.named_parameters() if "lora_" in n)
        check(f"fwd_bwd ate={v} ts={ts}", torch.isfinite(out["loss"]) and g_q and g_b and frozen and lora_g,
              f"loss={out['loss'].item():.3f}")

print("== window / query sensitivity")
for W, nq in [(0.5, 2), (1.0, 4)]:
    cfg_w, m_w = build(f"model.window_sec={W}", f"model.stride_sec={W}", f"model.num_query_token={nq}")
    tw, Lw = m_w.encode_audio(b)
    check(f"W={W} N_Q={nq}", tw[1].shape[0] == Lw[1] * nq and Lw[1] == 3000 // round(W * 50), f"L={Lw}")

print("== SALMONN init (LoRA merge, Q-Former keys)")
base = load_config([TINY]).model.llama_path
from transformers import LlamaForCausalLM
ref = LlamaForCausalLM.from_pretrained(base).model.layers[0].self_attn.q_proj.weight
mine = m.llama_model.base_model.model.model.layers[0].self_attn.q_proj.weight
sd = torch.load(cfg.model.salmonn_ckpt)["model"]
A = sd["llama_model.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"]
B = sd["llama_model.base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"]
check("lora_merge", torch.allclose(mine, ref + B @ A * (32 / 8), atol=1e-5))
check("qformer_loaded", torch.equal(m.speech_llama_proj.weight, sd["speech_llama_proj.weight"]))
check("new_lora_r32", m.llama_model.base_model.model.model.layers[0].self_attn.q_proj.lora_A.default.weight.shape[0] == 32)

print("== E6 options")
cfg_e, m_e = build("model.aux_span_weight=0.5", "data.response_format=duration_first", "data.loss_reweight=inv_duration")
b_e, ds_e = batch_for(cfg_e)
check("duration_first_text", b_e["response"][0].startswith("The event lasts 3.6 seconds."), b_e["response"][0])
check("loss_weight", b_e["loss_weight"][0].item() > 1.0, str(b_e["loss_weight"].tolist()))
o = m_e(b_e); o["loss"].backward()
check("aux_span_loss", "span_loss" in o and m_e.span_head[0].weight.grad is not None, f"span={o['span_loss'].item():.3f}")
# regression: rows without a target span (NaN) must not produce NaN gradients
m_e.zero_grad(set_to_none=True)
b_e["span_norm"][1] = float("nan")
o = m_e(b_e); o["loss"].backward()
finite = all(torch.isfinite(p.grad).all() for p in m_e.parameters() if p.grad is not None)
check("aux_span_nan_rows_finite_grads", torch.isfinite(o["loss"]) and finite)

print("== generation (beam 4) + batched == single + save/load")
m.eval()
g = OmegaConf.create({"num_beams": 4, "do_sample": False, "max_new_tokens": 8, "min_length": 1,
                      "repetition_penalty": 1.0, "length_penalty": 1.0})
outs = m.generate(b, g)
check("generate", len(outs) == 3, repr(outs[0][:60]))
single = m.generate(collate([ds[2]]), g)
check("left_pad_consistent", single[0] == outs[2], f"{single[0]!r} vs {outs[2]!r}")
with torch.no_grad():   # stronger: first-step greedy scores of a left-padded batch == unpadded single sample
    def first_scores(bb):
        at, _ = m.encode_audio(bb)
        e, a, _, _ = m.build_inputs(at, bb["instruction"], None, left_pad=True)
        o = m.llama_model.generate(inputs_embeds=e, attention_mask=a, max_new_tokens=1, num_beams=1, do_sample=False,
                                   output_scores=True, return_dict_in_generate=True, pad_token_id=0)
        return o.scores[0]
    sb, ss = first_scores(collate([ds[2], ds[0]])), first_scores(collate([ds[0]]))  # ds[0] (12 s) is left-padded
    check("left_pad_logits_match", torch.allclose(sb[1], ss[0], atol=1e-3), f"maxdiff={(sb[1]-ss[0]).abs().max():.2e}")
sd_t = m.trainable_state_dict()
with tempfile.TemporaryDirectory() as td:
    torch.save({"model": sd_t}, f"{td}/ck.pt")
    _, m2 = build()
    torch.manual_seed(123)
    for n, p in m2.named_parameters():
        if p.requires_grad:
            p.data.normal_()
    m2.load_state_dict(torch.load(f"{td}/ck.pt")["model"], strict=False)
    m2.eval()
    check("save_load_roundtrip", m2.generate(b, g) == outs)
print(f"\nALL {len(OK)} CHECKS PASSED")
