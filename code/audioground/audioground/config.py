"""Config handling: defaults <- YAML files (with `_base_` inheritance) <- CLI dotlist overrides."""
import os
from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
os.environ.setdefault("AG_ROOT", REPO_ROOT)
CKPT = os.environ.setdefault("AG_CKPT", os.path.join(os.environ["AG_ROOT"], "ckpt"))

DEFAULTS = {
    "model": {
        # ---- backbone paths ----
        "llama_path": f"{CKPT}/vicuna-7b-v1.5",
        "whisper_path": f"{CKPT}/whisper-large-v2",
        "beats_path": f"{CKPT}/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        "bert_path": f"{CKPT}/bert-base-uncased",
        # SALMONN checkpoint used to initialise Q-Former / proj / LNs (+ optionally LoRA). "" = none.
        "salmonn_ckpt": f"{CKPT}/salmonn_7b/salmonn_7b_v0.pth",
        "salmonn_lora": "merge",          # merge | drop : SALMONN's r=8 LoRA merged into LLM weights or discarded
        "salmonn_lora_alpha": 32,         # SALMONN LoRA scaling (alpha/r = 32/8)
        "freeze_whisper": True,
        "freeze_beats": True,
        "use_beats": True,
        # ---- long audio / chunking (Frame-level Interpolation section) ----
        "chunk_sec": 30.0,                # encoder chunk length (Whisper max 30 s)
        "chunk_hop_sec": None,            # None -> = chunk_sec (non-overlapping). < chunk_sec -> overlapping chunks
        "chunk_offset_sec": 0.0,          # shift chunk grid: first chunk = [0, offset), then offset + k*hop
        "overlap_merge": "center",        # mean | center | first : how overlapping frames are merged
        "keep_final_padding": True,       # True: final chunk keeps its zero-padded frames (ceil(D/30)*1500 frames, as in paper)
        "beats_align": "interp",          # interp (paper: linear interpolation to Whisper length) | pad (vanilla SALMONN)
        "max_audio_sec": 600.0,           # hard cap on audio length (0 = no cap)
        "encoder_batch_size": 16,         # chunks per encoder forward
        # ---- window-level Q-Former ----
        "window_sec": 0.333333,           # W_sec
        "stride_sec": 0.333333,           # S_sec
        "num_query_token": 1,             # N_Q
        "qformer_layers": 2,
        "timestamp_cond": True,           # textual timestamp conditioning in Q-Former
        "timestamp_template": "This segment is from {s} to {e} seconds.",
        "timestamp_decimals": 2,
        "timestamp_source": "frames",     # frames: s = start_frame/50 (exact) | nominal: s = (l-1)*S_sec (paper formula)
        "qformer_text_init": "bert",      # bert: init word/pos emb + text FFN from bert-base-uncased | random
        "query_init_multi": "repeat",     # if N_Q > ckpt N_Q: repeat (+small noise) | random
        # ---- absolute time embedding ----
        "ate": "hybrid",                  # none | sin | learned | hybrid
        "ate_beta_init": 0.05,
        "ate_beta_learnable": True,
        "ate_time_scale": 1.0,            # c is multiplied by this before TE (seconds by default)
        "ate_learned_hidden": 0,          # hidden size of learned MLP(t); 0 -> d_LLM
        "ate_learned_t_scale": 0.01,      # learned MLP input = t * scale (t in seconds)
        # ---- LLM / LoRA ----
        "lora": True,
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "lora_target_modules": ["q_proj", "v_proj"],
        "prompt_template": "USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:",
        "end_sym": "</s>",
        "max_txt_len": 300,
        "gradient_checkpointing": False,
        "llm_sdpa": True,                 # fused causal SDPA in LLaMA for right-padded train/val forwards (exact)
        # ---- E6: duration-collapse remedies ----
        "aux_span_weight": 0.0,           # >0 enables auxiliary span regression head (L1 + GIoU) on prompt-final hidden state
        "aux_span_giou_weight": 1.0,
        "aux_span_l1_weight": 1.0,
    },
    "data": {
        "train_jsonl": "${oc.env:AG_DATA_ROOT,${oc.env:AG_ROOT}/data}/variants/default/train.jsonl",
        "val_jsonl": "${oc.env:AG_DATA_ROOT,${oc.env:AG_ROOT}/data}/variants/default/val.jsonl",
        "tasks": None,                    # optional task filter on the "task" field, e.g. [grounding] (E5)
        "audio_root": "",                 # prefix for relative audio paths
        # offline frozen-encoder feature cache root ("" = encode online). Layout: <root>/<chunk_tag>/<hh>/<sha1>.{w,b}.npy
        "feature_cache": "${oc.env:AG_FEAT_CACHE,${oc.env:AG_ROOT}/cache/feats}",
        "sample_rate": 16000,
        "num_workers": 8,
        "max_val_samples": 300,          # fixed first-N subset of val for ckpt selection (E0 final runs: 1000)
        "response_format": "as_is",       # as_is | duration_first  (E6a)
        "duration_first_template": "The event lasts {dur} seconds. {response}",
        # E6b: per-sample loss reweighting for short targets
        "loss_reweight": "none",          # none | inv_duration
        "reweight_ref_sec": 10.0,
        "reweight_gamma": 0.5,
        "reweight_max": 4.0,
        # E6c: oversampling of short spans (curriculum)
        "short_oversample_factor": 1.0,   # 1.0 = off
        "short_threshold_sec": 5.0,
        "short_oversample_schedule": "constant",  # constant | decay (factor -> 1) | warmup (1 -> factor)
        "oversample_target": "short",     # short (span < short_threshold_sec) | strongcrop (AudioSet-Strong event targets)
    },
    "train": {
        "seed": 0,
        "output_dir": "${oc.env:AG_ROOT}/runs/debug",
        "max_steps": 6000,
        "num_epochs": None,               # if set, overrides max_steps = ceil(N * epochs / global_batch)
        "global_batch_size": 24,
        "micro_batch_size": 3,            # per GPU; grad accumulation = global / (micro * world)
        "lr": 3.0e-5,
        "min_lr": 1.0e-5,
        "warmup_steps": 100,
        "warmup_start_lr": 1.0e-6,
        "lr_schedule": "cosine",          # cosine (SALMONN default) | constant | linear
        "weight_decay": 0.05,
        "beta1": 0.9,
        "beta2": 0.999,
        "grad_clip": 1.0,
        "precision": "bf16",              # bf16 | fp16 | fp32
        "log_every": 10,
        "eval_every": 1000,
        "save_every": 1000,
        "keep_ckpt_every": 0,          # >0: also keep trainable weights every N steps (stepNNNNN.pt)
        "device": "cuda",
        "resume": "",
        "init_from": "",                  # load AudioGround trainable weights from this ckpt before training
    },
    "generate": {
        "num_beams": 4,
        "do_sample": False,
        "max_new_tokens": 100,
        "min_length": 1,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "batch_size": 8,
    },
}


def _load_yaml(path):
    cfg = OmegaConf.load(path)
    base = cfg.pop("_base_", None)
    if base is None:
        return cfg
    bases = [base] if isinstance(base, str) else list(base)
    merged = OmegaConf.create({})
    for b in bases:
        bp = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
        merged = OmegaConf.merge(merged, _load_yaml(bp))
    return OmegaConf.merge(merged, cfg)


def load_config(cfg_paths=(), overrides=()):
    cfg = OmegaConf.create(DEFAULTS)
    for p in cfg_paths or ():
        cfg = OmegaConf.merge(cfg, _load_yaml(p))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
