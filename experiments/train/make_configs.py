"""(Re)generates one YAML per experiment in experiments/train/configs/.
Every config inherits the paper default (code/audioground/configs/audioground_7b.yaml) and only states its diff.
Data paths use ${oc.env:AG_DATA_ROOT,...} so dataset variants can be relocated with one env var.
    python make_configs.py
"""
import os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "configs")
BASE = "../../../code/audioground/configs/audioground_7b.yaml"  # relative to the config file
DATA = "${oc.env:AG_DATA_ROOT,data}"


def variant(name):
    return {"train_jsonl": f"{DATA}/variants/{name}/train.jsonl", "val_jsonl": f"{DATA}/variants/{name}/val.jsonl"}


EXPS = {}
# ---------------- E0 main
EXPS["E0_main"] = ("AudioGround default: FI + timestamp-conditioned Q-Former + hybrid ATE (paper config); full 1000-sample val",
                   {"data": {"max_val_samples": 1000}})

# ---------------- E1 component ablation (Reviewer 1 #2)
for name, ts, ate, desc in [
    ("E1_fi", False, "none", "FI only"),
    ("E1_fi_ts", True, "none", "FI + timestamp"),
    ("E1_fi_sin", False, "sin", "FI + sinusoidal ATE (no timestamp)"),
    ("E1_fi_ts_sin", True, "sin", "FI + timestamp + sinusoidal ATE"),
    ("E1_fi_ts_learned", True, "learned", "FI + timestamp + learned ATE"),
    ("E1_fi_ts_hybrid", True, "hybrid", "FI + timestamp + hybrid ATE (== E0_main; queue aliases it)"),
    ("E1_fi_hybrid", False, "hybrid", "FI + hybrid ATE (no timestamp)"),
]:
    EXPS[name] = (desc, {"model": {"timestamp_cond": ts, "ate": ate}})

# ---------------- E2 window / query sensitivity (R3 #6)
for w, wtag in [(0.333333, "0.33"), (0.5, "0.5"), (1.0, "1.0")]:
    for nq in [1, 2, 4]:
        name = f"E2_W{wtag}_NQ{nq}"
        desc = f"window = stride = {wtag}s, N_Q = {nq}" + (" (== E0_main)" if (wtag == "0.33" and nq == 1) else "")
        EXPS[name] = (desc, {"model": {"window_sec": w, "stride_sec": w, "num_query_token": nq}})
for b in [0.01, 0.2]:
    EXPS[f"E2_beta{b}"] = (f"beta init {b} (0.05 == E0_main)", {"model": {"ate_beta_init": b}})

# ---------------- E3 chunking (R5) -- inference-only variants are queue `infer` jobs on E0; this one retrains
EXPS["E3_overlap_train"] = ("train + infer with overlapping 30 s chunks, hop 20 s (10 s overlap), centre-weighted merge",
                            {"model": {"chunk_hop_sec": 20.0, "overlap_merge": "center"}})
EXPS["E3_offset15_train"] = ("train with the chunk grid shifted by 15 s (boundary-shift control)",
                             {"model": {"chunk_offset_sec": 15.0}})

# ---------------- E4 data ablations (dataset variants built by the data agent under $AG_DATA_ROOT/variants/<name>)
for name, var, desc in [
    ("E4_nogap", "nogap", "no silence gap at junctions"),
    ("E4_crossfade", "crossfade", "crossfaded junctions"),
    ("E4_bgmix", "bgmix", "background-mixed junctions"),
    ("E4_tau0.25", "tau0.25", "similarity threshold tau = 0.25"),
    ("E4_tau0.41", "tau0.41", "similarity threshold tau = 0.41"),
    ("E4_strong", "strong", "AudioSet-Strong tightened labels"),
]:
    EXPS[name] = (f"data variant: {desc}", {"data": variant(var)})

# ---------------- E5 multitask controlled (R5). unified == E0_main (6000 steps ~= 2.945 epochs of 48.9K)
EP = round(6000 * 24 / 48900, 4)
EXPS["E5_grounding_only_steps"] = ("grounding task only, matched total steps (6000)", {"data": {"tasks": ["grounding"]}})
EXPS["E5_grounding_only_epochs"] = (f"grounding task only, matched epochs ({EP})",
                                    {"data": {"tasks": ["grounding"]}, "train": {"num_epochs": EP}})
EXPS["E5_conventional_steps"] = ("all tasks with conventional task-specific outputs, matched steps",
                                 {"data": variant("conventional")})
EXPS["E5_conventional_epochs"] = (f"all tasks with conventional outputs, matched epochs ({EP})",
                                  {"data": variant("conventional"), "train": {"num_epochs": EP}})
EXPS["E5_unified_epochs"] = (f"unified outputs, matched epochs ({EP}); identical to E0 when the dataset has 48.9K rows",
                             {"train": {"num_epochs": EP}})

# ---------------- E6 duration-collapse remedies (R3 #4, R4 #2)
EXPS["E6a_duration_first"] = ("(a) explicit duration-first output: 'The event lasts D seconds. <response>'",
                              {"data": {"response_format": "duration_first"}})
EXPS["E6b_reweight"] = ("(b1) short-target loss reweighting w = clip((10/dur)^0.5, 1, 4)",
                        {"data": {"loss_reweight": "inv_duration", "reweight_ref_sec": 10.0, "reweight_gamma": 0.5,
                                  "reweight_max": 4.0}})
EXPS["E6b_auxspan"] = ("(b2) auxiliary span head (L1 + GIoU) on prompt-final hidden state, weight 0.5",
                       {"model": {"aux_span_weight": 0.5}})
EXPS["E6c_oversample"] = ("(c) curriculum oversampling of spans < 5 s: x3 decaying to x1 over training",
                          {"data": {"short_oversample_factor": 3.0, "short_threshold_sec": 5.0,
                                    "short_oversample_schedule": "decay"}})
EXPS["E6_all"] = ("(a)+(b1)+(b2)+(c) combined",
                  {"model": {"aux_span_weight": 0.5},
                   "data": {"response_format": "duration_first", "loss_reweight": "inv_duration",
                            "short_oversample_factor": 3.0, "short_threshold_sec": 5.0,
                            "short_oversample_schedule": "decay"}})

EXPS["E6d_shortspan_data"] = ("(d) data-side remedy: 'shortspan' dataset variant built by the data agent",
                              {"data": variant("shortspan")})

# ---------------- E7 baseline: vanilla SALMONN (no temporal modules) + LoRA on AudioGround-IT (R3 #3)
EXPS["E7_salmonn_ft"] = ("vanilla SALMONN connector (zero-pad BEATs, no timestamp, no ATE) + same chunking, LoRA r32, same data/steps",
                         {"model": {"timestamp_cond": False, "ate": "none", "beats_align": "pad"}})
EXPS["E7_salmonn_ft_interp"] = ("E7 with frame-level interpolation (isolates temporal modules from FI)",
                                {"model": {"timestamp_cond": False, "ate": "none", "beats_align": "interp"}})
EXPS["E7_salmonn_zeroshot"] = ("vanilla SALMONN-7B, no fine-tuning (inference only; lora off -> pure SALMONN weights)",
                               {"model": {"timestamp_cond": False, "ate": "none", "beats_align": "pad", "lora": False}})

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, (desc, diff) in EXPS.items():
        body = {"_base_": BASE, **diff}
        with open(os.path.join(OUT, f"{name}.yaml"), "w") as f:
            f.write(f"# {name}: {desc}\n")
            yaml.safe_dump(body, f, sort_keys=False)
    print(f"wrote {len(EXPS)} configs to {OUT}")
