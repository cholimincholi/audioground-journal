# Shared settings for all experiment scripts (source this file). Override any variable from the environment.
export AG_ROOT=${AG_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}   # repository root
export AG_CODE=${AG_CODE:-$AG_ROOT/code/audioground}
export AG_PY=${AG_PY:-python}
export AG_TORCHRUN=${AG_TORCHRUN:-torchrun}
export AG_RUNS=${AG_RUNS:-$AG_ROOT/runs}
export AG_DATA_ROOT=${AG_DATA_ROOT:-$AG_ROOT/data}          # dataset variants: $AG_DATA_ROOT/variants/<name>/{train,val}.jsonl
export AG_BENCH_ROOT=${AG_BENCH_ROOT:-$AG_ROOT/eval/benchmarks}
# benchmarks evaluated after every training run: <bench>/<setting>.jsonl under $AG_BENCH_ROOT
export AG_BENCHMARKS=${AG_BENCHMARKS:-"clotho_moment/original clotho_moment/clipped unav100/original unav100/clipped tut2017/original tut2017/clipped"}
export AG_EXP_CONFIGS=${AG_EXP_CONFIGS:-$AG_ROOT/experiments/train/configs}
export HF_HOME=${HF_HOME:-$AG_ROOT/.cache/hf}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export AG_MICRO=${AG_MICRO:-12}      # per-GPU micro-batch (largest that fits, cap 12 on 2 GPUs); global batch stays 24 via accumulation

# CUDA indices follow nvidia-smi; select GPUs with CUDA_VISIBLE_DEVICES ("" = CPU only).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
ag_check_gpus() { :; }
export AG_FEAT_CACHE=${AG_FEAT_CACHE:-$AG_ROOT/cache/feats}
export AG_INFER_PRECISION=${AG_INFER_PRECISION_OVERRIDE:-bf16}   # 2026-10-09: always bf16 (matches training, ~2x faster) unless AG_INFER_PRECISION_OVERRIDE is set; batch size FIXED (AG_INFER_BS=16) so outputs are reproducible
export AG_INFER_BS=${AG_INFER_BS:-16}
