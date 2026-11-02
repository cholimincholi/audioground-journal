#!/bin/bash
# E8L_qwen25omni_ft: LoRA fine-tune Qwen2.5-Omni-7B (thinker) on AudioGround-IT "lead" with our recipe, then evaluate.
#   CUDA_VISIBLE_DEVICES=<one GPU id> ./qwen25omni_ft_run.sh <SEED> [SETTINGS="original"]   # SETTINGS e.g. original,clipped
# Output: $AG_RUNS/E8L_qwen25omni_ft/<SEED>/ {args.json, train.log, metrics.jsonl, last.pt, best_adapter/, DONE,
#         launch.log, preds/default/<bench>_<setting>.jsonl + infer_meta.json + DONE}
# Re-running resumes (training from last.pt; inference per qid). Tunables (env): MICRO (default 3 = our yaml), GRAD_CKPT (0; the
# trainer's preflight turns it on automatically if micro-batch MICRO OOMs on the longest samples), NUM_WORKERS (12),
# INFER_BS (1 = zero-shot runner path; keep fixed across compared runs), BENCHES.
set -euo pipefail
SEED=${1:?usage: CUDA_VISIBLE_DEVICES=N $0 <SEED> [SETTINGS]}
SETTINGS=${2:-original}

# ---------------------------------------------------------------- GPU policy (same rule as experiments/env.sh)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
[[ -n ${CUDA_VISIBLE_DEVICES+x} ]] || { echo "REFUSED: CUDA_VISIBLE_DEVICES unset (set it to one GPU id)" >&2; exit 2; }
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | grep -c . || true)
# NGPU > 1: torchrun, every rank takes every NGPU-th micro-batch of the same global batch (identical update to 1 GPU)

export PYTHONNOUSERSITE=1 HF_HOME=${AG_ROOT:-.}/ckpt/hf_home HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}   # long, variable-length sequences
PY=python
RUNNERS=${AG_ROOT:-.}/eval/runners
AG_RUNS=${AG_RUNS:-${AG_ROOT:-.}/runs}
OUT=$AG_RUNS/${QWEN_EXP:-E8L_qwen25omni_ft}/$SEED   # QWEN_EXP=E8_qwen25omni_ft_v1 with TRAIN_JSONL/VAL_JSONL for the rho0=22% data
mkdir -p "$OUT"
echo "[$(date)] E8L_qwen25omni_ft seed=$SEED gpu=$CUDA_VISIBLE_DEVICES micro=${MICRO:-3} settings=$SETTINGS out=$OUT" | tee -a "$OUT/launch.log"

if [[ ! -f $OUT/DONE ]]; then
  set +e
  if [[ $NGPU -gt 1 ]]; then LAUNCH=(torchrun --standalone --nproc_per_node "$NGPU"); else LAUNCH=("$PY"); fi
  "${LAUNCH[@]}" "$RUNNERS/qwen25_omni_ft_train.py" --output-dir "$OUT" --seed "$SEED" --device cuda \
      --micro-batch "${MICRO:-3}" ${TRAIN_JSONL:+--train-jsonl "$TRAIN_JSONL"} ${VAL_JSONL:+--val-jsonl "$VAL_JSONL"} --grad-ckpt "${GRAD_CKPT:-0}" --num-workers "${NUM_WORKERS:-12}" \
      2>&1 | tee -a "$OUT/launch.log"
  rc=${PIPESTATUS[0]}; set -e
  [[ $rc == 0 && -f $OUT/DONE ]] || { echo "training failed (rc=$rc)" >&2; exit 1; }
fi
[[ -f $OUT/best_adapter/adapter_config.json ]] || { echo "no best_adapter in $OUT" >&2; exit 1; }

# inference: one benchmark per GPU in parallel (round-robin over the visible GPUs)
IFS=, read -ra GL <<< "$CUDA_VISIBLE_DEVICES"; IFS=, read -ra BL <<< "${BENCHES:-clotho_moment,unav100,tut2017}"
pids=(); k=0
for b in "${BL[@]}"; do
  g=${GL[$((k % ${#GL[@]}))]}; k=$((k+1))
  CUDA_VISIBLE_DEVICES=$g "$PY" "$RUNNERS/qwen25_omni_ft_infer.py" --adapter "$OUT/best_adapter" --out-dir "$OUT/preds/default" \
      --benches "$b" --settings "$SETTINGS" --device cuda:0 --batch-size "${INFER_BS:-1}" >> "$OUT/infer_$b.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
[[ $rc == 0 ]] || { echo "inference failed (see $OUT/infer_*.log)" >&2; exit 1; }
echo "[$(date)] finished seed=$SEED" | tee -a "$OUT/launch.log"
