#!/bin/bash
# Launch one AudioGround training run with torchrun.
#   CUDA_VISIBLE_DEVICES=0,1 ./launch.sh <EXP_NAME|config.yaml> <SEED> [key=value overrides ...]
# Output: $AG_RUNS/<EXP_NAME>/<SEED>/  (config.yaml, train.log, metrics.jsonl, best.pt, last.pt, DONE)
# Global batch stays 24 for any GPU count: micro-batch / grad-accumulation are derived automatically.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../env.sh"
EXP=${1:?exp name or config path}; SEED=${2:?seed}; shift 2
if [[ -f "$EXP" ]]; then CFG=$(readlink -f "$EXP"); EXP=$(basename "$EXP" .yaml); else CFG=$AG_EXP_CONFIGS/$EXP.yaml; fi
[[ -f $CFG ]] || { echo "no config $CFG"; exit 1; }
NGPU=${AG_NPROC:-$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | grep -c .)}
GB=24; MICRO=""
for m in ${AG_MICRO:-} 3 4 2 6 1; do if (( GB % (m * NGPU) == 0 )); then MICRO=$m; break; fi; done
[[ -n $MICRO ]] || { echo "global batch $GB not divisible for $NGPU GPUs"; exit 1; }
OUT=$AG_RUNS/$EXP/$SEED; mkdir -p "$OUT"
PORT=$("$AG_PY" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")   # free port
RESUME=""; [[ -f $OUT/last.pt && ! -f $OUT/DONE ]] && RESUME="train.resume=$OUT/last.pt"
echo "[$(date)] $EXP seed=$SEED gpus=$CUDA_VISIBLE_DEVICES micro=$MICRO accum=$(( GB / (MICRO*NGPU) )) out=$OUT $RESUME" | tee -a "$OUT/launch.log"
cd "$AG_CODE"
"$AG_TORCHRUN" --nproc_per_node "$NGPU" --master_port "$PORT" train.py --cfg "$CFG" \
  --set train.seed=$SEED train.output_dir=$OUT train.micro_batch_size=$MICRO train.global_batch_size=$GB $RESUME "$@" \
  2>&1 | tee -a "$OUT/launch.log"
exit ${PIPESTATUS[0]}
