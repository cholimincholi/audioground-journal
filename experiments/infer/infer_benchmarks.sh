#!/bin/bash
# Run inference of one trained run on all benchmarks, keep raw outputs, then quick-score (parse rate, R1).
#   CUDA_VISIBLE_DEVICES=0,1 ./infer_benchmarks.sh <RUN_DIR> <TAG> [--ckpt best.pt] [key=value overrides ...]
# e.g. TAG=offset15 with model.chunk_offset_sec=15 ; TAG=overlap10 with model.chunk_hop_sec=20
# Output: <RUN_DIR>/preds/<TAG>/<bench>_<setting>.jsonl (+ .meta.json), scores.json, DONE
# For runs without a trained ckpt (zero-shot), pass --ckpt "" and RUN_DIR containing config.yaml (created if missing).
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/../env.sh"
RUN=${1:?run dir}; TAG=${2:?tag}; shift 2
CKPT=best.pt
if [[ "${1:-}" == "--ckpt" ]]; then CKPT=$2; shift 2; fi
NGPU=${AG_NPROC:-$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | grep -c .)}
OUT=$RUN/preds/$TAG; mkdir -p "$OUT"
echo "$@" > "$OUT/overrides.txt"
PORT=$("$AG_PY" -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")   # free port
cd "$AG_CODE"
INS=(); OUTS=()
for B in $AG_BENCHMARKS; do
  IN=$AG_BENCH_ROOT/$B.jsonl; NAME=${B//\//_}
  [[ -f $IN ]] || { echo "missing benchmark $IN, skipping"; continue; }
  [[ -f $OUT/$NAME.jsonl ]] && { echo "exists: $OUT/$NAME.jsonl"; continue; }
  INS+=("$IN"); OUTS+=("$OUT/$NAME.jsonl")
done
if (( ${#INS[@]} > 0 )); then
  SETS=(--set generate.batch_size=$AG_INFER_BS "$@")
  echo "[$(date)] infer $RUN [$TAG] ckpt=$CKPT on ${#INS[@]} files: ${INS[*]}"
  TPL=(); [[ -n "${AG_INSTR_TEMPLATE:-}" ]] && TPL=(--instruction_template "$AG_INSTR_TEMPLATE")   # e.g. the shared baseline prompt
  "$AG_TORCHRUN" --nproc_per_node "$NGPU" --master_port "$PORT" infer.py --run_dir "$RUN" --ckpt "$CKPT" --precision "$AG_INFER_PRECISION" "${TPL[@]}" \
     --input "$(IFS=,; echo "${INS[*]}")" --output "$(IFS=,; echo "${OUTS[*]}")" "${SETS[@]}" 2>&1 | tee -a "$OUT/infer.log"
  [[ ${PIPESTATUS[0]} -eq 0 ]] || exit 1
fi
"$AG_PY" tools/score.py "$OUT"/*_*.jsonl > "$OUT/scores.json"
cat "$OUT/scores.json"
touch "$OUT/DONE"
