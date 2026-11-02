#!/bin/bash
# Sequentially run probe 1 (shift inference) and probe 2 (onset prior) for every checkpoint, ONE process on ONE GPU
# at a time; a GPU in {4..7} is used only if it has >= 35 GB free at launch (else poll every 2 min).
#   setsid nohup bash eval/scripts/probes/run_probes.sh > runs/logs/probes.log 2>&1 &
set -uo pipefail
source ${AG_ROOT:-.}/experiments/env.sh
P=${AG_ROOT:-.}/data/probes/shift
RUNS="E0_main/1 E0_main/2 E7_salmonn_ft/0 E0L_main/0 E0L_main/1 E0L_main/2 E7L_salmonn_ft/0"
STEPS=${STEPS:-"prior shift"}
NEED=35000

pick_gpu() {
  while true; do
    for g in 4 5 6 7; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g | tr -d ' ')
      if (( free >= NEED )); then echo $g; return; fi
    done
    echo "[$(date)] no GPU with >= ${NEED} MiB free; waiting" >&2
    sleep 120
  done
}

for R in $RUNS; do
  RUN=$AG_RUNS/$R
  for S in $STEPS; do
    if [[ $S == shift ]]; then
      OUT=$RUN/preds/probe_shift
      [[ -f $OUT/DONE ]] && { echo "skip $R shift"; continue; }
      mkdir -p $OUT
      G=$(pick_gpu); echo "[$(date)] $R shift on GPU $G"
      INS=""; OUTS=""
      for d in 0 3 7 15; do INS+="$P/shift_d$d.jsonl,"; OUTS+="$OUT/shift_d$d.jsonl,"; done
      (cd $AG_CODE && CUDA_VISIBLE_DEVICES=$G $AG_PY infer.py --run_dir $RUN --ckpt best.pt --precision bf16 \
          --input ${INS%,} --output ${OUTS%,} \
          --set generate.batch_size=8 generate.num_beams=4 generate.max_new_tokens=100 data.feature_cache=null \
          > $OUT/infer.log 2>&1) && touch $OUT/DONE || echo "FAILED $R shift"
    else
      OUT=$RUN/preds/probe_prior
      [[ -f $OUT/DONE ]] && { echo "skip $R prior"; continue; }
      mkdir -p $OUT
      G=$(pick_gpu); echo "[$(date)] $R prior on GPU $G"
      CUDA_VISIBLE_DEVICES=$G $AG_PY ${AG_ROOT:-.}/eval/scripts/probes/onset_prior.py --run_dir $RUN --bs 4 \
          > $OUT/run.log 2>&1 && touch $OUT/DONE || echo "FAILED $R prior"
    fi
  done
done
echo "[$(date)] ALL DONE"
