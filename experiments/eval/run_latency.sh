#!/bin/bash
# Latency / peak memory (paper protocol: 1 GPU alone, bs 1, fp16, 1 warm-up + median of 8 runs, 30 generated tokens, 60/90/120 s).
# Reported: ATE_l (final model). Measured for consistency checking only: ATE_h (main of the previous submission), SALMONN baseline.
set -e
cd ${AG_ROOT:-.}/code/audioground
source ${AG_ROOT:-.}/experiments/env.sh
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}       # use only the first GPU of the group
export PYTHONPATH=${AG_ROOT:-.}/code/audioground:${PYTHONPATH:-}
OUT=${AG_ROOT:-.}/eval/results/latency
WAV=${AG_ROOT:-.}/eval/benchmarks/clotho_moment/audio_original/$(ls ${AG_ROOT:-.}/eval/benchmarks/clotho_moment/audio_original | head -1)
for pair in "ate_l:E1L_fi_ts_learned/0" "ate_h:E0L_main/0" "salmonn_baseline:E7L_salmonn_ft/0"; do
  name=${pair%%:*}; run=${AG_ROOT:-.}/runs/${pair##*:}
  [ -f $OUT/$name.json ] && continue
  "$AG_PY" tools/latency.py --cfg $run/config.yaml --ckpt $run/best.pt --wav "$WAV" --tmp $OUT --out $OUT/$name.json
done
touch $OUT/DONE
