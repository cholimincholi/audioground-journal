#!/bin/bash
# TACOS (real recordings) inference for all seeds of one experiment; GPUs from the queue runner.
#   bash experiments/eval/tacos_infer.sh <EXP> [seeds]
EXP=$1; SEEDS=${2:-"0 1 2"}
cd "${AG_ROOT:-.}"
export AG_BENCHMARKS="tacos/original"
for s in $SEEDS; do bash experiments/infer/infer_benchmarks.sh ${AG_ROOT:-.}/runs/$EXP/$s tacos || exit 1; done
