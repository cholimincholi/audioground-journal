#!/bin/bash
# TACOS silence probes (0.4 s prepended / 0.4 s inserted away from every boundary) for all seeds of one experiment
EXP=$1; SEEDS=${2:-"0 1 2"}
cd "${AG_ROOT:-.}"
export AG_BENCHMARKS="tacosprobe/pre04 tacosprobe/insert04"
for s in $SEEDS; do bash experiments/infer/infer_benchmarks.sh ${AG_ROOT:-.}/runs/$EXP/$s tacosprobe || exit 1; done
