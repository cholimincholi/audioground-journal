#!/bin/bash
# Training-dynamics evaluation: kept checkpoints (stepNNNNN.pt) of a run on
#   v1 val (diag_default/ground), v2 val (diag_lead/ground) and the FULL Clotho-Moment test (Original, 6,649 queries).
#   usage: CUDA_VISIBLE_DEVICES=<gpu> [STEPS="00500 01500 ..."] bash experiments/eval/dynamics_eval.sh <run_dir>
RUN=${1:?run dir}
cd "${AG_ROOT:-.}"
export AG_BENCHMARKS="diag_default/ground diag_lead/ground clotho_moment/original"
for ck in $(ls $RUN/step*.pt | sort); do
  st=$(basename $ck .pt); st=${st#step}
  [[ -n "${STEPS:-}" && " $STEPS " != *" $st "* ]] && continue
  tag=dyn_step$st
  [ -f $RUN/preds/$tag/DONE ] && continue
  bash experiments/infer/infer_benchmarks.sh $RUN $tag --ckpt $(basename $ck)
done
