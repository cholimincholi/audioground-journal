#!/bin/bash
# Synthesize remaining variants (bit-identical for >99.8% of rows), then restore the snapshot JSONL as canonical
# and render any audio it references that synth did not produce (render_from_meta.py).
set -uo pipefail
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""
D=${AG_ROOT:-.}/data; PY=python; cd $D/scripts
for v in crossfade bgmix nogap tau0.25 tau0.41 strong; do
  echo "[$(date)] synth $v"; $PY synth.py --config ../configs/$v.yaml > $D/../runs/logs/synth_$v.log 2>&1 || echo "synth $v FAILED"
done
for v in default crossfade bgmix nogap tau0.25 tau0.41 strong conventional shortspan; do
  for s in train val; do
    cmp -s $D/variants/$v/$s.jsonl $D/snapshot_jsonl/variants/$v/$s.jsonl && echo "IDENTICAL $v/$s" || echo "RESTORED-SNAPSHOT $v/$s"
    mkdir -p $D/variants/$v; cp $D/snapshot_jsonl/variants/$v/$s.jsonl $D/variants/$v/$s.jsonl
  done
  cp $D/snapshot_jsonl/variants/$v/stats.json $D/variants/$v/ 2>/dev/null
  echo "[$(date)] render missing $v"; $PY render_from_meta.py $D/variants/$v/train.jsonl $D/variants/$v/val.jsonl
done
echo "[$(date)] FINALIZE_DONE"
