#!/bin/bash
# lead-based ablation variants (2026-10-09): same synthesis as lead + one changed factor each
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=""
D=${AG_ROOT:-.}/data; cd $D/scripts
python make_conventional.py lead conventional_lead && echo "[$(date)] conventional_lead done"
for v in crossfade bgmix nogap tau0.25 tau0.41 strong; do
  python synth.py --config ../configs/${v}_lead.yaml > $D/../runs/logs/synth_${v}_lead.log 2>&1 && echo "[$(date)] ${v}_lead done" || echo "[$(date)] ${v}_lead FAILED"
  python validate.py $D/variants/${v}_lead/train.jsonl 300 2>&1 | tail -1
done
echo "[$(date)] ALL_LEAD_VARIANTS_DONE"
