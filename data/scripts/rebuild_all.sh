#!/bin/bash
# Rebuild AudioGround-IT audio on a fresh server from raw downloads (2026-10-08).
# Stages: prep_clips -> embed_captions -> build_pools -> synth(default) -> diff against snapshot jsonl.
set -euo pipefail
export PYTHONNOUSERSITE=1 HF_HOME=${AG_ROOT:-.}/.cache/hf CUDA_VISIBLE_DEVICES=""
D=${AG_ROOT:-.}/data; PY=python; cd $D/scripts
if [ ! -f $D/clips/clips.parquet ]; then echo "[$(date)] prep_clips"
$PY prep_clips.py --parquet_glob "$D/raw/opensound_audiocaps/data/train-*.parquet" \
    --captions $D/raw/d0rj/data/train-00000-of-00001-dfc9e682a70b4097.parquet --out $D/clips --workers 96; fi
if [ ! -f $D/emb/nbr_sim.npy ]; then echo "[$(date)] embed_captions"
NTHREADS=64 $PY embed_captions.py --clips $D/clips/clips.parquet --out $D/emb; fi
echo "[$(date)] build_pools"
mkdir -p $D/pools; $PY build_pools.py --root $D
for v in ${VARIANTS:-default}; do
  echo "[$(date)] synth $v"
  mv $D/variants/$v $D/variants/${v}.snapshot_tmp 2>/dev/null || true
  $PY synth.py --config ../configs/$v.yaml
  for s in train val; do
    if cmp -s $D/variants/$v/$s.jsonl $D/snapshot_jsonl/variants/$v/$s.jsonl; then echo "IDENTICAL $v/$s.jsonl"
    else echo "DIFFERS $v/$s.jsonl"; fi
  done
  rm -rf $D/variants/${v}.snapshot_tmp
done
echo "[$(date)] REBUILD_DONE"
