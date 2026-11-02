#!/usr/bin/env bash
# Run all baseline LALMs x benchmarks x {original, clipped} on a configurable GPU list.
#
#   GPUS=0,1 MODELS="qwen2_audio qwen25_omni" bash experiments/eval/run_baselines.sh
#
# Env vars (all optional):
#   GPUS        comma-separated GPU ids (default 0).
#               One worker per GPU; workers pull
#               jobs from a shared queue (flock), so any mix of models/benches load-balances.
#   MODELS      subset of: qwen2_audio qwen25_omni desta25 gama af2 salmonn timeaudio
#   BENCHES     subset of: clotho_moment unav100 tut2017
#   SETTINGS    subset of: original clipped
#   PROMPT_ID   prompt variant in eval/runners/common.py (default v1 = shared paper-style prompt)
#   CLOTHO_SHARDS  number of shards for Clotho-Moment (6,649 queries) per model/setting (default: #GPUs)
#   LIMIT       if set, only the first N queries (smoke test; outputs get a .limitN suffix and are ignored by make_tables)
#   CHECK_FREE  if 1 (default), a worker skips launching while its GPU has < MIN_FREE_MB free memory
#   MIN_FREE_MB default 40000
# Outputs: ${AG_ROOT:-.}/runs/baselines/<model>/<bench>_<setting>[.shardIofK].jsonl  (resumable)
# After all jobs: shards are merged and experiments/eval/make_tables.py is run.
set -uo pipefail

ROOT=${AG_ROOT:-.}
RUNNERS=$ROOT/eval/runners
ENVS=$ROOT/envs
RUNS=$ROOT/runs/baselines
LOGS=$ROOT/runs/baselines/_logs
mkdir -p "$RUNS" "$LOGS"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # all baseline assets are cached locally (avoid HF 429 rate limit)
GPUS=${GPUS:-0}
MODELS=${MODELS:-"qwen2_audio qwen25_omni desta25 gama af2 salmonn timeaudio"}
BENCHES=${BENCHES:-"clotho_moment unav100 tut2017"}
SETTINGS=${SETTINGS:-"original clipped"}
PROMPT_ID=${PROMPT_ID:-v1}
IFS=, read -ra GPU_ARR <<< "$GPUS"
NG=${#GPU_ARR[@]}
CLOTHO_SHARDS=${CLOTHO_SHARDS:-$NG}
CHECK_FREE=${CHECK_FREE:-1}
MIN_FREE_MB=${MIN_FREE_MB:-40000}
LIMIT_ARG=""; [ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit $LIMIT"

# model -> python interpreter of its env   (filled in by the setup step; see eval/runners/*.py docstrings)
declare -A PY=(
  [qwen2_audio]=$ENVS/qwen/bin/python
  [qwen25_omni]=$ENVS/qwen/bin/python
  [desta25]=$ENVS/desta25/bin/python
  [salmonn]=$ENVS/salmonn/bin/python
  [gama]=$ENVS/gama/bin/python
  [af2]=$ENVS/af2/bin/python
  [timeaudio]=$ENVS/timeaudio/bin/python   # NOTE: env has torch 2.6.0+cpu; install the CUDA torch 2.6.0 wheel before GPU runs
)
# extra per-model CLI args (e.g. ckpt dirs are defaulted inside each adapter)
declare -A EXTRA=(
  [qwen2_audio]=""
  [qwen25_omni]=""
  [desta25]=""
  [salmonn]=""
  [gama]=""
  [af2]=""
  [timeaudio]=""
)

QUEUE=$(mktemp "$LOGS/queue.XXXX")
for m in $MODELS; do
  if [ ! -x "${PY[$m]:-/nonexistent}" ] || [ ! -f "$RUNNERS/$m.py" ]; then
    echo "[skip] $m: env (${PY[$m]:-?}) or runner ($RUNNERS/$m.py) missing"; continue
  fi
  for b in $BENCHES; do
    for s in $SETTINGS; do
      if [[ "$b" = clotho_moment || "$b" = tacos ]] && [ "$CLOTHO_SHARDS" -gt 1 ]; then
        for ((i=0;i<CLOTHO_SHARDS;i++)); do echo "$m $b $s $i $CLOTHO_SHARDS" >> "$QUEUE"; done
      else
        echo "$m $b $s 0 1" >> "$QUEUE"
      fi
    done
  done
done
echo "jobs: $(wc -l < "$QUEUE")  gpus: ${GPU_ARR[*]}  queue: $QUEUE"

pop_job() {  # atomically pop first line of the queue
  ( flock -x 9
    line=$(head -n1 "$QUEUE")
    [ -n "$line" ] && sed -i '1d' "$QUEUE"
    echo "$line"
  ) 9>"$QUEUE.lock"
}

worker() {
  local gpu=$1
  while true; do
    job=$(pop_job); [ -z "$job" ] && break
    read -r m b s i k <<< "$job"
    if [ "$CHECK_FREE" = 1 ]; then
      while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
        [ "${free:-0}" -ge "$MIN_FREE_MB" ] && break
        echo "[gpu$gpu] only ${free}MB free, waiting..."; sleep 120
      done
    fi
    log="$LOGS/${m}_${b}_${s}_${i}of${k}.log"
    echo "[gpu$gpu] start $m $b $s shard $i/$k -> $log"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONNOUSERSITE=1 HF_HOME=$ROOT/ckpt/hf_home TOKENIZERS_PARALLELISM=false \
      "${PY[$m]}" "$RUNNERS/$m.py" --device cuda:0 --bench "$b" --setting "$s" \
      --shard "$i" --nshards "$k" --prompt-id "$PROMPT_ID" $LIMIT_ARG ${EXTRA[$m]} >> "$log" 2>&1
    echo "[gpu$gpu] done  $m $b $s shard $i/$k (exit $?)"
  done
}

for g in "${GPU_ARR[@]}"; do worker "$g" & done
wait
rm -f "$QUEUE" "$QUEUE.lock"

$ENVS/eval/bin/python $RUNNERS/merge_shards.py "$RUNS"
$ENVS/eval/bin/python $ROOT/experiments/eval/make_tables.py --runs $ROOT/runs --prompt-id "$PROMPT_ID"
