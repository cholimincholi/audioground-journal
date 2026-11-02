"""AudioGround evaluation library (rebuilt pipeline).

Modules
-------
clipping  : 30-s window selection / remapping exactly as in the paper's Evaluation section
parser    : free-form LALM response -> (start, end) interval(s)
metrics   : IoU, R1@{0.5,0.7} under old / parse-failure-as-miss protocols, multi-interval
            scoring, paired bootstrap CIs and p-values, prediction diagnostics
baselines : trivial (non-model) baselines
data      : unified benchmark loaders
"""
EVAL_ROOT = "eval"
BENCH_ROOT = f"{EVAL_ROOT}/benchmarks"
