#!/bin/bash
# SALMONN-7B raw (no fine-tuning) with the shared baseline prompt v1, all 6 settings; GPUs come from CUDA_VISIBLE_DEVICES (queue runner).
export AG_INSTR_TEMPLATE='Given the audio, find the time interval in which the following sound event occurs: "{query}". The audio is {L:.1f} seconds long. Answer with the start and end time in seconds, in the format: from <start> to <end> seconds.'
export AG_INFER_BS=4 AG_INFER_PRECISION=bf16
cd "${AG_ROOT:-.}"
bash experiments/infer/infer_benchmarks.sh ${AG_ROOT:-.}/runs/E7_salmonn_zeroshot/0 promptv1 --ckpt ''
