#!/bin/bash
# Creates conda env `audioground` matching SALMONN (salmonn branch) requirements.txt
set -e
CONDA=${CONDA:-conda}
$CONDA create -y -n audioground python=3.9.17 -c conda-forge
P="$CONDA run -n audioground pip"
$P install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
$P install transformers==4.28.0 peft==0.3.0 accelerate==0.20.3 sentencepiece==0.1.97 soundfile librosa \
   "numpy<2" pyyaml omegaconf tensorboard tqdm "huggingface_hub<0.20" safetensors
# gradio==3.23.0 and bitsandbytes==0.35.0 intentionally skipped (web demo / 8-bit low-resource only)
echo ENV_DONE
