#!/bin/bash
# Downloads all weights needed for AudioGround into ${AG_ROOT:-.}/ckpt
export HF_HOME=${AG_ROOT:-.}/.cache/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
C=${AG_ROOT:-.}/ckpt
mkdir -p $C/logs $C/beats
L=$C/logs/download_status.txt
: > $L
dl() { # name, cmd...
  name=$1; shift
  if "$@" > $C/logs/$name.log 2>&1; then echo "OK   $name" >> $L; else echo "FAIL $name (see logs/$name.log)" >> $L; fi
}
dl salmonn_7b  hf download tsinghua-ee/SALMONN-7B salmonn_7b_v0.pth README.md --local-dir $C/salmonn_7b &
dl beats wget -q -O $C/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D" &
dl whisper hf download openai/whisper-large-v2 --local-dir $C/whisper-large-v2 --exclude "*.msgpack" "*.h5" "*.ot" "flax_model*" "tf_model*" "pytorch_model.bin" &
dl vicuna-7b-v1.5 hf download lmsys/vicuna-7b-v1.5 --local-dir $C/vicuna-7b-v1.5 &
dl bert-base-uncased hf download google-bert/bert-base-uncased config.json vocab.txt tokenizer.json tokenizer_config.json model.safetensors --local-dir $C/bert-base-uncased &
wait
echo DONE >> $L
