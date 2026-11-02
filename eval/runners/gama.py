"""GAMA-IT (Ghosh et al., EMNLP 2024) adapter for the AudioGround baseline driver.

Official code : https://github.com/Sreyan88/GAMA  (cloned to eval/third_party/GAMA)
Weights       : HF Space sonalkum/GAMA-IT (the official GAMA-IT demo; same files as the README's
                Google-Drive folder):
                  <ckpt>/Llama-2-7b-chat-hf-qformer/   base LLaMA-2-7B-chat + audio encoder + Q-Former
                  <ckpt>/stage5_ckpt/pytorch_model.bin  stage-5 (instruction-tuned, CompA-R) LoRA +
                                                       projector/aggregator weights  == "GAMA-IT"
Code path     : mirrors inference_gradio_gama_it.py (official GAMA-IT inference):
  * LlamaForCausalLM from GAMA's modified transformers (4.28.0.dev0, hf-dev-train/transformers-main),
    fp32, LoRA r=8/alpha=16 on q_proj,v_proj (dropout 0), stage-5 state dict loaded strict=False,
    pad=0 / bos=1 / eos=2.
  * Prompt template 'alpaca_short':  "### Instruction:\\n{prompt}\\n\\n### Response:\\n"
  * Audio: torchaudio load -> resample to 16 kHz -> mean removal -> Kaldi fbank (128 mel, 10 ms hop,
    25 ms hanning) -> pad / TRUNCATE to 1024 frames -> (x+5.081)/4.4849.

>10 s / >30 s audio (OFFICIAL BEHAVIOUR KEPT): GAMA has a fixed 1024-frame (10.24 s) input.
  Longer audio is hard-truncated to the FIRST 10.24 s; shorter audio is zero-padded. There is no
  chunking / sliding window in the official code. Consequently, for 30 s clipped and up to 60 s
  original-setting audio the model only hears the first ~10 s; the prompt still states the full
  duration L (shared prompt v1). This is a property of the baseline, not of the runner.

Deviations from the official script (documented):
  1. Decoding: official GAMA-IT uses sampling (do_sample=True, temperature=0.1, top_p=0.95,
     max_new_tokens=400). We honour the shared paper setting gen={num_beams:4, do_sample:False,
     max_new_tokens:100}. GAMA's modified transformers only wires `audio_input` into `sample()`;
     `beam_search()` silently drops the audio. We therefore install a patched COPY of GAMA's
     transformers (eval/third_party/gama_transformers_cpu_beam) whose beam_search mirrors sample():
     audio is repeat_interleaved over beams, the attention mask is extended by the 96 audio tokens,
     and audio is only fed at step 0 (afterwards it lives in past_key_values). No other change.
     num_beams=1 + do_sample=False would hit the unpatched greedy_search (audio dropped), so
     num_beams must be > 1 (asserted) -- or pass --extra '{"official_sampling": true}' to use the
     official sampling config instead.
  2. Output trimming: official code does tokenizer.decode(seq)[len(prompt)+6:-4]; we decode only the
     newly generated token ids with skip_special_tokens=True (equivalent when generation ends with
     </s>, and robust when it hits max_new_tokens).
  3. Device: official script uses device_map="auto" on CUDA; here model.to(device) (CPU or a single
     GPU), fp32 as in the official script. --extra '{"dtype": "bf16"}' optional.
"""
from __future__ import annotations

import os
import sys

RUNNERS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNERS)
GAMA_REPO = "eval/third_party/GAMA"
DEFAULT_CKPT = "ckpt/baselines/gama"
os.environ.setdefault("HF_HOME", "ckpt/hf_home")  # bert-base-uncased (Q-Former cfg)

import common  # noqa: E402


class Model:
    def __init__(self, ckpt_dir, device, dtype="fp32", official_sampling=False, **kw):
        import json
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaForCausalLM, LlamaTokenizer

        ckpt_dir = ckpt_dir or DEFAULT_CKPT
        base = os.path.join(ckpt_dir, "Llama-2-7b-chat-hf-qformer")
        stage5 = os.path.join(ckpt_dir, "stage5_ckpt", "pytorch_model.bin")
        self.torch = torch
        self.device = device
        self.dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
        self.official_sampling = official_sampling
        with open(os.path.join(GAMA_REPO, "templates", "alpaca_short.json")) as f:
            self.template = json.load(f)  # == utils.prompter.Prompter('alpaca_short')

        self.tokenizer = LlamaTokenizer.from_pretrained(base)
        model = LlamaForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
        config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0,
                            bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
        sd = torch.load(stage5, map_location="cpu")
        msg = model.load_state_dict(sd, strict=False)
        n_unexp = len(msg.unexpected_keys)
        print(f"[gama] loaded stage5 ({len(sd)} tensors), unexpected={n_unexp}", flush=True)
        assert n_unexp == 0, msg.unexpected_keys[:10]
        model.is_parallelizable = True
        model.model_parallel = True
        model.config.pad_token_id = self.tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        self.model = model.to(device=device, dtype=self.dtype).eval()

    def load_audio(self, filename):
        torch, torchaudio = self.torch, __import__("torchaudio")
        waveform, sr = torchaudio.load(filename)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr,
                                                  use_energy=False, window_type="hanning",
                                                  num_mel_bins=128, dither=0.0, frame_shift=10)
        target_length = 1024  # 10.24 s -- official fixed length (truncate / zero-pad)
        p = target_length - fbank.shape[0]
        if p > 0:
            fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        return (fbank + 5.081) / 4.4849

    def generate(self, audio_path, prompt, gen):
        torch = self.torch
        from transformers import GenerationConfig
        text = self.template["prompt_no_input"].format(instruction=prompt)
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        audio = self.load_audio(audio_path).unsqueeze(0).to(self.device, dtype=self.dtype)
        if self.official_sampling:
            gc = dict(do_sample=True, temperature=0.1, top_p=0.95, max_new_tokens=400)
        else:
            assert gen["num_beams"] > 1 and not gen["do_sample"], \
                "GAMA audio is only wired into sample()/patched beam_search(); greedy would drop audio"
            gc = dict(do_sample=False, num_beams=gen["num_beams"], max_new_tokens=gen["max_new_tokens"])
        generation_config = GenerationConfig(**gc, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                                             num_return_sequences=1)
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, audio_input=audio,
                                      generation_config=generation_config,
                                      return_dict_in_generate=True, output_scores=False,
                                      max_new_tokens=gc["max_new_tokens"])
        seq = out.sequences[0][input_ids.shape[1]:]
        return self.tokenizer.decode(seq, skip_special_tokens=True).strip()


if __name__ == "__main__":
    common.main(Model, "gama")
