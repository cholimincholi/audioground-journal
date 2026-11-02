"""Qwen2-Audio-7B-Instruct adapter (HF Qwen/Qwen2-Audio-7B-Instruct, transformers path).

Official path (model card "Audio Analysis Inference"):
  conversation = [system "You are a helpful assistant.", user [audio, text]]
  text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
  audio = librosa.load(path, sr=processor.feature_extractor.sampling_rate)[0]   # 16 kHz mono
  inputs = processor(text=text, audio=[audio], return_tensors="pt", padding=True)
  model.generate(**inputs, ...); decode new tokens with skip_special_tokens=True.
The processor's chat template renders the audio as "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n".

Long audio (> 30 s):
  The feature extractor is a WhisperFeatureExtractor with n_samples=480000 (30 s @ 16 kHz),
  called with padding="max_length" and truncation=True (default), so the official path SILENTLY
  KEEPS ONLY THE FIRST 30 s and pads shorter audio to 30 s (attention mask marks the real part).
  The audio encoder itself is hard-limited to 3000 mel frames (max_source_positions=1500), so a
  single segment longer than 30 s cannot be fed. Adapter kwarg `long_audio`:
    "truncate" (default, = official behaviour): first 30 s only.
    "chunk": NON-OFFICIAL. Split into consecutive <=30 s chunks and pass them as multiple audio
             items in one user turn ("Audio 1: ... Audio 2: ..."), which the official multi-audio
             chat template supports. The model was not trained to treat chunks as one timeline,
             so timestamps beyond 30 s are unlikely to be meaningful; use only as an ablation.
  For the clipped setting (L = 30 s) both modes are identical.

Generation: gen = {num_beams:4, do_sample:False, max_new_tokens:100} is honoured. The checkpoint's
generation_config.json also sets repetition_penalty=1.1 (plus top_k/top_p/temperature, which are
inert with do_sample=False); repetition_penalty is kept as the official default (override with
--extra '{"repetition_penalty": 1.0}').

Kwargs (--extra JSON): dtype ("auto": bf16 on cuda, fp32 on cpu), long_audio, system_prompt,
attn_implementation (default "sdpa"), repetition_penalty (None = checkpoint default).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "ckpt/hf_home")

import common  # noqa: E402

DEFAULT_CKPT = "ckpt/baselines/qwen2_audio"
SYSTEM_PROMPT = "You are a helpful assistant."


class Model:
    def __init__(self, ckpt_dir, device, dtype="auto", long_audio="truncate", system_prompt=SYSTEM_PROMPT,
                 attn_implementation="sdpa", repetition_penalty=None, **kw):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        assert long_audio in ("truncate", "chunk"), long_audio
        self.torch = torch
        self.device = device
        self.long_audio = long_audio
        self.system_prompt = system_prompt
        self.repetition_penalty = repetition_penalty
        ckpt_dir = ckpt_dir or DEFAULT_CKPT
        if dtype == "auto":
            dtype = "bfloat16" if str(device).startswith("cuda") else "float32"
        self.processor = AutoProcessor.from_pretrained(ckpt_dir)
        self.sr = self.processor.feature_extractor.sampling_rate  # 16000
        self.max_samples = self.processor.feature_extractor.n_samples  # 480000 = 30 s
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            ckpt_dir, dtype=getattr(torch, dtype), attn_implementation=attn_implementation
        ).to(device).eval()

    def generate(self, audio_path, prompt, gen):
        import librosa

        audio = librosa.load(audio_path, sr=self.sr, mono=True)[0]
        if self.long_audio == "chunk" and len(audio) > self.max_samples:
            chunks = [audio[i:i + self.max_samples] for i in range(0, len(audio), self.max_samples)]
        else:
            chunks = [audio]  # processor truncates to the first 30 s (official)
        content = [{"type": "audio", "audio_url": audio_path} for _ in chunks]
        content.append({"type": "text", "text": prompt})
        conv = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=chunks, sampling_rate=self.sr, return_tensors="pt", padding=True)
        inputs = inputs.to(self.device)
        inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
        kw = dict(num_beams=gen["num_beams"], do_sample=gen["do_sample"], max_new_tokens=gen["max_new_tokens"])
        if not gen["do_sample"]:
            kw.update(temperature=None, top_p=None, top_k=None)
        if self.repetition_penalty is not None:
            kw["repetition_penalty"] = self.repetition_penalty
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **kw)
        out = out[:, inputs["input_ids"].size(1):]
        return self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


if __name__ == "__main__":
    common.main(Model, "qwen2_audio")
