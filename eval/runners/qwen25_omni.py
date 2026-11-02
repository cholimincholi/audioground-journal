"""Qwen2.5-Omni-7B adapter (HF Qwen/Qwen2.5-Omni-7B, transformers + qwen-omni-utils path).

Official path (model card):
  model = Qwen2_5OmniForConditionalGeneration.from_pretrained(...); model.disable_talker()
  processor = Qwen2_5OmniProcessor.from_pretrained(...)
  conversation = [system <official Qwen system prompt>, user [audio, text]]
  text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
  audios, images, videos = process_mm_info(conv, use_audio_in_video=False)   # qwen_omni_utils, librosa @16 kHz
  inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
                     padding=True, use_audio_in_video=False)
  text_ids = model.generate(**inputs, use_audio_in_video=False, return_audio=False)
Text-only output: the model is loaded with enable_audio_output=False, so the talker and token2wav
modules are never instantiated (their weights are skipped at load -> "unused weights" warnings are
expected); disable_talker() is also called as on the model card; generate(return_audio=False).
System prompt: the model card's official one (all card examples use it):
  "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving
   auditory and visual inputs, as well as generating text and speech."
(override with --extra '{"system_prompt": "..."}').

Long audio: the processor's WhisperFeatureExtractor is configured with n_samples=4,800,000
(chunk_length=300 s), and the audio encoder processes features in n_window windows, so the
official path feeds the FULL audio up to 300 s (truncated beyond that). Our audio (<= 60 s) is
therefore passed whole; no adapter-side truncation. Kwarg `max_audio_s` (default None = official)
can force truncation to the first N seconds for ablations.

Generation: gen (num_beams=4, do_sample=False, max_new_tokens=100) is routed to the thinker via
thinker_* kwargs. The checkpoint generation_config has no sampling defaults for the thinker that
interfere with beam search.

Kwargs (--extra JSON): dtype ("auto": bf16 on cuda, fp32 on cpu), system_prompt, max_audio_s,
attn_implementation (default "sdpa"; "flash_attention_2" needs flash-attn, not installed).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "ckpt/hf_home")

import common  # noqa: E402

DEFAULT_CKPT = "ckpt/baselines/qwen25_omni"
SYSTEM_PROMPT = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
                 "perceiving auditory and visual inputs, as well as generating text and speech.")


class Model:
    def __init__(self, ckpt_dir, device, dtype="auto", system_prompt=SYSTEM_PROMPT, max_audio_s=None,
                 attn_implementation="sdpa", **kw):
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.torch = torch
        self.device = device
        self.system_prompt = system_prompt
        self.max_audio_s = max_audio_s
        ckpt_dir = ckpt_dir or DEFAULT_CKPT
        if dtype == "auto":
            dtype = "bfloat16" if str(device).startswith("cuda") else "float32"
        self.processor = Qwen2_5OmniProcessor.from_pretrained(ckpt_dir)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            ckpt_dir, dtype=getattr(torch, dtype), attn_implementation=attn_implementation,
            enable_audio_output=False,
        )
        self.model.disable_talker()
        self.model.to(device).eval()

    def generate(self, audio_path, prompt, gen):
        from qwen_omni_utils import process_mm_info

        audio_item = {"type": "audio", "audio": audio_path}
        if self.max_audio_s is not None:
            audio_item.update(audio_start=0.0, audio_end=float(self.max_audio_s))
        conv = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [audio_item, {"type": "text", "text": prompt}]},
        ]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
                                padding=True, use_audio_in_video=False)
        inputs = inputs.to(self.device)
        inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs, use_audio_in_video=False, return_audio=False,
                thinker_num_beams=gen["num_beams"], thinker_do_sample=gen["do_sample"],
                thinker_max_new_tokens=gen["max_new_tokens"],
            )
        out = out[:, inputs["input_ids"].size(1):]
        return self.processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


if __name__ == "__main__":
    common.main(Model, "qwen25_omni")
