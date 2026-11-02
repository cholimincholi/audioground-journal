"""DeSTA2.5-Audio (DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B) baseline adapter (official code path).

Code:    eval/third_party/DeSTA2.5-Audio (github kehanlu/DeSTA2.5-Audio @ e9b28ff,
         installed `pip install -e --no-deps` into envs/desta25; unmodified).
Weights: HF cache layout under ckpt/baselines/desta25/hf_cache:
           DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B   (connector/Q-Former, model.safetensors)
           DeSTA-ntu/Llama-3.1-8B-Instruct         (the LLM id hard-coded in the DeSTA config; an
                                                     ungated re-upload of Llama-3.1-8B-Instruct. Its
                                                     files/tokenizer differ in hash from the meta-llama
                                                     copy in the shared cache, so the official
                                                     DeSTA-referenced repo is used, not the shared copy)
           openai/whisper-large-v3                 (encoder + ASR decoder)
         Silero-VAD (torch.hub snakers4/silero-vad) under .../desta25/torch_hub.
The adapter points HF_HOME / HF_HUB_CACHE at that cache (DeSTA's from_pretrained uses
os.getenv("HF_HOME") as cache_dir) and sets HF_HUB_OFFLINE=1, so the official model ids resolve locally.

Inference = README "Basic Usage" / DeSTA25AudioModel.generate:
  messages = [system: "Focus on the audio clips and instructions." (README quick-start),
              user: "<|AUDIO|>\n{prompt}", audios=[{audio: path, text: None}]]
  Llama-3.1 chat template (tokenizer.apply_chat_template), audio -> 64 Q-Former tokens + Whisper ASR
  transcript tokens (ASR only if Silero-VAD finds speech, else " ").
  Audio handling (official): AudioSegment.from_file(target_sr=16000, channel average); the
  WhisperFeatureExtractor pads/TRUNCATES to 30 s, so both the encoder features and the ASR transcript
  only cover the first 30 s of longer (original-setting, up to 60 s) audio; VAD runs on the full clip.
  Kept as is.
Generation deviation: DeSTA25AudioModel.generate() exposes no `num_beams`; to honour the paper's
  deterministic beam search (num_beams=4, do_sample=False, max_new_tokens=100) the adapter wraps
  llm_model.generate to inject num_beams (everything else, incl. eos ids from the Llama generation
  config, is the official path). Pass --num-beams 1 for official greedy decoding.
dtype: official (LLM bf16, Whisper + connector fp32) on every device.
"""
from __future__ import annotations

import os
import site
import sys

# ignore ~/.local site-packages (they shadow env packages on this machine: torchvision mismatch)
sys.path[:] = [p for p in sys.path if not p.startswith(site.USER_SITE or "~")]

CKPT_DEFAULT = "ckpt/baselines/desta25"
REPO_ID = "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B"
SYSTEM_PROMPT = "Focus on the audio clips and instructions."

RUNNERS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNERS)
import common  # noqa: E402


def _set_env(d):
    os.environ["HF_HOME"] = os.path.join(d, "hf_cache")
    os.environ["HF_HUB_CACHE"] = os.path.join(d, "hf_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["TORCH_HOME"] = os.path.join(d, "torch_hub")


class Model:
    def __init__(self, ckpt_dir=None, device="cuda:0", **kw):
        d = ckpt_dir or CKPT_DEFAULT
        _set_env(d)  # before importing transformers / huggingface_hub
        import torch
        from desta import DeSTA25AudioModel

        if str(device).startswith("cpu"):
            torch.set_num_threads(int(kw.get("threads", os.cpu_count() or 1)))
        self.torch = torch
        model = DeSTA25AudioModel.from_pretrained(REPO_ID)
        model.to(device)
        model.eval()
        self.model = model
        self.system_prompt = kw.get("system_prompt", SYSTEM_PROMPT)
        self._orig_llm_generate = model.llm_model.generate

    def generate(self, audio_path, prompt, gen):
        import functools
        torch = self.torch
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "<|AUDIO|>\n" + prompt.strip(),
             "audios": [{"audio": audio_path, "text": None}]},
        ]
        nb = int(gen.get("num_beams", 1))
        self.model.llm_model.generate = functools.partial(self._orig_llm_generate, num_beams=nb)
        try:
            with torch.inference_mode():
                out = self.model.generate(
                    messages=messages,
                    do_sample=gen.get("do_sample", False),
                    top_p=1.0,
                    temperature=1.0,
                    max_new_tokens=gen.get("max_new_tokens", 100),
                )
        finally:
            self.model.llm_model.generate = self._orig_llm_generate
        self.last_transcript = out.audios[0][1] if out.audios else None
        return out.text[0].strip()


if __name__ == "__main__":
    common.main(Model, "desta25")
