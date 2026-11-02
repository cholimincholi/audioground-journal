"""TimeAudio baseline adapter (Wang et al., "Listening Between the Frames: Bridging Temporal Gaps in
Large Audio-Language Models", AAAI 2026 / arXiv:2511.11039; bib key wang2025listening).

Code    : https://github.com/lysanderism/TimeAudio  (cloned to eval/third_party/TimeAudio, commit 22db713, 2025-11-18)
Weights : https://huggingface.co/lysanderism/TimeAudio  (timeaudio.pth, sha 84daf76, 2025-11-17)
Backbone: SALMONN-style  Whisper-large-v2 enc + BEATs iter3+ AS2M cpt2 + window Q-Former (1/3 s windows,
          absolute-time position embedding) + Vicuna-7B-v1.5 with LoRA (r=8) + time tokens <a_d>/<f_d>.
Ckpt dir layout (default ckpt/baselines/timeaudio):
    pretrained_model/{vicuna-7b-v1.5, whisper-large-v2, bert-base-uncased,
                      BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt, timeaudio.pth}
  (official code hard-codes the relative path "pretrained_model/bert-base-uncased" for the Q-Former
   config, so the model is constructed with cwd = ckpt_dir.)

Official inference path reproduced (configs/infer_config.yaml + instruct_dataset.TimeAudioDataset +
TimeAudio.generate):
  * audio: mono (first channel), padded to >=1 s, truncated to 180 s (datasets.max_len=180).
    <=30 s -> one Whisper log-mel (padded to 30 s); >30 s -> 30 s chunks, each zero-padded to 30 s, a
    trailing chunk shorter than 5 s is DROPPED from the Whisper branch (official behaviour); BEATs gets the
    full raw waveform. duration = round(len/16k, 1) (official uses the annotation duration, rounded to .1).
    -> our 60 s "original" audio is handled as 2 Whisper chunks, nothing truncated.
  * prompt: template "USER: {}\nASSISTANT:" applied to "<Speech><SpeechHere></Speech> " + question.
  * decoding: generate() with stop token </s>; time tokens decoded by models.utils.decode_time_answer_v3
    (e.g. "<a_1><a_2><f_3> - ..." -> "12.3 - ..."). Answers look like "12.3 - 25.0".

Kwargs (--extra JSON):
  prompt_style : "shared" (default) -> use the driver's prompt (e.g. v1) verbatim as the question.
                 "native"           -> TimeAudio's own TAG test prompt (prompts/test_prompt.json["tag"]):
                     "What are the start and end times of audio matching '<query>'?"
                 The query is recovered from the driver prompt (text between the double quotes).
                 NOTE: common.py still records the shared prompt in the JSONL "prompt" field; run native
                 with a distinct --model-name (e.g. timeaudio_native) or --out. The effective prompt is
                 printed for the first query.
  dtype        : "fp32" (default on CPU) | "fp16" (default on CUDA: official = fp16 LLM weights + fp16
                 autocast) | "bf16".

Deviations from the official code:
  * Benchmark wavs are 32 kHz; official code assumes 16 kHz files (WhisperFeatureExtractor would reject
    other rates). We resample 32k->16k with librosa (soxr_hq) before the official preprocessing.
  * TimeAudio.generate hard-codes `torch.tensor([2]).cuda()` for the stop criterion; we re-implement the
    identical generate() with the stop tensor on the model device (needed for CPU).
  * On CPU the whole model is cast to fp32 (official uses CUDA fp16 autocast); output is post-processed
    only by stripping "</s>"/"<unk>" and surrounding whitespace.
  * gen = {num_beams:4, do_sample:False, max_new_tokens:100} honoured (official default
    max_new_tokens=200, num_beams=4, do_sample=False, min_length=1, length/repetition penalty 1.0 kept).
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import site
import sys

# keep ~/.local site-packages (incompatible torchvision/transformers) out of this env
_us = site.getusersitepackages() if hasattr(site, "getusersitepackages") else None
sys.path[:] = [p for p in sys.path if not (_us and p.startswith(_us))]

RUNNERS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNERS)
import common  # noqa: E402

REPO = "eval/third_party/TimeAudio"
DEFAULT_CKPT = "ckpt/baselines/timeaudio"
os.environ.setdefault("HF_HOME", "ckpt/hf_home")

NATIVE_TAG_PROMPT = "What are the start and end times of audio matching '{query}'?"
PROMPT_TEMPLATE = "USER: {}\nASSISTANT:"
SR = 16000
MAX_LEN = 180

_QUERY_RES = [
    re.compile(r'occurs: "(.*)"\.', re.S),            # v1 / v1_nodur
    re.compile(r'described as "(.*)" occur', re.S),    # agit
    re.compile(r'"(.*)"', re.S),                       # fallback: first..last double quote
]


def extract_query(prompt: str) -> str:
    for r in _QUERY_RES:
        m = r.search(prompt)
        if m:
            return m.group(1).strip()
    raise ValueError(f"cannot recover query from prompt: {prompt!r}")


class Model:
    def __init__(self, ckpt_dir=None, device="cpu", prompt_style="shared", dtype=None, **kw):
        if kw:
            raise TypeError(f"unknown kwargs {kw}")
        assert prompt_style in ("shared", "native"), prompt_style
        self.prompt_style = prompt_style
        ckpt_dir = os.path.abspath(ckpt_dir or DEFAULT_CKPT)
        pm = os.path.join(ckpt_dir, "pretrained_model")
        sys.path.insert(0, REPO)
        import torch
        from omegaconf import OmegaConf
        from transformers import WhisperFeatureExtractor
        from models.timeaudio import TimeAudio
        from models.utils import StoppingCriteriaSub, decode_time_answer_v3
        self.torch = torch
        self.StoppingCriteriaSub = StoppingCriteriaSub
        self.decode_v3 = decode_time_answer_v3

        cfg = OmegaConf.load(os.path.join(REPO, "configs/infer_config.yaml")).model
        cfg.llama_path = os.path.join(pm, "vicuna-7b-v1.5")
        cfg.whisper_path = os.path.join(pm, "whisper-large-v2")
        cfg.beats_path = os.path.join(pm, "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
        cfg.ckpt = os.path.join(pm, "timeaudio.pth")
        cfg.prompt_path = os.path.join(REPO, cfg.prompt_path)
        cfg.test_prompt_path = os.path.join(REPO, cfg.test_prompt_path)

        cwd = os.getcwd()
        os.chdir(ckpt_dir)  # official code loads "pretrained_model/bert-base-uncased" relatively
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):  # from_config prints every ckpt key
                model = TimeAudio.from_config(cfg)
        finally:
            os.chdir(cwd)
        # sanity: report ckpt keys that did not match model params (load is strict=False upstream)
        sd = torch.load(cfg.ckpt, map_location="cpu")["model"]
        own = set(model.state_dict().keys())
        unexpected = [k for k in sd if k not in own]
        print(f"[timeaudio] ckpt tensors={len(sd)} unexpected={len(unexpected)} {unexpected[:5]}", flush=True)
        if unexpected:
            raise RuntimeError(f"ckpt keys not matched by model: {unexpected[:10]}")
        del sd

        self.device = torch.device(device)
        if dtype is None:
            dtype = "fp16" if self.device.type == "cuda" else "fp32"
        self.dtype = dtype
        if self.device.type == "cuda" and dtype == "fp16":
            model.to(self.device)  # official: fp16 LLM weights, fp32 rest, fp16 autocast
            self.autocast = lambda: torch.autocast("cuda", dtype=torch.float16)
        else:
            tdt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]
            model.to(self.device, dtype=tdt)
            self.autocast = contextlib.nullcontext
        model.eval()
        self.model = model
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.whisper_path)
        self._printed = False

    # ---- official TimeAudioDataset.__getitem__ + collater, for one sample ----
    def _prepare(self, audio_path):
        import numpy as np
        import soundfile as sf
        import librosa
        torch = self.torch
        audio, sr = sf.read(audio_path)  # float64, like official
        if audio.ndim == 2:
            audio = audio[:, 0]
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
            sr = SR
        duration = round(min(len(audio), sr * MAX_LEN) / sr, 1)
        if len(audio) < sr:
            audio = np.concatenate((audio, np.zeros(sr - len(audio), dtype=float)), axis=0)
        audio = audio[: sr * MAX_LEN]
        if len(audio) <= sr * 30:
            spec = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]  # (1,80,3000)
        else:
            specs = []
            for start in range(0, len(audio), sr * 30):
                chunk = audio[start:min(start + sr * 30, len(audio))]
                if sr * 5 <= len(chunk) <= sr * 30:
                    chunk = np.pad(chunk, (0, sr * 30 - len(chunk)), "constant")
                else:
                    continue
                specs.append(self.wav_processor(chunk, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze())
            spec = torch.stack(specs, dim=0).unsqueeze(0)  # (1,n,80,3000)
        raw = torch.from_numpy(audio).unsqueeze(0)
        fdt = next(self.model.speech_encoder.parameters()).dtype
        return {
            "spectrogram": spec.to(self.device, dtype=fdt),
            "raw_wav": raw.to(self.device),
            "padding_mask": torch.zeros(raw.shape, dtype=torch.bool, device=self.device),
            "duration": torch.tensor([duration], device=self.device),
        }

    def generate(self, audio_path, prompt, gen):
        torch = self.torch
        from transformers import StoppingCriteriaList
        if self.prompt_style == "native":
            question = NATIVE_TAG_PROMPT.format(query=extract_query(prompt))
        else:
            question = prompt.strip()
        prompts = [PROMPT_TEMPLATE.format("<Speech><SpeechHere></Speech> " + question)]
        if not self._printed:
            print(f"[timeaudio] effective prompt ({self.prompt_style}): {prompts[0]!r}", flush=True)
            self._printed = True
        samples = self._prepare(audio_path)
        m = self.model
        with torch.inference_mode(), self.autocast():
            # == TimeAudio.generate, with the stop tensor on the right device ==
            speech_embeds, speech_atts = m.encode_speech(
                samples["spectrogram"], raw_wav=samples["raw_wav"],
                audio_padding_mask=samples["padding_mask"], absolute_timestamp=samples["duration"])
            speech_embeds, speech_atts = m.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=True)
            embed_tokens = m.llama_model.model.model.embed_tokens if m.lora else m.llama_model.model.embed_tokens
            bos = torch.ones([1, 1], dtype=torch.long, device=speech_embeds.device) * m.llama_tokenizer.bos_token_id
            bos_embeds = embed_tokens(bos)
            embeds = torch.cat([bos_embeds, speech_embeds.to(bos_embeds.dtype)], dim=1)
            attns = torch.cat([speech_atts[:, :1], speech_atts], dim=1)
            stop = StoppingCriteriaList([self.StoppingCriteriaSub(stops=[torch.tensor([2], device=embeds.device)])])
            out = m.llama_model.generate(
                inputs_embeds=embeds, attention_mask=attns, stopping_criteria=stop,
                max_new_tokens=gen.get("max_new_tokens", 200), num_beams=gen.get("num_beams", 4),
                do_sample=gen.get("do_sample", False), min_length=1, temperature=1.0, top_p=0.9,
                repetition_penalty=1.0, length_penalty=1.0)
        text = m.llama_tokenizer.batch_decode(out, add_special_tokens=False)
        if "<f_" in text[0]:
            text = self.decode_v3(text)
        return text[0].replace("</s>", "").replace("<unk>", "").strip()


if __name__ == "__main__":
    common.main(Model, "timeaudio")
