"""SALMONN-13B v1 baseline adapter (official code path).

Model: tsinghua-ee/SALMONN `salmonn_v1.pth` (SALMONN-13B v1) on top of
  - LLM:      lmsys/vicuna-13b-v1.1
  - Whisper:  openai/whisper-large-v2 (encoder only)
  - BEATs:    BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
    (official OneDrive link is auth-walled; fetched from HF mirror camenduru/beats,
     sha256 e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c, identical across
     3 independent HF mirrors)
Weights: ckpt/baselines/salmonn/
Code:    eval/third_party/SALMONN  (copy of eval/third_party/SALMONN,
         branch `salmonn` = SALMONN-13B v1 inference/training code). Only patch: in
         SALMONN.generate the stop-word tensor `torch.tensor([2]).cuda()` -> placed on the model device
         (needed for CPU; no behavioural change on GPU).
(SALMONN-7B `salmonn_7b_v0.pth` is also on HF (tsinghua-ee/SALMONN-7B, vicuna-7b-v1.5) but it is not
 used: the 13B v1 checkpoint is the official/paper SALMONN baseline.)

Inference = cli_inference.py:
  * audio: soundfile read, stereo -> channel 0, pad to >= 1 s, TRUNCATE TO THE FIRST 30 s
    (utils.prepare_one_sample: `audio = audio[: sr * 30]`), Whisper features + raw wav for BEATs.
    => For audio longer than 30 s (original-setting, up to 60 s) SALMONN only hears the first 30 s.
    This official behaviour is kept.
    prepare_one_sample passes the file's native sr to the Whisper feature extractor, which only
    accepts 16 kHz; benchmark wavs are 32 kHz, so the adapter resamples to 16 kHz (librosa) first and
    then applies exactly the same pad/truncate logic.
  * prompt: decode_config prompt_template "USER: {}\nASSISTANT:" with
    "<Speech><SpeechHere></Speech> " + prompt.strip()
  * generation: model.generate(samples, generate_cfg, prompts) with decode_config defaults
    (num_beams=4, do_sample=False, min_length=1, temperature=1.0, top_p=0.9, repetition_penalty=1.0,
    length_penalty=1.0) except max_new_tokens/num_beams taken from `gen` (paper: 4 beams, 100 tokens;
    official decode_config uses 200 tokens).
  * dtype: on CUDA exactly as cli_inference (LLaMA fp16 weights, rest fp32, torch.cuda.amp.autocast
    fp16). On CPU the official code disables autocast, so the whole model is cast to one dtype
    (default float32 = lossless upcast of the fp16 release; --extra '{"cpu_dtype":"bf16"}' for bf16).
  * output: official `batch_decode(outputs, add_special_tokens=False)` (special tokens not skipped);
    the adapter only strips leading/trailing "<s>"/"</s>"/whitespace.
"""
from __future__ import annotations

import os
import site
import sys

# ignore ~/.local site-packages (they shadow env packages on this machine)
sys.path[:] = [p for p in sys.path if not p.startswith(site.USER_SITE or "~")]
os.environ.setdefault("HF_HOME", "ckpt/hf_home")  # bert-base-uncased config for Q-Former

RUNNERS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RUNNERS)
import common  # noqa: E402

SALMONN_CODE = "eval/third_party/SALMONN"
CKPT_DEFAULT = "ckpt/baselines/salmonn"


class Model:
    def __init__(self, ckpt_dir=None, device="cuda:0", cpu_dtype="fp32", **kw):
        import contextlib
        import torch
        from omegaconf import OmegaConf

        sys.path.insert(0, SALMONN_CODE)
        from models.salmonn import SALMONN
        from transformers import WhisperFeatureExtractor

        d = ckpt_dir or CKPT_DEFAULT
        cfg = OmegaConf.load(os.path.join(SALMONN_CODE, "configs/decode_config.yaml"))
        cfg.model.llama_path = os.path.join(d, "vicuna-13b-v1.1")
        cfg.model.whisper_path = os.path.join(d, "whisper-large-v2")
        cfg.model.beats_path = os.path.join(d, "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
        cfg.model.ckpt = os.path.join(d, "salmonn_v1.pth")
        # decode_config uses repo-relative prompt paths (official code is run from the repo dir)
        for k in ("prompt_path", "test_prompt_path"):
            if cfg.model.get(k) and not os.path.isabs(cfg.model[k]):
                cfg.model[k] = os.path.join(SALMONN_CODE, cfg.model[k])
        self.cfg = cfg
        self.device = torch.device(device)
        self.torch = torch

        model = SALMONN.from_config(cfg.model)
        if self.device.type == "cpu":
            model = model.to(torch.bfloat16 if cpu_dtype == "bf16" else torch.float32)
            torch.set_num_threads(int(kw.get("threads", os.cpu_count() or 1)))
            self.autocast = contextlib.nullcontext
        else:
            self.autocast = lambda: torch.cuda.amp.autocast(dtype=torch.float16)
        model.to(self.device)
        model.eval()
        self.model = model
        self.cpu_dtype = next(model.parameters()).dtype
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.model.whisper_path)

    def _prepare(self, wav_path):
        # == utils.prepare_one_sample, plus resampling to 16 kHz (Whisper/BEATs rate)
        import numpy as np
        import soundfile as sf
        import librosa
        torch = self.torch
        audio, sr = sf.read(wav_path)
        if len(audio.shape) == 2:  # stereo to mono
            audio = audio[:, 0]
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        if len(audio) < sr:  # pad audio to at least 1s
            audio = np.concatenate((audio, np.zeros(sr - len(audio), dtype=float)), axis=0)
        audio = audio[: sr * 30]  # truncate audio to at most 30s (official)
        spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]
        samples = {
            "spectrogram": spectrogram,
            "raw_wav": torch.from_numpy(audio).unsqueeze(0),
            "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0),
        }
        out = {}
        for k, v in samples.items():
            v = v.to(self.device)
            if self.device.type == "cpu" and v.is_floating_point():
                v = v.to(self.cpu_dtype)
            elif v.is_floating_point():
                v = v.float()
            out[k] = v
        return out

    def generate(self, audio_path, prompt, gen):
        torch = self.torch
        samples = self._prepare(audio_path)
        prompts = [self.cfg.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())]
        gcfg = dict(self.cfg.generate)
        gcfg.update(num_beams=gen.get("num_beams", 4), do_sample=gen.get("do_sample", False),
                    max_new_tokens=gen.get("max_new_tokens", 100))
        with torch.inference_mode(), self.autocast():
            text = self.model.generate(samples, gcfg, prompts=prompts)[0]
        text = text.strip()
        for tok in ("<s>", "</s>"):
            text = text.replace(tok, "")
        return text.strip()


if __name__ == "__main__":
    common.main(Model, "salmonn")
