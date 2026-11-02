"""AudioGround: SALMONN + frame-level interpolation + time-aware (timestamp-conditioned) sliding-window
Q-Former + absolute time embedding.

Module names (speech_Qformer, speech_query_tokens, speech_llama_proj, ln_speech, ln_audio, beats, llama_model)
follow SALMONN so that SALMONN checkpoints load directly.
"""
import logging
import math
import os
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer, WhisperModel, BertTokenizerFast

from .Qformer import BertConfig, BertLMHeadModel
from .beats.BEATs import BEATs, BEATsConfig
from .time_embedding import AbsoluteTimeEmbedding
from . import llama_sdpa
from .audio import FPS, WHISPER_FRAMES

log = logging.getLogger("audioground")


def _load_state_dict_any(path):
    """Loads a HF model dir (safetensors or bin) or a single file into a flat state dict (CPU)."""
    if os.path.isdir(path):
        st = os.path.join(path, "model.safetensors")
        if os.path.exists(st):
            from safetensors.torch import load_file
            return load_file(st)
        return torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    return torch.load(path, map_location="cpu")


class AudioGround(nn.Module):
    def __init__(self, cfg, compute_dtype=torch.float32):
        super().__init__()
        m = cfg
        self.cfg = m
        self.compute_dtype = compute_dtype

        # ------------------------------------------------------------- LLM
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(m.llama_path, use_fast=False)
        self.llama_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llama_tokenizer.padding_side = "right"
        llm = LlamaForCausalLM.from_pretrained(m.llama_path, torch_dtype=compute_dtype)
        if llm.get_input_embeddings().weight.shape[0] < len(self.llama_tokenizer):
            llm.resize_token_embeddings(len(self.llama_tokenizer))  # [PAD] row; always masked out
        for p in llm.parameters():
            p.requires_grad = False
        salmonn_sd = torch.load(m.salmonn_ckpt, map_location="cpu")["model"] if m.salmonn_ckpt else {}
        if salmonn_sd and m.salmonn_lora == "merge":
            self._merge_salmonn_lora(llm, salmonn_sd, m.salmonn_lora_alpha)
        if m.gradient_checkpointing:
            llm.gradient_checkpointing_enable()
            llm.config.use_cache = False
        if m.lora:
            from peft import LoraConfig, TaskType, get_peft_model
            pc = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=m.lora_rank,
                            lora_alpha=m.lora_alpha, lora_dropout=m.lora_dropout,
                            target_modules=list(m.lora_target_modules))
            llm = get_peft_model(llm, pc)
        self.llama_model = llm
        self.d_llm = llm.config.hidden_size

        # ------------------------------------------------------------- encoders (frozen)
        self.speech_encoder = WhisperModel.from_pretrained(m.whisper_path, torch_dtype=compute_dtype).encoder
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
        enc_width = self.speech_encoder.config.d_model
        self.use_beats = bool(m.use_beats and m.beats_path)
        if self.use_beats:
            bck = torch.load(m.beats_path, map_location="cpu")
            self.beats = BEATs(BEATsConfig(bck["cfg"]))
            self.beats.load_state_dict(bck["model"])
            # BEATs stays fp32 (weight_norm has no bf16 kernel in torch 2.0); it runs under autocast like SALMONN
            self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
            enc_width += self.beats.cfg.encoder_embed_dim
        for mod in [self.speech_encoder] + ([self.beats] if self.use_beats else []):
            for p in mod.parameters():
                p.requires_grad = False
            mod.eval()

        # ------------------------------------------------------------- window-level Q-Former
        qc = BertConfig.from_pretrained(m.bert_path)
        qc.num_hidden_layers = m.qformer_layers
        qc.encoder_width = enc_width
        qc.add_cross_attention = True
        qc.cross_attention_freq = 1
        qc.query_length = m.num_query_token
        self.speech_Qformer = BertLMHeadModel(config=qc)
        self.speech_Qformer.cls = None
        self.speech_query_tokens = nn.Parameter(torch.zeros(1, m.num_query_token, qc.hidden_size))
        self.speech_query_tokens.data.normal_(mean=0.0, std=qc.initializer_range)
        self.timestamp_cond = bool(m.timestamp_cond)
        if self.timestamp_cond:
            self.bert_tokenizer = BertTokenizerFast.from_pretrained(m.bert_path)
            if m.qformer_text_init == "bert":
                self._init_qformer_text_from_bert(m.bert_path)
        else:  # vanilla SALMONN: drop text pathway
            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        self.speech_llama_proj = nn.Linear(qc.hidden_size, self.d_llm)

        self.win_frames = max(1, int(round(m.window_sec * FPS)))     # 0.333 s -> 17 frames (as SALMONN)
        self.stride_frames = max(1, int(round(m.stride_sec * FPS)))
        self._ts_cache = {}

        # ------------------------------------------------------------- absolute time embedding
        self.ate = AbsoluteTimeEmbedding(m.ate, self.d_llm, beta_init=m.ate_beta_init,
                                         beta_learnable=m.ate_beta_learnable, time_scale=m.ate_time_scale,
                                         learned_hidden=m.ate_learned_hidden,
                                         learned_t_scale=m.ate_learned_t_scale)

        # ------------------------------------------------------------- E6b auxiliary span head
        self.aux_span_weight = float(m.aux_span_weight)
        if self.aux_span_weight > 0:
            self.span_head = nn.Sequential(nn.Linear(self.d_llm, 1024), nn.GELU(), nn.Linear(1024, 2))

        # ------------------------------------------------------------- SALMONN init
        if salmonn_sd:
            self._load_salmonn(salmonn_sd)
        prompt = m.prompt_template
        assert "<SpeechHere>" in prompt and "{}" in prompt

    # ================================================================= init helpers
    @staticmethod
    def _merge_salmonn_lora(llm, sd, alpha):
        n = 0
        mods = dict(llm.named_modules())
        for k, a in sd.items():
            if ".lora_A." not in k:
                continue
            b = sd[k.replace(".lora_A.", ".lora_B.")]
            name = k.split("base_model.model.", 1)[1].split(".lora_A.")[0]  # model.layers.0.self_attn.q_proj
            lin = mods[name]
            r = a.shape[0]
            delta = (b.float() @ a.float()) * (alpha / r)
            lin.weight.data = (lin.weight.data.float() + delta).to(lin.weight.dtype)
            n += 1
        log.info(f"merged {n} SALMONN LoRA matrices into LLM (alpha={alpha})")

    def _init_qformer_text_from_bert(self, bert_path):
        sd = _load_state_dict_any(bert_path)
        sd = {k[5:] if k.startswith("bert.") else k: v for k, v in sd.items()}
        sd = {k.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias"): v
              for k, v in sd.items()}  # legacy TF naming in google-bert/bert-base-uncased
        bert = self.speech_Qformer.bert
        tgt = {}
        for k in ["embeddings.word_embeddings.weight", "embeddings.position_embeddings.weight"]:
            tgt[k] = sd[k]
        for i in range(len(bert.encoder.layer)):
            for sub in ["intermediate.dense.weight", "intermediate.dense.bias", "output.dense.weight",
                        "output.dense.bias", "output.LayerNorm.weight", "output.LayerNorm.bias"]:
                tgt[f"encoder.layer.{i}.{sub}"] = sd[f"encoder.layer.{i}.{sub}"]
        missing, unexpected = bert.load_state_dict(tgt, strict=False)
        assert not unexpected, unexpected
        log.info(f"Q-Former text pathway initialised from {bert_path} ({len(tgt)} tensors)")

    def _load_salmonn(self, sd):
        sd = {k: v for k, v in sd.items() if not k.startswith("llama_model.")}
        q = sd.pop("speech_query_tokens", None)
        if q is not None:
            nq = self.speech_query_tokens.shape[1]
            if q.shape[1] == nq:
                self.speech_query_tokens.data.copy_(q)
            elif self.cfg.query_init_multi == "repeat":
                rep = q.repeat(1, math.ceil(nq / q.shape[1]), 1)[:, :nq]
                self.speech_query_tokens.data.copy_(rep + 0.02 * torch.randn_like(rep))
        if sd.get("speech_llama_proj.weight") is not None and sd["speech_llama_proj.weight"].shape[0] != self.d_llm:
            raise ValueError("SALMONN ckpt LLM width does not match llama_path (7B vs 13B mix-up?)")
        own = self.state_dict()
        sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        log.info(f"loaded {len(sd)} SALMONN tensors (non-LLM)")

    # ================================================================= utils
    def train(self, mode=True):
        super().train(mode)
        self.speech_encoder.eval()
        if self.use_beats:
            self.beats.eval()
        return self

    @property
    def device(self):
        return self.speech_query_tokens.device

    def autocast(self):
        if self.device.type == "cuda" and self.compute_dtype != torch.float32:
            return torch.autocast(device_type="cuda", dtype=self.compute_dtype)
        return contextlib.nullcontext()

    def embed_tokens(self, ids):
        return self.llama_model.get_input_embeddings()(ids)

    def trainable_state_dict(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        sd = self.state_dict()
        return {k: v.detach().cpu() for k, v in sd.items() if k in keep or k == "ate.beta"}

    # ================================================================= audio encoding
    @torch.no_grad()
    def _encode_chunks(self, mel, wav, wav_pad):
        """Frozen encoders on 30 s chunks -> raw (pre-LN) Whisper (N,1500,1280) and BEATs (N,~1496,768) features."""
        bs = self.cfg.encoder_batch_size
        sp, au = [], []
        for i in range(0, mel.shape[0], bs):
            s = self.speech_encoder(mel[i:i + bs].to(self.device, self.compute_dtype), return_dict=True).last_hidden_state
            sp.append(s)
            if self.use_beats:
                a, _ = self.beats.extract_features(wav[i:i + bs].to(self.device), padding_mask=wav_pad[i:i + bs].to(self.device),
                                                   feature_only=True)
                au.append(a)
        # round to the compute dtype (bf16 on GPU): this is exactly what the offline feature cache stores
        sp = torch.cat(sp).to(self.compute_dtype)
        au = torch.cat(au).to(self.compute_dtype) if au else None
        return sp, au

    def encode_audio(self, batch):
        """Returns window tokens per sample: list of (L_b*N_Q, D) and list of L_b."""
        with self.autocast():
            if "feat_w" in batch:    # offline feature cache (identical bf16 encoder outputs)
                sp = batch["feat_w"].to(self.device)
                au = batch["feat_b"].to(self.device) if (self.use_beats and "feat_b" in batch) else None
            else:
                sp, au = self._encode_chunks(batch["mel"], batch["wav"], batch["wav_pad"])
            T = sp.shape[1]
            feats = self.ln_speech(sp.float())
            if au is not None:
                au = self.ln_audio(au.float())
                if self.cfg.beats_align == "interp":      # Frame-level interpolation (paper)
                    au = F.interpolate(au.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)
                else:                                     # vanilla SALMONN zero pad / truncate
                    au = F.pad(au, (0, 0, 0, max(0, T - au.shape[1])))[:, :T]
                feats = torch.cat([feats, au], dim=-1)
            Z, n_frames = self._assemble(feats, batch)
            return self._qformer_windows(Z, n_frames)

    def _assemble(self, feats, batch):
        B = int(batch["n_frames"].shape[0])
        n_frames = batch["n_frames"].tolist()
        Tmax = max(n_frames)
        C = feats.shape[-1]
        Z = feats.new_zeros(B, Tmax, C)
        W = feats.new_zeros(B, Tmax, 1)
        mode = self.cfg.overlap_merge
        for c in range(feats.shape[0]):
            b = int(batch["chunk_sample"][c]); s = int(batch["chunk_start"][c]); k = int(batch["chunk_keep"][c])
            k = min(k, Tmax - s, feats.shape[1])
            if mode == "center":
                i = torch.arange(k, device=feats.device, dtype=feats.dtype)
                w = torch.minimum(i + 1, k - i).clamp(min=1.0).unsqueeze(-1)
            elif mode == "first":
                w = (W[b, s:s + k] == 0).to(feats.dtype)
            else:
                w = torch.ones(k, 1, device=feats.device, dtype=feats.dtype)
            Z[b, s:s + k] = Z[b, s:s + k] + feats[c, :k] * w
            W[b, s:s + k] = W[b, s:s + k] + w
        Z = Z / W.clamp(min=1e-6)
        return Z, n_frames

    def _timestamp_ids(self, L):
        """Token ids of 'This segment is from {s} to {e} seconds.' for windows 0..L-1 (cached)."""
        key = L
        if key not in self._ts_cache:
            m = self.cfg
            texts = []
            for l in range(L):
                s, e, _ = self._window_time(l)
                texts.append(m.timestamp_template.format(s=f"{s:.{m.timestamp_decimals}f}", e=f"{e:.{m.timestamp_decimals}f}"))
            tok = self.bert_tokenizer(texts, padding="longest", return_tensors="pt", add_special_tokens=True)
            if len(self._ts_cache) > 64:
                self._ts_cache.clear()
            self._ts_cache[key] = (tok.input_ids, tok.attention_mask)
        return self._ts_cache[key]

    def _window_time(self, l):
        """(start, end, centre) in seconds of window l (0-indexed)."""
        m = self.cfg
        if m.timestamp_source == "nominal":     # paper formula: s_l = (l-1) S_sec, e_l = s_l + W_sec
            s = l * m.stride_sec; w = m.window_sec
        else:                                   # exact frame times of the unfolded window
            s = l * self.stride_frames / FPS; w = self.win_frames / FPS
        return s, s + w, s + w / 2.0

    def _qformer_windows(self, Z, n_frames):
        K, S = self.win_frames, self.stride_frames
        wins, Ls = [], []
        for b, T in enumerate(n_frames):
            z = Z[b, :T]
            if T < K:
                z = F.pad(z, (0, 0, 0, K - T))
            w = z.unfold(0, K, S).permute(0, 2, 1)          # (L, K, C)
            wins.append(w); Ls.append(w.shape[0])
        win = torch.cat(wins)                               # (sum L, K, C)
        N = win.shape[0]
        q = self.speech_query_tokens.expand(N, -1, -1)
        nq = q.shape[1]
        enc_att = torch.ones(win.shape[:-1], dtype=torch.long, device=win.device)
        if self.timestamp_cond:
            ids, att = self._timestamp_ids(max(Ls))
            idx = torch.cat([torch.arange(L) for L in Ls])
            ids, att = ids[idx].to(win.device), att[idx].to(win.device)
            full_att = torch.cat([torch.ones(N, nq, dtype=att.dtype, device=win.device), att], dim=1)
            out = self.speech_Qformer.bert(input_ids=ids, attention_mask=full_att, query_embeds=q,
                                           encoder_hidden_states=win, encoder_attention_mask=enc_att, return_dict=True)
        else:
            out = self.speech_Qformer.bert(query_embeds=q, encoder_hidden_states=win,
                                           encoder_attention_mask=enc_att, return_dict=True)
        h = self.speech_llama_proj(out.last_hidden_state[:, :nq])   # (N, N_Q, D)
        res, o = [], 0
        for L in Ls:
            hb = h[o:o + L].unsqueeze(0); o += L                       # (1, L, N_Q, D)
            centers = torch.tensor([self._window_time(l)[2] for l in range(L)], device=h.device).unsqueeze(0)
            hb = self.ate(hb, centers)                                  # H' = H + beta * TE(c)
            res.append(hb.reshape(L * nq, -1))
        return res, Ls

    # ================================================================= LLM input building
    def _prompt_parts(self, instruction):
        inst = instruction if "<SpeechHere>" in instruction else self.cfg.prompt_template.format(instruction)
        left, right = inst.split("<SpeechHere>", 1)
        return left, right

    def _tok(self, text):
        return self.llama_tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]

    def build_inputs(self, audio_tokens, instructions, responses=None, left_pad=False):
        dev = self.device
        tok = self.llama_tokenizer
        seqs, labels, prompt_end = [], [], []
        for i, at in enumerate(audio_tokens):
            left, right = self._prompt_parts(instructions[i])
            pre_ids = torch.cat([torch.tensor([tok.bos_token_id]), self._tok(left)]).to(dev)
            post_ids = self._tok(right).to(dev)
            parts = [self.embed_tokens(pre_ids), at.to(self.embed_tokens(pre_ids).dtype), self.embed_tokens(post_ids)]
            lab = [torch.full((len(pre_ids) + at.shape[0] + len(post_ids),), -100, dtype=torch.long, device=dev)]
            if responses is not None:
                r_ids = self._tok(responses[i] + self.cfg.end_sym)[: self.cfg.max_txt_len].to(dev)
                parts.append(self.embed_tokens(r_ids)); lab.append(r_ids)
            prompt_end.append(len(pre_ids) + at.shape[0] + len(post_ids) - 1)
            seqs.append(torch.cat(parts)); labels.append(torch.cat(lab))
        Lmax = max(s.shape[0] for s in seqs)
        D = seqs[0].shape[-1]
        emb = seqs[0].new_zeros(len(seqs), Lmax, D)
        att = torch.zeros(len(seqs), Lmax, dtype=torch.long, device=dev)
        lab = torch.full((len(seqs), Lmax), -100, dtype=torch.long, device=dev)
        pad_emb = self.embed_tokens(torch.tensor([tok.pad_token_id], device=dev))[0]
        for i, s in enumerate(seqs):
            n = s.shape[0]
            if left_pad:
                emb[i, Lmax - n:] = s; att[i, Lmax - n:] = 1; lab[i, Lmax - n:] = labels[i]
                emb[i, :Lmax - n] = pad_emb.to(emb.dtype)
                prompt_end[i] += Lmax - n
            else:
                emb[i, :n] = s; att[i, :n] = 1; lab[i, :n] = labels[i]
                emb[i, n:] = pad_emb.to(emb.dtype)
        return emb, att, lab, torch.tensor(prompt_end, device=dev)

    # ================================================================= forward / generate
    def forward(self, batch):
        audio_tokens, _ = self.encode_audio(batch)
        emb, att, lab, prompt_end = self.build_inputs(audio_tokens, batch["instruction"], batch["response"])
        with self.autocast(), llama_sdpa.causal_right_padded(self.cfg.get("llm_sdpa", True)):  # batch is right-padded
            out = self.llama_model(inputs_embeds=emb, attention_mask=att, return_dict=True,
                                   output_hidden_states=self.aux_span_weight > 0)
        logits = out.logits[:, :-1].float()
        tgt = lab[:, 1:]
        tok_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100,
                                   reduction="none").view(tgt.shape)
        mask = (tgt != -100).float()
        per_sample = (tok_loss * mask).sum(1) / mask.sum(1).clamp(min=1)
        w = batch.get("loss_weight")
        if w is not None:
            w = w.to(per_sample.device).float()
            lm_loss = (per_sample * w).sum() / w.sum().clamp(min=1e-6)
        else:
            lm_loss = per_sample.mean()
        res = {"lm_loss": lm_loss.detach(), "n_tokens": mask.sum().detach()}
        loss = lm_loss
        if self.aux_span_weight > 0:
            hid = out.hidden_states[-1][torch.arange(emb.shape[0], device=emb.device), prompt_end].float()
            span_loss = self._span_loss(hid, batch)
            loss = loss + self.aux_span_weight * span_loss
            res["span_loss"] = span_loss.detach()
        res["loss"] = loss
        return res

    def _span_loss(self, hid, batch):
        gt = batch["span_norm"].to(hid.device).float()       # (B, 2) in [0,1], NaN if no span
        valid = ~torch.isnan(gt[:, 0])
        if valid.sum() == 0:
            return hid.sum() * 0.0
        # NaN targets must be removed BEFORE the loss math: masking afterwards (loss[valid]) still back-propagates
        # 0 * NaN = NaN through the invalid rows (this NaN'd every parameter in the first E6b/E6_all runs).
        hid, gt = hid[valid], gt[valid]
        pr = torch.sigmoid(self.span_head(hid.to(self.span_head[0].weight.dtype)).float())
        c, w = pr[:, 0], pr[:, 1]
        ps, pe = (c - w / 2).clamp(0, 1), (c + w / 2).clamp(0, 1)
        gs, ge = gt[:, 0], gt[:, 1]
        l1 = (ps - gs).abs() + (pe - ge).abs()
        inter = (torch.minimum(pe, ge) - torch.maximum(ps, gs)).clamp(min=0)
        union = (pe - ps) + (ge - gs) - inter
        enclose = torch.maximum(pe, ge) - torch.minimum(ps, gs)
        giou = inter / union.clamp(min=1e-6) - (enclose - union) / enclose.clamp(min=1e-6)
        loss = self.cfg.aux_span_l1_weight * l1 + self.cfg.aux_span_giou_weight * (1 - giou)
        return loss.mean()

    @torch.no_grad()
    def generate(self, batch, gen_cfg):
        audio_tokens, _ = self.encode_audio(batch)
        emb, att, _, _ = self.build_inputs(audio_tokens, batch["instruction"], None, left_pad=True)
        tok = self.llama_tokenizer
        with self.autocast():
            out = self.llama_model.generate(
                inputs_embeds=emb, attention_mask=att,
                max_new_tokens=gen_cfg.max_new_tokens, num_beams=gen_cfg.num_beams, do_sample=gen_cfg.do_sample,
                min_length=gen_cfg.min_length, repetition_penalty=gen_cfg.repetition_penalty,
                length_penalty=gen_cfg.length_penalty, eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id)
        return [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]
