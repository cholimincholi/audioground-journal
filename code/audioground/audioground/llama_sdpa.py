"""Fused attention (torch SDPA, is_causal) for LLaMA in transformers 4.28, used only inside
`causal_right_padded()`: training / validation forwards whose batches are RIGHT-padded. There, plain causal
attention is exact for every non-pad token (pads come after all real tokens and carry label -100), so the padding mask
can be dropped and the flash / memory-efficient kernels used. Generation (left padding, KV cache) keeps the original
eager implementation.
"""
import contextlib

import torch.nn.functional as F
from transformers.models.llama import modeling_llama as ML

_ORIG = ML.LlamaAttention.forward
_STATE = {"on": False}


def _forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
             output_attentions=False, use_cache=False):
    if not _STATE["on"] or past_key_value is not None or output_attentions:
        return _ORIG(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache)
    bsz, q_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    cos, sin = self.rotary_emb(v, seq_len=q_len)
    q, k = ML.apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), is_causal=True)
    out = self.o_proj(out.transpose(1, 2).reshape(bsz, q_len, self.hidden_size))
    return out, None, ((k, v) if use_cache else None)


ML.LlamaAttention.forward = _forward


@contextlib.contextmanager
def causal_right_padded(enabled=True):
    prev = _STATE["on"]
    _STATE["on"] = bool(enabled)
    try:
        yield
    finally:
        _STATE["on"] = prev
