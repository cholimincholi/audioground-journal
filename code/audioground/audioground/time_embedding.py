"""Absolute time embedding (ATE):  H'_l = H_l + beta * TE(c_l),  c_l = window centre (seconds)."""
import math
import torch
import torch.nn as nn


def sinusoidal(t, dim):
    """TE_sin(c)_{2k} = sin(w_k c), TE_sin(c)_{2k+1} = cos(w_k c), w_k = 10000^{-2k/d}. t: (...,) -> (..., dim)."""
    k = torch.arange(dim // 2, device=t.device, dtype=torch.float32)
    w = torch.exp(-math.log(10000.0) * 2.0 * k / dim)
    ang = t.float().unsqueeze(-1) * w
    te = torch.zeros(*t.shape, dim, device=t.device, dtype=torch.float32)
    te[..., 0::2] = torch.sin(ang)
    te[..., 1::2] = torch.cos(ang)
    return te


class AbsoluteTimeEmbedding(nn.Module):
    VARIANTS = ("none", "sin", "learned", "hybrid")

    def __init__(self, variant, dim, beta_init=0.05, beta_learnable=True, time_scale=1.0,
                 learned_hidden=0, learned_t_scale=0.01):
        super().__init__()
        assert variant in self.VARIANTS, variant
        self.variant, self.dim, self.time_scale = variant, dim, time_scale
        self.learned_t_scale = learned_t_scale
        if variant == "none":
            return
        beta = torch.tensor(float(beta_init))
        if beta_learnable:
            self.beta = nn.Parameter(beta)
        else:
            self.register_buffer("beta", beta)
        if variant == "learned":           # ATE_l(t) = MLP(t), scalar input
            h = learned_hidden or dim
            self.mlp = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, dim))
        elif variant == "hybrid":          # TE_sin + W2 SiLU(W1 TE_sin + b1) + b2 ; W2, b2 zero-init
            self.fc1 = nn.Linear(dim, dim)
            self.act = nn.SiLU()
            self.fc2 = nn.Linear(dim, dim)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def embed(self, t):
        """t: (...,) window centres in seconds -> (..., dim) fp32 (beta not applied)."""
        t = t.float() * self.time_scale
        if self.variant == "sin":
            return sinusoidal(t, self.dim)
        if self.variant == "learned":
            return self.mlp((t * self.learned_t_scale).unsqueeze(-1).to(self.mlp[0].weight.dtype)).float()
        if self.variant == "hybrid":
            base = sinusoidal(t, self.dim)
            x = base.to(self.fc1.weight.dtype)
            return base + self.fc2(self.act(self.fc1(x))).float()
        raise ValueError(self.variant)

    def forward(self, h, centers):
        """h: (B, L, N_Q, D) projected window tokens; centers: (B, L). TE shared across the N_Q queries."""
        if self.variant == "none":
            return h
        te = self.embed(centers)                                  # (B, L, D)
        return h + (self.beta.float() * te).unsqueeze(2).to(h.dtype)
