"""
The models compared in the do-or-die experiment.  All map a window of the last K steps
of (state, executed action) to the expert's action at every position of the window.

  TinyTransformer  causal transformer, K tokens = one per time step; the only model that
                   can look at history, hence the only one that can identify the system.
  StateMLP         sees the current state only (no history): the "no identification" floor.
  OracleMLP        sees the current state *and the true hidden parameters*: the ceiling.

Kept deliberately small and plain PyTorch so every activation is easy to hook.
"""

from __future__ import annotations

import torch
import torch.nn as nn

TOKEN_DIM = 5  # x, x_dot, theta, theta_dot, executed action


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.last_attn = None  # (batch, heads, K, K) attention pattern of the last forward

    def forward(self, x, mask):
        h = self.ln1(x)
        a, w = self.attn(h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=False)
        self.last_attn = w.detach()
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, context: int = 32, d_model: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.context = context
        self.embed = nn.Linear(TOKEN_DIM, d_model)
        self.pos = nn.Parameter(torch.zeros(1, context, d_model))
        self.blocks = nn.ModuleList(Block(d_model, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer("mask", torch.triu(torch.ones(context, context, dtype=torch.bool), 1))

    def forward(self, tokens, return_residuals: bool = False):
        """tokens: (batch, K, 5) -> predicted expert action (batch, K).  With
        return_residuals also returns the residual stream after the embedding and after
        every block: list of (batch, K, d_model)."""
        K = tokens.shape[1]
        x = self.embed(tokens) + self.pos[:, :K]
        residuals = [x]
        for blk in self.blocks:
            x = blk(x, self.mask[:K, :K])
            residuals.append(x)
        out = self.head(self.ln_f(x)).squeeze(-1)
        return (out, residuals) if return_residuals else out

    def forward_patched(self, tokens, layer: int, delta):
        """Forward pass adding `delta` (d_model,) to the residual stream after `layer`
        (0 = after embedding) at every position."""
        K = tokens.shape[1]
        x = self.embed(tokens) + self.pos[:, :K]
        if layer == 0:
            x = x + delta
        for i, blk in enumerate(self.blocks, start=1):
            x = blk(x, self.mask[:K, :K])
            if i == layer:
                x = x + delta
        return self.head(self.ln_f(x)).squeeze(-1)


class StateMLP(nn.Module):
    """Current state only; applied independently at every position of the window."""

    def __init__(self, extra: int = 0, hidden: int = 128):
        super().__init__()
        self.extra = extra
        self.net = nn.Sequential(nn.Linear(4 + extra, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, tokens, hidden_params=None):
        x = tokens[..., :4]
        if self.extra:
            x = torch.cat([x, hidden_params[:, None, :].expand(-1, x.shape[1], -1)], dim=-1)
        return self.net(x).squeeze(-1)


def OracleMLP(n_hidden: int = 3) -> StateMLP:
    return StateMLP(extra=n_hidden)
