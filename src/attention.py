from torch import nn
from src.config import ModelConfig
import torch
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.key = nn.Linear(config.d_model, config.head_size, bias=False)
        self.query = nn.Linear(config.d_model, config.head_size, bias=False)
        self.value = nn.Linear(config.d_model, config.head_size, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)),
        )

    def apply_rope(self, x, cos, sin):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        x_rotated = torch.cat((-x2, x1), dim=-1)

        if cos.ndim == 2:
            leading = (1,) * (x.ndim - 2)
            cos = cos.view(*leading, *cos.shape)
            sin = sin.view(*leading, *sin.shape)

        return (x * cos) + (x_rotated * sin)

    def forward(self, x, cos, sin):
        B, T, C = x.shape

        K = self.key(x)  # (B,T,hs)
        Q = self.query(x)  # (B,T,hs)
        V = self.value(x)  # (B,T,hs)

        Q = self.apply_rope(Q, cos, sin)
        K = self.apply_rope(K, cos, sin)

        head_size = K.shape[-1]

        wei = Q @ K.transpose(-2, -1) * head_size**-0.5
        wei = wei.masked_fill(~self.tril[:T, :T], float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ V
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.head_size = config.head_size
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.proj_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin):
        out = torch.cat([h(x, cos, sin) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out
