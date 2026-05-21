from torch import nn
from src.config import ModelConfig
import torch
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.head_size = config.head_size
        self.n_heads = config.n_heads

        self.qkv = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj = nn.Linear(self.d_model, self.d_model)
        self.proj_dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
            ),
        )

    def apply_rope(self, x, cos, sin):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]

        x_rotated = torch.cat((-x2, x1), dim=-1)

        if cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(1)
            sin = sin.unsqueeze(0).unsqueeze(1)
        if cos.ndim == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        return (x * cos) + (x_rotated * sin)

    def forward(self, x, cos, sin):
        B, T, C = x.shape

        qkv = self.qkv(x)

        Q,K,V = qkv.split(self.d_model, dim=-1)

        Q = Q.view(B, T, self.n_heads, self.head_size).transpose(1,2)
        K = K.view(B, T, self.n_heads, self.head_size).transpose(1,2)
        V = V.view(B, T, self.n_heads, self.head_size).transpose(1,2)

        Q = self.apply_rope(Q, cos, sin)
        K = self.apply_rope(K, cos, sin)

        wei = Q @ K.transpose(-2, -1)# * (self.head_size**-0.5)
        wei = wei.masked_fill(~self.tril[:T, :T], float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ V

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out

