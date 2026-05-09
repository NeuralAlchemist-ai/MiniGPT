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
        self.register_buffer('tril', torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)))

    def forward(self, x):
        B,T,C = x.shape

        K = self.key(x)   # (B,T,hs) 
        Q = self.query(x) # (B,T,hs)
        V = self.value(x) # (B,T,hs)

        head_size = K.shape[-1]

        # compute attention scores
        wei = Q @ K.transpose(-2,-1) * head_size**-0.5 # (B,T,hs) @ (B,hs,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.attn_dropout(wei)

        # perform the weighted aggregation of the values
        out = wei @ V # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.proj_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out