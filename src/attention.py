from torch import nn
import torch
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.key = nn.Linear(config.d_model, config.head_size, bias=False)
        self.query = nn.Linear(config.d_model, config.head_size, bias=False)
        self.value = nn.Linear(config.d_model, config.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B,T,C = x.shape
        head_size = K.shape[-1]

        K = self.key(x)   # (B,T,hs) 
        Q = self.query(x) # (B,T,hs)
        V = self.value(x) # (B,T,hs)

        # compute attention scores
        wei = Q @ K.transpose(-2,-1) * head_size**-0.5 # (B,T,hs) @ (B,hs,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)

        # perform the weighted aggregation of the values
        out = wei @ V # (B,T,T) @ (B,T,hs) -> (B,T,hs)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out