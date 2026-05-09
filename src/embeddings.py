import torch
from torch import nn
from src.config import ModelConfig


class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_inds: torch.Tensor):
        return self.dropout(self.emb(input_inds))


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

    def forward(self, T: int, device: torch.device):
        pos = torch.arange(T, device=device).unsqueeze(0)
        return self.pos_emb(pos)
