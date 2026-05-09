import torch
import torch.nn as nn
from src.config import ModelConfig


class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(input_ids)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(T, device=device)
        return self.pos_emb(pos)
