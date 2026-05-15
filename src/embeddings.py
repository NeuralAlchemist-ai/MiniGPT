import torch
import torch.nn as nn
from src.config import ModelConfig


class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, input_ids: torch.Tensor):
        return self.emb(input_ids)
