import torch.nn as nn

from src.config import ModelConfig
from src.attention import MultiHeadAttention
from src.feedforward import FeedForwardNetwork


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mha = MultiHeadAttention(config)
        self.ffn = FeedForwardNetwork(config)

        self.norm1 = config.build_norm()
        self.norm2 = config.build_norm()

        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x, cos, sin):
        x = x + self.residual_dropout(self.mha(self.norm1(x), cos, sin))
        x = x + self.residual_dropout(self.ffn(self.norm2(x)))

        return x
