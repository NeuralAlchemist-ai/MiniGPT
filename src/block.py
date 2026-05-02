import torch.nn as nn

from config import ModelConfig
from attention import MultiHeadAttention
from feedforward import FeedForwardNetwork

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mha = MultiHeadAttention(config)
        self.ffn = FeedForwardNetwork(config)

        self.norm1 = config.build_norm()
        self.norm2 = config.build_norm()

    def forward(self, x):
        x = x + self.mha(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x