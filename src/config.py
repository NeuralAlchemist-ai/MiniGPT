from dataclasses import dataclass
from typing import Literal

import torch.nn as nn

@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    vocab_size: int = 49152
    max_seq_len: int = 512
    dropout: float = 0.1
    epsilon: float = 1e-8

    norm_type:   Literal['layernorm', 'rmsnorm']           = 'rmsnorm'
    pos_enc_type: Literal['sinusoidal', 'rope']            = 'sinusoidal'
    activation:  Literal['relu', 'gelu']                   = 'relu'

    @property
    def head_size(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        return self.d_model // self.n_heads
    
    def build_norm(self):
        if self.norm_type == 'layernorm':
            return nn.LayerNorm(self.d_model, eps=self.epsilon)
        elif self.norm_type == 'rmsnorm':
            return nn.RMSNorm([self.d_model], eps=self.epsilon)    
        raise ValueError(f"Unknown norm_type: {self.norm_type}")
    
    def build_pos_enc_type(self):
        """Currently only 'sinusoidal' positional encoding is implemented via LearnedPositionalEncoding.
        This method is reserved for future RoPE or other advanced positional encoding schemes."""
        if self.pos_enc_type == 'sinusoidal':
            # Currently using LearnedPositionalEncoding in embeddings.py
            return None
        raise ValueError(f"Unknown pos_enc_type: {self.pos_enc_type}. Only 'sinusoidal' is supported.")

    def build_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'gelu':
            return nn.GELU()
        raise ValueError(f"Unknown activation: {self.activation}")
    