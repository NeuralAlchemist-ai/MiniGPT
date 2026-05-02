from dataclasses import dataclass
from typing import Literal

import torch.nn as nn

@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    head_size: int = 32
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
    def head_size(self) -> int:
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        return self.d_model // self.n_heads
    
    def build_norm(self):
        if self.norm_type == 'layernorm':
            return nn.LayerNorm()
        elif self.norm_type == 'rmsnorm':
            return nn.RMSNorm()    
        raise ValueError(f"Unknown norm_type: {self.norm_type}")
    
    def build_pos_enc_type(self):
        if self.pos_enc_type == 'sinusoidal':
            #TODO Sinusoidal
            pass
        elif self.pos_enc_type == 'rope':
            #TODO RoPE
            pass
        raise ValueError(f"Unknown pos_enc_type: {self.pos_enc_type}")

    def build_activation(self):
        if self.activation == 'relu':
            return nn.relu
        elif self.activation == 'gelu':
            return nn.gelu
        raise ValueError(f"Unknown activation: {self.activation}")
    