import torch
import pytest
from src.attention import MultiHeadAttention
from src.config import ModelConfig


def build_rope_cache(seq_len: int, head_size: int):
    inv_freq = 1 / (10000 ** (torch.arange(0, head_size, 2).float() / head_size))
    t = torch.arange(seq_len, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)
    return emb.cos(), emb.sin()


@pytest.mark.parametrize("batch, seq_len", [(1, 8), (2, 16)])
def test_multihead_attention_shape_and_finiteness(batch: int, seq_len: int):
    config = ModelConfig(d_model=64, n_heads=8, max_seq_len=32)
    mha = MultiHeadAttention(config)

    x = torch.randn(batch, seq_len, config.d_model) * 50.0
    cos, sin = build_rope_cache(seq_len, config.head_size)

    output = mha(x, cos, sin)

    assert output.shape == (batch, seq_len, config.d_model)
    assert torch.isfinite(output).all()
