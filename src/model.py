import torch
import torch.nn as nn
from src.config import ModelConfig
from src.block import TransformerBlock
from src.embeddings import TokenEmbedding


class MiniGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_emb = TokenEmbedding(config)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.config = config
        self.head_size = config.head_size
        self.norm = config.build_norm()
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        cos_cache, sin_cache = self._build_rope_cache(
            config.max_seq_len, self.head_size
        )
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def _build_rope_cache(self, max_seq_len: int, head_size: int, theta: int = 10000):
        inv_freq = 1 / (theta ** (torch.arange(0, head_size, 2).float() / head_size))
        t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        return emb.cos(), emb.sin()

    def freq_rope(self, seq_len: int, device: torch.device):
        cos = self.cos_cache[:seq_len].to(device)
        sin = self.sin_cache[:seq_len].to(device)
        return cos, sin

    def forward(self, input_ids: torch.Tensor):
        T = input_ids.shape[1]

        x = self.token_emb(input_ids)
        x = self.dropout(x)

        cos, sin = self.freq_rope(T, x.device)

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ):

        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len :]

            logits = self.forward(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
