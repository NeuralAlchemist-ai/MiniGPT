import torch
import torch.nn as nn
from src.config import ModelConfig
from src.block import TransformerBlock
from src.embeddings import TokenEmbedding, LearnedPositionalEncoding


class MiniGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_emb = TokenEmbedding(config)
        self.pos_emb = LearnedPositionalEncoding(config)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.config = config
        self.norm = config.build_norm()
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        T = input_ids.shape[1]

        x = self.token_emb(input_ids)
        x = x + self.pos_emb(T, device=input_ids.device)

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

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
