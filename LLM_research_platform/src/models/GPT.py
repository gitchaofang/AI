import torch
import torch.nn as nn
from .transformer import TransformerBlock

class GPT(nn.Module):
    def __init__(self,
                 verb_size: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 max_seq_len: int,
                 ):
        super().__init__()
        self.token_embedding = nn.embedding(verb_size,d_model)
        self.position_embedding = nn.embedding(max_seq_len, d_model)
        self.mask = torch.tril(torch.ones(size = (max_seq_len, max_seq_len)))
        self.max_seq_len = max_seq_len
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                max_seq_len,
                self.mask,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            verb_size,
            bias=False,
        )

    def forward(self, input_ids):
        B, T = input_ids.shape
        assert T <= self.max_seq_len

        positions = torch.arange(
            T,
            device = input_ids.device,
        )

        x = self.token_embedding(input_ids) + self.position_embedding(self.positions)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_hea(x)

        return logits