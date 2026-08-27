import torch
import torch.nn as nn
from .transformer import TransformerBlock

class GPT(nn.Module):
    def __init__(self,
            vocab_size :int,
            d_model: int,
            n_layers: int,
            n_heads: int,
            max_seq_len: int,
        ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                max_seq_len,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_linear = nn.Linear(d_model, vocab_size)

        #build causal mask
        mask = torch.tril(
            torch.ones(
                size = (max_seq_len,max_seq_len),
                dtype = torch.int)
        )

        self.register_buffer(
            "causal_mask",
            mask.view(1,1,max_seq_len,max_seq_len)
        )

    def forward(self,x: torch.Tensor, pad_mask: torch.Tensor):
        B,T = x.shape
        assert T <= self.max_seq_len

        # build combined mask: causal mask [1,1,max_seq_len, max_seq_len] + pad_mask[B,T] -> [B,1,T,T]
        pad_mask = pad_mask.unsqueeze(-1)
        pad_mask = pad_mask @ pad_mask.transpose(-1,-2)
        combined_mask = self.causal_mask[:,:,:T,:T] * pad_mask.unsqueeze(1)

        # embedding
        pos_seq = torch.arange(
            T,
            dtype = torch.int64,
            device = x.device
        )
        x = self.token_embedding(x) + self.position_embedding(pos_seq)

        # transformer blocks
        for block in self.blocks:
            x = block(x, combined_mask)

        # forward
        x = self.norm(x)
        logits = self.lm_linear(x)

        return logits