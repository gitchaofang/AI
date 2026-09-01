import torch
import torch.nn as nn
from .transformer import TransformerBlock

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size + 1 ,d_model,)
        self.position_embedding = nn.Embedding(max_seq_len,d_model,)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model,)
        self.lm_linear = nn.Linear(d_model,vocab_size + 1,)
        self.position_offset = None

    def forward(
        self,
        x,
        pad_mask,
        is_prefill=False,
        is_generate=False,
    ):
        B, T_q = x.shape

        assert T_q <= self.max_seq_len

        assert not (
            is_prefill and is_generate
        ), "is_prefill and is_generate cannot both be True"

        if not is_prefill and not is_generate:
            position_ids = torch.arange(T_q,device=x.device,).unsqueeze(0).expand(B,T_q,)
        elif is_prefill:
            position_ids = torch.arange(T_q,device=x.device,).unsqueeze(0).expand(B,T_q,)
            self.position_offset = (pad_mask.to(torch.int64).sum(dim=1))
        else:
            assert is_generate
            assert self.position_offset is not None, (
                "Generation requires a previous prefill"
            )
            relative_positions = torch.arange(T_q,device=x.device,).unsqueeze(0)
            position_ids = (self.position_offset.unsqueeze(1)+ relative_positions)

        assert torch.all(position_ids >= 0)
        assert torch.all(position_ids < self.max_seq_len), ("Position ID exceeds max_seq_len")

        x = (self.token_embedding(x)+ self.position_embedding(position_ids))
        for block in self.blocks:
            x = block(
                x,
                pad_mask,
                is_prefill=is_prefill,
                is_generate=is_generate,
            )
        x = self.norm(x)
        logits = self.lm_linear(x)
        if is_generate:
            self.position_offset += (pad_mask.to(torch.int64).sum(dim=1))

        return logits