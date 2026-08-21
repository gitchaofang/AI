import torch
import torch.nn as nn
from .SelfAttention import SelfAttention

class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4):
        super().__init__()

        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, mask: torch.Tensor):
        super().__init__()

        self.norm1 == nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model, n_heads, max_seq_len, mask)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x 
