import torch
import torch.nn as nn
from .self_attention import SelfAttention

# transformer:
class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio = 4):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model)
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model, n_heads, max_seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x, pad_mask, is_prefill=False, is_generate=False):
        x = x + self.attention(self.norm1(x), pad_mask, is_prefill, is_generate)
        x = x + self.ffn(self.norm2(x))
        return x