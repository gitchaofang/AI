import torch
import torch.nn as nn
import math
import numpy as np
#B batch sizae
#T token serial size
#D hidden layer dimension
#H head number
#Dh header dimension (multi header)

# self attention
class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, pad_token=0):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        mask = torch.tril(
            torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.int64)
        )
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, self.max_seq_len, self.max_seq_len),
        )

        self.kv_cache = None
        self.kv_mask = None
        self.pad_token = pad_token

    def reset_cache(self):
        self.kv_cache = None
        self.kv_mask = None

    def _dot_product(self, q, k, v, combined_mask):
        B, T_q, D = q.shape
        T_k = k.shape[1]

        q = q.view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-1, -2)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(combined_mask == 0, -1e10)

        attention = torch.softmax(scores, dim=-1)
        out = attention @ v
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)
        return self.out_proj(out)

    def forward(self, x, pad_mask, is_prefill=False, is_generate=False):
        B, T_q, D = x.shape
        pad_mask = pad_mask.to(device=x.device, dtype=torch.float32)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if is_generate:
            if self.kv_cache is None:
                self.kv_cache = (k, v)
                self.kv_mask = pad_mask
            else:
                k_full = torch.cat([self.kv_cache[0], k], dim=1)
                v_full = torch.cat([self.kv_cache[1], v], dim=1)
                self.kv_cache = (k_full, v_full)
                self.kv_mask = torch.cat([self.kv_mask, pad_mask], dim=1)

            k_full, v_full = self.kv_cache
            T_k = k_full.size(1)

            valid_mask = self.kv_mask.unsqueeze(-1) @ self.kv_mask.unsqueeze(-2)
            valid_mask = valid_mask.unsqueeze(1)

            causal_mask = self.causal_mask[:, :, :T_k, :T_k].to(x.device).float()
            combined_mask = causal_mask * valid_mask
            combined_mask = combined_mask[:, :, -T_q:, :]

            return self._dot_product(q, k_full, v_full, combined_mask)

        if is_prefill:
            self.kv_cache = (k, v)
            self.kv_mask = pad_mask

        pad_mask = pad_mask.unsqueeze(-1)
        pad_mask = pad_mask @ pad_mask.transpose(-1, -2)
        combined_mask = self.causal_mask[:, :, :T_q, :T_q].to(x.device).float() * pad_mask.unsqueeze(1)

        return self._dot_product(q, k, v, combined_mask)