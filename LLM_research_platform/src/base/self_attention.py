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
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, rope_dims, RoPE=True, pad_token=0, dropout = 0.2):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // n_heads

        # M-rope config
        self.rope_dims = rope_dims
        self.RoPE = RoPE

        assert len(rope_dims) >= 1
        assert sum(rope_dims) == self.head_dim
        assert all(dim % 2 == 0 for dim in rope_dims)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.atten_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        mask = torch.tril(
            torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.int64)
        )
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, self.max_seq_len, self.max_seq_len),
        )

        self.kv_cache = None
        self.kv_mask = None
        self.kv_positions = None

        self.pad_token = pad_token

    def reset_cache(self):
        self.kv_cache = None
        self.kv_mask = None
        self.kv_positions = None

    def _apply_rope(self, x, positions):
        '''
        x: 
            [B, H, T ,D]
        
        positions:
            [B, T, M]
        
        rope_dims:
            [M]
            sum must equal D
            Each dimention must be even
        '''

        B, H, T, D = x.shape
        B_pos, T_pos, M = positions.shape

        assert B_pos == B
        assert T_pos == T
        assert len(self.rope_dims) == M
        assert sum(self.rope_dims) == D
        assert all(dim % 2 == 0 for dim in self.rope_dims)

        out = torch.empty_like(x)

        dim_start = 0

        for m, rope_dim in enumerate(self.rope_dims):

            dim_end = dim_start + rope_dim

            # Features assigned to positional dimension m
            x_m = x[...,dim_start:dim_end]

            # Position coordinate for dimension m
            # [B, T]
            position_m = positions[...,m]

            # Number of rotation pairs
            n_pairs = rope_dim // 2

            # [n_pairs]
            inv_freq = 1.0 / (
                10000.0 ** (
                    2.0
                    * torch.arange(
                        n_pairs,
                        device=x.device,
                        dtype=torch.float32,
                    )
                    / rope_dim
                )
            )

            # [B, T, n_pairs]
            angles = (position_m.float().unsqueeze(-1) * inv_freq)


            # [B, 1, T, n_pairs]
            cos = torch.cos(angles).unsqueeze(1)
            sin = torch.sin(angles).unsqueeze(1)

            # [B, H, T, n_pairs]
            x_even = x_m[...,0::2]
            x_odd = x_m[...,1::2]

            # apply rotation
            out[..., dim_start:dim_end:2] = (x_even * cos - x_odd * sin)
            out[..., dim_start + 1:dim_end:2] = (x_even * sin + x_odd * cos)

            dim_start = dim_end

        return out

    
    def _attention(self, q, k, v, combined_mask): #combined_mask: [B,H,T_q, T_k]
        B, H, T_q, D_h = q.shape
        
        scores = q @ k.transpose(-1, -2)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(combined_mask == 0, -1e10)

        attention = torch.softmax(scores, dim=-1)

        # dropout on attention
        attention = self.atten_dropout(attention)
        out = attention @ v
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)

        out = self.out_proj(out)
        # dropout on out project
        out  = self.out_dropout(out)
        return out


    def forward(self, x, pad_mask, positions, is_prefill=False, is_generate=False):# For normal self-attention, k_positions and v_positions are the same
        B, T_q, D = x.shape
        pad_mask = pad_mask.to(device=x.device, dtype=torch.float32)

        # q,k,v: [B,H,T,D_h]
        q = self.q_proj(x).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        
        # M-RoPE
        if self.RoPE:
            q = self._apply_rope(q, positions)
            k = self._apply_rope(k, positions) 

        # ---------------------------------
        # Generation with KV cache
        # ---------------------------------
        if is_generate:
            if self.kv_cache is None:
                self.kv_cache = (k, v)
                self.kv_mask = pad_mask
                self.kv_positions = positions
            else:
                k_full = torch.cat([self.kv_cache[0], k], dim=2)
                v_full = torch.cat([self.kv_cache[1], v], dim=2)
                self.kv_cache = (k_full, v_full)
                self.kv_mask = torch.cat([self.kv_mask, pad_mask], dim=1)
                self.kv_positions = torch.cat([self.kv_positions, positions], dim=1,)


            k_full, v_full = self.kv_cache
            T_k = k_full.shape[2]
            assert T_k <= self.max_seq_len, ( "KV cache exceeds max_seq_len")

            valid_mask = self.kv_mask.unsqueeze(-1) @ self.kv_mask.unsqueeze(-2)
            valid_mask = valid_mask.unsqueeze(1)

            causal_mask = self.causal_mask[:, :, :T_k, :T_k].to(x.device).float()
            combined_mask = causal_mask * valid_mask
            combined_mask = combined_mask[:, :, -T_q:, :]

            return self._attention(q, k_full, v_full, combined_mask)

        # ---------------------------------
        # Prefill
        # ---------------------------------
        if is_prefill:
            self.kv_cache = (k, v)
            self.kv_mask = pad_mask
            self.kv_positions = positions

        # ---------------------------------
        # Normal training / prefill attention
        # ---------------------------------
        pad_mask = pad_mask.unsqueeze(-1)
        pad_mask = pad_mask @ pad_mask.transpose(-1, -2)
        combined_mask = self.causal_mask[:, :, :T_q, :T_q].to(x.device).float() * pad_mask.unsqueeze(1)

        return self._attention(q, k, v, combined_mask)