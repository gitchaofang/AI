import torch
from torch import nn
import math

class CrossAttention(nn.Module):
    def __init__(self,d_kv: int, d_q: int, d_model: int, n_head: int, dropout = 0.2): # For cross_attention k and v have the same dimension
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_q = d_q
        self.n_head = n_head
        self.head_dim = d_model // n_head

        # projections
        self.k_proj = nn.Linear(d_kv, d_model)
        self.v_proj = nn.Linear(d_kv, d_model)
        self.q_proj = nn.Linear(d_q, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _dot_product(self, k_x, v_x, q_x, combined_mask):
        B, T_kv, _ = k_x.shape
        _, T_q, _ = q_x.shape

        # multi_heads
        k_x = k_x.view(B, T_kv, self.n_head, self.head_dim).transpose(1,2)
        v_x = v_x.view(B, T_kv, self.n_head, self.head_dim).transpose(1,2)
        q_x = q_x.view(B, T_q, self.n_head, self.head_dim).transpose(1,2)

        # scores
        scores = q_x @ k_x.transpose(-1,-2) #[B_q,H, T_q, D_h] @ [B_q, H, D_h, T_kv] -> [B_q, H, T_q, T_k]
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(combined_mask == 0,-1e10)

        attention = torch.softmax(scores, dim = -1)
        attention = self.attn_dropout(attention)

        out = attention @ v_x
        out = out.transpose(1,2).contiguous().view(B, T_q, self.d_model)
        out = self.out_proj(out)

        # dropout on out
        out = self.out_dropout(out)
        return out

    # here masks are pad_masks. causal mask is not needed in cross-attention
    def forward(self, content, q_x, kv_mask,q_mask): # usually k and va come from the same encode so we only use one input "content"
        # k_x: [B_kv, T_kv, D_kv]
        # q_x: [B_q, T_q, D_q]
        # kv_mask: [B_kv, T_kv]
        # q_mask: [B_q, T_q]
        B_kv, T_kv, D_kv = content.shape
        B_q, T_q, D_q = q_x.shape
        assert B_kv == B_q and D_kv ==self.d_kv and D_q == self.d_q

        # transform to n_model
        k_x = self.k_proj(content)
        v_x = self.v_proj(content)
        q_x = self.q_proj(q_x)
        
        # build a combined mask
        pad_mask = q_mask.unsqueeze(-1) @ kv_mask.unsqueeze(-2) #[B_q, T_q] @ [B_k,T_k] -> [B_q, T_q, T_k]
        combined_mask = pad_mask.unsqueeze(1) # [B_q, T_q, T_k] -> [B_q, 1, T_q, T_k]

        return self._dot_product(k_x, v_x, q_x, combined_mask)






