import torch
from torch import nn

class CrossAttention(nn.Module):
    def __init__(self,d_kv: int, d_q: int, d_model: int, n_head: int, max_len: int, pad_token = 0, dropout = 0.2): # For cross_attention k and v have the same dimension
        super.__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_q = d_q
        self.n_head = n_head
        self.max_len = max_len
        self.pad_token = pad_token
        self.head_dim = d_model // n_head

        # projections
        self.k_proj = nn.Linear(d_kv, d_model)
        self.v_proj = nn.Linear(d_kv, d_model)
        self.q_proj = nn.Linear(d_q, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _dot_product(self,k_x,v_x,q_x, combined_mask):
        B_kv, T_kv, D_kv = k_x.shape
        B_q, T_q, D_q = q_x.shape

        # multi_heads
        k_v = k_v.view(B_kv, T_kv, self.n_head, self.head_dim)
        v_v = v_v.view(B_kv, T_kv, self.n_head, self.head_dim)
        q_v = q_v.view(B_q, T_q, self.n_head, self.head_dim)

    # here masks are pad_masks. causal mask is not needed in cross-attention
    def forward(self, k_x, v_x, q_x, kv_mask,q_mask):
        # k_x: [B_kv, T_kv, D_kv]
        # q_x: [B_q, T_q, D_q]
        # kv_mask: [B_kv, T_kv]
        # q_mask: [B_q, T_q]
        B_kv, T_kv, D_kv = k_x.shape
        B_q, T_q, D_q = q_x.shape
        assert B_kv == B_q and D_kv ==self.d_kv and D_q == self.d_q

        # transform to n_model
        k_x = self.k_proj(k_x)
        v_x = self.v_proj(v_x)
        q_x = self.q_proj(q_x)
        
        # build a combined mask
        pad_mask = q_mask.unsqueeze(-1) @ kv_mask.unsqueeze(-2) #[B_q, T_q] @ [B_k,T_k] -> [B_q, T_q, T_k]
        combined_mask = pad_mask.unsqueez(1) # [B_q, T_q, T_k] -> [B_q, H, T_q, T_k]






