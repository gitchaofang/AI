import torch
import torch.nn as nn
import math
import numpy as np
#B batch sizae
#T token serial size
#D hidden layer dimension
#H head number
#Dh header dimension (multi header)

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
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

  

    def forward(self, x, combined_mask: torch.Tensor):
        B,T,D = x.shape

        assert combined_mask.shape[-1] == self.max_seq_len
        # build attentions [B,T,D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # multi-heead [B,T,D] -> [B,H,T,Dh]
        q = q.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_heads,self.head_dim).transpose(1,2)

        # scores
        scores = q @ k.transpose(-1,-2)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            combined_mask == 0,
            float("-inf"),
        )
        # attention
        attention = torch.softmax(scores, dim = -1)

        # value: [B,H,T,T] @ [B,H,T,Dh] -> [B,T,T,Dh]
        out = attention @ v
        # [B,H,T,Dh] ->  [B,T,D]
        out = out.transpose(1, 2).contigous()
        out = out.view(B,T,D)

        return self.out_proj(out)