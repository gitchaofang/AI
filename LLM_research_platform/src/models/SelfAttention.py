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
    def __init__(self,d_model: int, n_heads: int, max_seq_len: int, mask: torch.Tensor):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_heads = d_model / n_heads
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.register_buffer(
            "causal_mask",
            mask.view(1,1,mask.shape[0],mask.shape[1])
        )
    def forward(self, x: torch.Tensor){
        B,T,D = x.shape

        # [B,T,D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B,T,D] ->[B,H,T,Dh]
        q = q.view(B, T, self.n_heads, self.d_heads).transpose(1,2)
        k = k.view(B, T, self.n_heads, self.d_heads).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.d_heads).transpose(1,2)

        # [B,H,T,Dh] @ [B,H,Dh,T] -> [B,H,T,T]
        scores =  torch.matmul(q, k.transposer(-2,-1))
        scores = scorers / math.sqrt(self,head.d_heads)
        # apply mask
        scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        attention = torch.softmax(scores, dim = -1)

        # [B,H,T,Dh] -> [B,H,T,D] -> [B,T,D]
        out = torch.matmul(attention, v)
        out = out.transpose(1,2).contiguous()
        out = out.view(B,T,D)

        returrn self.out_proj(out)




        # apply mask

    }



    
