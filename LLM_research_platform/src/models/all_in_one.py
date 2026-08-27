import torch
import torch.nn as nn
import math

# self attention
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
        scores = scores / math.sqrt(self.haed_dim)
        scores = scores.masked_fill(
            combined_mask,
            float("-inf")
        )
        # attention
        attention = torch.softmax(scores, dim = -1)

        # value: [B,H,T,T] @ [B,H,T,Dh] -> [B,T,T,Dh]
        out = attention @ v
        # [B,H,T,Dh] ->  [B,T,D]
        out = out.transpose(1,2).contigous()
        out = out.view(B,T,D)

        return self.out_proj(out)

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
    def __init__(self, d_model: int, n_heads:int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model, n_heads, max_seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)
    def forward(self, x: torch.Tensor, mask):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self,
            vocab_size :int,
            d_model: int,
            n_layers: int,
            n_heads: int,
            max_seq_len: int,
        ):
        super().__init__()
        self.token_embedding = nn.embedding(vocab_size, d_model)
        self.position_embedding = nn.embedding(max_seq_len, d_model)
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

    def forwrd(self,x: torch.Tensor, pad_mask: torch.Tensor):
        B,T = x.shape
        assert T <= self.max_seq_len

        # build combined mask: causal mask [1,1,max_seq_len, max_seq_len] + pad_mask[B,T] -> [B,1,T,T]
        pad_mask = pad_mask.unsqueeze(-1)
        pad_mask = pad_mask @ pad_mask.transpose(-1,-2)
        combined_mask = self.causal_mask[:,:,:T,:T] * pad_mask.unsqueeze(1)

        # embedding
        pos_seq = torch.arange(
            T,
            dtype = torch.int
        )
        x = self.token_embedding(x) + self.position_embedding(pos_seq)

        # transformer blocks
        for block in self.blocks:
            x = block(x, combined_mask)

        # forward
        x = self.norm(x)
        logits = self.lm_linear(x)

        return logits



