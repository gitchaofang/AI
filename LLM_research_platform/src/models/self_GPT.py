import torch
import torch.nn as nn
from .transformer import TransformerBlock

class GPT(nn.Module):
    def __init__(self, verb_size: int, n_layers: int, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        self.verb_size = verb_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        #embedding
        self.token_embedding = nn.embedding(self.verb_size, self.d_model)
        self,position_embedding = nn.embedding(self.max_seq_len, self.d_model)
        self.mask = torch.tril(torch.ones(size = (self.max_seq_len, self.max_seq_len)))

        # transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(self.d_model,
                             self.n_heads,
                              self.max_seq_len, 
                              self.mask,)
                              for _ in range(self.n_layers)]
        )

        self.norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, verb_size, bias = False)

        def forward(self, input_ids: torch.Tensor):

            B,T = input_ids.shape
            assert T <= self.max_seq_len

            # embedding
            position = torch.arange(size = [T], dtype = torch.float, device = input_ids.device)
            position.unsqueeze(dim = 0).expand(B,-1)

            x = self.token_embedding(input_ids) + self.position_embedding(position)

            # transfomer
            for block in (self.blocks):
                x = block(x)

            # linear
            x = self.norm(x)
            out = self.lm_head(x)
            return out

                    


