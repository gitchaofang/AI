import torch
import yaml
import numpy as np
from pathlib import Path
from torch import nn
from PIL import Image
from .helper import patchify
from torchvision import transforms
from src.base.transformer import TransformerBlock

# load config file
LOCAL_YAML_PATH = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/vit.yaml")
COLAB_YAML_PATH = Path("/content/AI/LLM_research_platform/configs/vit.yaml")
# load yaml config
with open(COLAB_YAML_PATH,"r") as f:
    vit_config = yaml.safe_load(f)
class ViT(nn.Module):
    def __init__(self,
                 d_model: int, 
                 n_heads: int, 
                 in_channels: int = 3, 
                 max_seq_len: int = 512,
                 patch_size = 16,
                 n_layers: int = 5,
                 mlp_ratio = 4.0,
                 dropout: float = 0.2,
                 RoPE: bool = True,
                 num_class: int = None,
                 rope_dim = [64],):

        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.patch_size = patch_size
        self.RoPE = RoPE
        self.rope_dim = rope_dim
        self.mlp_ratio = mlp_ratio
        self.patch_dim = in_channels * patch_size * patch_size
        self.num_class = num_class

        # token embedding
        self.token_embedding = nn.Linear(self.patch_dim, d_model)
        # if not using RoPE, we use learnable positional embedding
        if not RoPE:
            self.position_embedding = nn.Embedding(max_seq_len,d_model,)

         # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))

        # if not using RoPE, we use learnable positional embedding
        if not self.RoPE:
            self.pos_embed = nn.Parameter(
            torch.zeros(1,self.patch_dim + 1,d_model,))

        # self attention layers
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    max_seq_len=max_seq_len,
                    rope_dims=self.rope_dims,
                    cross_attention_enabled=False,
                    RoPE=self.RoPE,
                )
                for _ in range(n_layers)
            ]
        )

        # norm
        self.norm = nn.LayerNorm(d_model)

        # classification head
        if self.num_class:
            self.head = nn.Linear(d_model, self.num_class)

        # initialization for weights
        self._init_weight()

    def _init_weight(self):
        nn.init.trunc_normal_(
        self.cls_token,
        std=0.02),


    def forward(self, patch_items): # x, pad_mask, positions, is_prefill=False, is_generate=False, content = None,
        patch_input = patch_items["patched_input"]
        B, T, D = patch_input.shape
        pad_mask = patch_items["pad_mask_patch"] #[B, T]
        patch_positions = patch_items["patch_positions"] # [B,T,2]
        
        cls = self.cls_token.expand(B,-1,-1,)
        patched_seq = torch.cat([cls, self.token_embedding(patch_input)], dim=1,) #[B, T + 1, d_model]

        if not self.RoPE:
            position_ids = torch.arange(T,device=patch_input.device,).unsqueeze(0).expand(B,T,)
            patched_seq = (patched_seq + self.position_embedding(position_ids))

        # Transformer blocks
        for block in self.blocks:
            patched_seq = self.block(
                x=patched_seq,
                pad_mask=pad_mask,
                positions=patch_positions,
                is_prefill=False,
                is_generate=False,
                content = None,
            )
        out = self.norm(patched_seq)

        # For classification
        if self.num_class:
            cls = out[:,0] # [B,D]
            logits = self.head(cls)
            return {"class_logits": logits,
                    "patch_seq": out[:,1:]} 
        return out[:,1:] # [B, T, d_model]





