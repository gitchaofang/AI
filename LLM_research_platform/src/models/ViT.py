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
                rope_dims = [64],
                cls_enabled=True,
                causal = False,
                cross_attention_enabled=False):

        super().__init__()

        self.d_model=d_model
        self.n_heads=n_heads
        self.dropout=dropout
        self.patch_size=patch_size
        self.max_seq_len = max_seq_len
        self.RoPE=RoPE
        self.rope_dims=rope_dims
        self.mlp_ratio=mlp_ratio
        self.patch_dim=in_channels * patch_size * patch_size
        self.num_class=num_class
        self.cls_enabled=cls_enabled
        self.causal=causal
        self.dropout=dropout
        self.cross_attention_enabled=cross_attention_enabled

        # token embedding
        self.token_embedding = nn.Linear(self.patch_dim, self.d_model)

        # if not using RoPE, we use learnable positional embedding
        if not self.RoPE:
            self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model,)

        if self.cls_enabled:
            self.cls_token = nn.Parameter(torch.zeros(1,1,self.d_model))
            self.register_buffer(
                "cls_pos",
                torch.ones(
                    1, 1, len(self.rope_dims),
                    dtype=torch.int64,
                )
            ) #[1,1,self.rope_dims]
            self.max_seq_len += 1


        # self attention layers
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    max_seq_len=self.max_seq_len,
                    rope_dims=self.rope_dims,
                    RoPE=self.RoPE,
                    cross_attention_enabled=self.cross_attention_enabled,
                    dropout = self.dropout,
                    causal = self.causal,
                    cls_enabled = self.cls_enabled,
                )
                for _ in range(n_layers)
            ]
        )

        # norm
        self.norm = nn.LayerNorm(d_model)

        # classification head
        if self.num_class is not None:
            self.head = nn.Linear(d_model, self.num_class)

        # initialization for weights
        self._init_weight()

    def _init_weight(self):
        if self.cls_enabled:
            nn.init.trunc_normal_(
                self.cls_token,
             std=0.02,
            )


    def forward(self, patch_items):
        # inputs
        patch_input = patch_items["patched_input"] # [B, T, in_channel * patch_size * patch_size]
        B, T, patch_dim = patch_input.shape #B: batch_siae, T: patch numbers, D: patchify dimention

        assert self.patch_dim == patch_dim, "Patch dimension mismatch"

        pad_mask = patch_items["pad_mask_patch"] #[B, T]
        patch_positions = patch_items["patch_positions"] # [B,T,2]

        # patch embedding
        patched_seq = self.token_embedding(patch_input)

        # update when CLS is enabled
        if self.cls_enabled:
            # add cls token at teh beggining of the patch series
            cls = self.cls_token.expand(B,-1,-1,)
            patched_seq = torch.cat([cls, patched_seq], dim=1,) #[B, T + 1, d_model]

            # add extra dimention for patch_mask and patch_position
            cls_mask = torch.ones(
                B, 1,
                dtype=pad_mask.dtype,
                device=pad_mask.device,
            )

            pad_mask = torch.cat(
                [cls_mask, pad_mask],
                dim=1,
            )

            # update position for RoPE
            if self.RoPE:
                cls_position = self.cls_pos.expand(B,1,-1,)
                patch_positions = torch.cat([cls_position,patch_positions], dim = 1)

            # T is original T + 1 if CLS is enabled
            T += 1

        if not self.RoPE:
            position_ids = torch.arange(T,device=patch_input.device,).unsqueeze(0).expand(B,T,) # [B, T]
            patched_seq = (patched_seq + self.position_embedding(position_ids))

        # Transformer blocks
        for block in self.blocks:
            patched_seq = block(
                x=patched_seq,
                pad_mask=pad_mask,
                positions=patch_positions,
                is_prefill=False,
                is_generate=False,
                content = None,
            )
        out = self.norm(patched_seq)

        # Separate CLS and patches
        if self.cls_enabled:
            cls_out = out[:, 0]
            patch_out = out[:, 1:]
        else:
            cls_out = None
            patch_out = out

        # Classification
        if self.num_class is not None:
            assert self.cls_enabled, "Classification requires CLS token"
            logits = self.head(cls_out)

            return {
                "class_logits": logits,
                "patch_seq": patch_out,
            }

        return patch_out