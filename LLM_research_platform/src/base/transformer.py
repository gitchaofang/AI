import torch
import yaml
from pathlib import Path
import torch.nn as nn
from .self_attention import SelfAttention
from .cross_attention import CrossAttention

# load config file for vit
LOCAL_YAML_PATH = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/vit.yaml")
COLAB_YAML_PATH = Path("/content/AI/LLM_research_platform/configs/vit.yaml")
# load yaml config
with open(COLAB_YAML_PATH,"r") as f:
    # vit config
    vit_config = yaml.safe_load(f)

# load config file for gpt
LOCAL_YAML_PATH = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/gpt.yaml")
COLAB_YAML_PATH = Path("/content/AI/LLM_research_platform/configs/gpt.yaml")
# load yaml config
with open(COLAB_YAML_PATH,"r") as f:
    gpt_config = yaml.safe_load(f)

# transformer:
class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio = 4, dropout = 0.2,):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_heads, 
                 max_seq_len, 
                 rope_dims, 
                 RoPE=True, 
                 cross_attention_enabled=True,
                 dropout = 0.2 
    ):
        super().__init__()
        self.cross_attention_enabled = cross_attention_enabled
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model=d_model, 
                                       n_hdaes=n_heads, 
                                       max_seq_len=max_seq_len, 
                                       rope_dims=rope_dims, 
                                       RoPE=RoPE)
        self.norm2 = nn.LayerNorm(d_model)

        if cross_attention_enabled:
            # parameters for content(vision) side
            self.cross_attention = CrossAttention(
                d_kv=vit_config["model"]["d_model"],
                d_q=d_model,
                n_head=n_heads,
                dropout=gpt_config["model"]["dropout"])
        self.norm3 = nn.LayerNorm(d_model)     
        self.ffn = FeedForward(d_model=d_model,
                               mlp_ratio = gpt_config["model"]["mlp_ratio"],
                               dropout = dropout)

    def forward(self, x, pad_mask, positions, is_prefill=False, is_generate=False, content = None,):
        """
        Text:
            x: [B, T, D]
        Image content:
            content["patches"]: [B, N, D]
            content["mask"]:    [B, N]

        Output:
            [B, T, D]
        """ 
        """
        Text -> LayerNorm -> Self-Attention -> Residual -> LayerNorm -> Cross-Attention -> Residual -> LayerNorm -> FFN -> Residual
        """
        # 1. Causal text self-attention
        x = x + self.attention(
            x=self.norm1(x), 
            pad_mask=pad_mask, 
            positions=positions,
            is_prefill=is_prefill, 
            is_generate=is_generate
        )
        # 2.Text -> image cross-attention
        if self.cross_attention_enabled:
            x = x + self.cross_attention(
                content = content["patches"], 
                q_x = self.norm2(x), 
                kv_mask = content["mask"], 
                q_mask = pad_mask,
            )
        else:
            assert content is None
        # 3. Feed-forward network
        x = x + self.ffn(self.norm3(x))
        
        return x