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
    def __init__(self, d_model, n_heads, max_seq_len, rope_dims, RoPE=gpt_config["model"]["rope"], cross_attention = vit_config["model"]["cross_attention"], ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model, n_heads, max_seq_len, rope_dims, RoPE=RoPE)
        if cross_attention:
            # check two models have same batch size
            assert vit_config["data"]["batch-size"] == gpt_config["data"]["batch_size"], "vision and test models have different batch sizes!"
            # parameters for content(vision) side
            self.cross_attention = CrossAttention(
                d_kv=vit_config["model"]["d_model"],
                d_q=d_model,
                n_head=n_heads,
                dropout=gpt_config["model"]["dropout"])
            self.norm3 = nn.LayerNorm(d_model)
            self.norm4 = nn.LayerNorm(d_model)
            self.ffn_ca = FeedForward(d_model)          
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x, pad_mask, positions, is_prefill=False, is_generate=False, content = None,): # content_input is a dict: {"patches", "patch_mask"}
        '''
        Text -> LayerNorm -> Self-Attention -> Residual -> LayerNorm -> Cross-Attention -> Residual -> LayerNorm -> FFN -> Residual
        '''
        assert (content is None and not self.self.cross_attention) or (content and self.self.cross_attention)
        # self_attention
        x = x + self.attention(self.norm1(x), pad_mask, positions ,is_prefill=is_prefill, is_generate=is_generate)
        # cross_attention
        if self.cross_attention:
            x = x + self.cross_attention(content["patches"], self.norm3(x), content["mask"], pad_mask)
        x = x + self.ffn(self.norm2(x))
        
        return x