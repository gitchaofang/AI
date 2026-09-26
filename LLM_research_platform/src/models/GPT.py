import torch
import yaml
from pathlib import Path
import torch.nn as nn
from src.base.transformer import TransformerBlock

# load config file
LOCAL_YAML_PATH = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/gpt.yaml")
COLAB_YAML_PATH = Path("/content/AI/LLM_research_platform/configs/gpt.yaml")
# load yaml config
with open(LOCAL_YAML_PATH,"r") as f:
    config = yaml.safe_load(f)

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size= config["tokenizer"]["vocab_size"],
        d_model=config["model"]["d_model"],
        max_seq_len=config["data"]["max_len"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        rope_dims=config["model"]["rope_dims"],
        RoPE=config["model"]["rope"],
        cross_attention= config["model"]["cross_attention"]
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.RoPE = RoPE
        self.rope_dims = rope_dims

        self.token_embedding = nn.Embedding(vocab_size + 2 ,d_model,) # "+2" becasue 0 is for padding and 1 is for EOS
        # if not using RoPE, we use learnable positional embedding
        if not RoPE:
            self.position_embedding = nn.Embedding(max_seq_len,d_model,)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    max_seq_len=max_seq_len,
                    rope_dims=rope_dims,
                    cross_attention_enabled=cross_attention,
                    RoPE = RoPE,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model,)
        self.lm_linear = nn.Linear(d_model, vocab_size + 2,)
        self.position_offset = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_cache(self):
        """
        Clear all KV caches and reset logical position tracking.
        Call this before starting a new generation session.
        """
        self.position_offset = None

        for block in self.blocks:
            block.attention.reset_cache()

    def forward(
        self,
        x,
        pad_mask,
        positions,
        is_prefill=False,
        is_generate=False,
    ):
        B, T_q = x.shape

        assert T_q <= self.max_seq_len

        assert positions.ndim == 3, (
            "positions must have shape [B, T, M]"
        )

        assert positions.shape[0] == B
        assert positions.shape[1] == T_q
        if self.RoPE:
            assert positions.shape[2] == len(self.rope_dims)

        assert not (
            is_prefill and is_generate
        ), "is_prefill and is_generate cannot both be True"


        # -----------------------------------------
        # Token embedding
        # -----------------------------------------
        x = self.token_embedding(x)

        if not self.RoPE:
            if not is_prefill and not is_generate:
                position_ids = torch.arange(T_q,device=x.device,).unsqueeze(0).expand(B,T_q,)
            elif is_prefill:
                position_ids = torch.arange(T_q,device=x.device,).unsqueeze(0).expand(B,T_q,)
                self.position_offset = (pad_mask.to(torch.int64).sum(dim=1))
            else:
                assert is_generate
                assert self.position_offset is not None, (
                    "Generation requires a previous prefill"
                )
                relative_positions = torch.arange(T_q,device=x.device,).unsqueeze(0)
                position_ids = (self.position_offset.unsqueeze(1) + relative_positions)

            assert torch.all(position_ids >= 0)
            assert torch.all(position_ids < self.max_seq_len), ("Position ID exceeds max_seq_len")
            # ------------------------------------------------------
            # learnable positional embedding only when RoPE is False
            # ------------------------------------------------------
            x = (x + self.position_embedding(position_ids))

        # -----------------------------------------
        # Transformer blocks
        # ----------------------------------------- 
        for block in self.blocks:
            x = block(
                x,
                pad_mask,
                positions,
                is_prefill=is_prefill,
                is_generate=is_generate,
            )
        x = self.norm(x)
        logits = self.lm_linear(x)

        # -----------------------------------------
        # Update generation position
        # -----------------------------------------
        if is_generate and not self.RoPE:
            self.position_offset += (pad_mask.to(torch.int64).sum(dim=1))

        return logits