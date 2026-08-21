import torch
from src.models.SelfAttension import SelfAttention

def test_attention_shape():
    model = SelfAttention(
        d_model = 128,
        n_heads = 4,
        max_seq_len = 32,
    )

    x = torch.randn(shape = (2,16,128))
    y = model(x)
    assert y.shape == x.shape