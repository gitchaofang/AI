import torch
import pytest
import os
import regex as re
from torch.utils.data import DataLoader
from src.data.regex_tokenizer import RegexTokenizer
from src.data.all_in_one import TextDecodeDataset
from src.data.all_in_one import TokenBatchSampler
from src.data.all_in_one import PaddingCollator
from src.models.all_in_one import SelfAttention
from src.models.all_in_one import Trainer
from src.models.all_in_one import GPT

DATA_DIR =  "/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/"
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
ENCODE_DECODE_FILENAME = "swift.txt"
DATASET_INPUT_FILENAME = "swift.txt"
B = 10
T = 26
D = 32

@pytest.fixture
# create tokenizer and train it
def tokenizer():
    tokenizer = RegexTokenizer()
    tokenizer.train()
    return tokenizer

@pytest.fixture
# test text
def test_text():
    read_path = os.path.join(DATA_DIR, ENCODE_DECODE_FILENAME)
    with open(read_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

@pytest.fixture
# model input package:
# x: input_tensor
# positions: tentor for RoPE
# pad_mask: padding mask
# rope_dims: dims for RoPE
def model_pram():
    # x [B,T,D]
    # pad_mask [B,T]
    x = torch.randint(2,255,(B,T,D), dtype = torch.int64)
    pad_mask = torch.ones((B,T), dtype = torch.int)
    # add padding
    x[:,-10:,:] = 0
    pad_mask[:,-10:] = 0

    # positions: [B,T,M] M = 3
    positions = torch.full((B,T,3), -1, dtype = torch.int64)
    
    for j in range(B):
        coord = torch.randint(0,224,(3,), dtype = torch.int)
        positions[j,:-10,:] = coord

    # rope_dims
    rope_dims = [4,6,6]

    return {"ids": x,
            "pos": positions,
            "mask": pad_mask,
            "rdim": rope_dims}

# ------dataset---------
def test_encode_decode(tokenizer, test_text):
    print(f"start testing")
    encoded_ids = tokenizer.encode(test_text)
    decoded_chunks = [tokenizer.decode(ids) for ids in encoded_ids]
    decoded_text = "".join(decoded_chunks)
    print(f"{decoded_text} \n {test_text}")
    assert decoded_text == test_text


def test_dataset_shift(tokenizer):
    dataset = TextDecodeDataset(
        tokenizer=tokenizer,
        max_len=512,
        filename=DATASET_INPUT_FILENAME
    )

    for i in range(len(dataset)):
        item = dataset[i]

        x = item["input_ids"]
        y = item["labels"]

        assert len(x) == len(y), f"Length mismatch at {i}"

        if not torch.equal(x[1:], y[:-1]):
            print(f"FAILED at dataset index {i}")

            for j in range(min(len(x[1:]), len(y[:-1]))):
                if x[1:][j] != y[:-1][j]:
                    print(
                        f"Mismatch at position {j}: "
                        f"x={x[1:][j].item()}, "
                        f"y={y[:-1][j].item()}"
                    )
                    break

            raise AssertionError

def test_collator(tokenizer):
    dataset = TextDecodeDataset(
        tokenizer=tokenizer,
        max_len=512,
        filename=DATASET_INPUT_FILENAME
    )

    collator = PaddingCollator()

    samples = [
        dataset[0],
        dataset[1],
    ]

    batch = collator(samples)

    x = batch["input_ids"]
    y = batch["labels"]
    mask = batch["pad_mask"]

    for i, item in enumerate(samples):
        length = len(item["input_ids"])

        assert torch.equal(
            x[i, :length],
            item["input_ids"]
        )

        assert torch.equal(
            y[i, :length],
            item["labels"]
        )

def test_dataloader(tokenizer):
    dataset = TextDecodeDataset(
        tokenizer=tokenizer,
        max_len=512,
        filename=DATASET_INPUT_FILENAME
    )

    batch_sampler = TokenBatchSampler(
        dataset=dataset,
        batch_size=8,
        shuffle=False
    )

    collator = PaddingCollator()

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        pin_memory=True
    )

    for batch_indices, batch in zip(batch_sampler, loader):

        x = batch["input_ids"]
        y = batch["labels"]
        mask = batch["pad_mask"]

        # Check every sample against the original Dataset
        for row, dataset_idx in enumerate(batch_indices):
            original = dataset[dataset_idx]

            length = len(original["input_ids"])

            assert torch.equal(
                x[row, :length],
                original["input_ids"]
            )

            assert torch.equal(
                y[row, :length],
                original["labels"]
            )

def test_dataset(tokenizer):
    dataset = TextDecodeDataset(tokenizer=tokenizer,
                                max_len=512,
                                filename=DATASET_INPUT_FILENAME)
    batch_sampler = TokenBatchSampler(dataset=dataset, 
                                batch_size=8,
                                shuffle=False)
    collator = PaddingCollator()
    loader = DataLoader(dataset,
                        collate_fn = collator,
                        batch_sampler = batch_sampler,
                        pin_memory = True)

    for batch in loader:
        x = batch["input_ids"]
        y = batch["labels"]
        mask = batch["pad_mask"]

        for i in range(x.size(0)):
            valid_len = mask[i].sum().item()
            if valid_len > 1:
                assert torch.equal(x[i, 1:valid_len],y[i,:valid_len - 1])


# ----------model--------------
def test_Selfattention(model_pram):
    x = model_pram["ids"]
    positions = model_pram["pos"]
    pad_mask = model_pram["mask"]
    rope_dims = model_pram["rdim"]

    self_attention = SelfAttention(d_model = D,
                                   n_heads = 2,
                                   max_seq_len=T,
                                   rope_dims= rope_dims,
                                   RoPE = True)

    result_attention = self_attention(x, pad_mask, positions)
    assert result_attention.shape == x.shape
    assert torch.isfinite(result_attention).all()
