import torch
import pytest
import os
import regex as re
from torch.utils.data import DataLoader
from src.data.regex_tokenizer import RegexTokenizer
from src.data.all_in_one import TextDecodeDataset
from src.data.all_in_one import TokenBatchSampler
from src.data.all_in_one import PaddingCollator

DATA_DIR =  "/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/"
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
ENCODE_DECODE_FILENAME = "swift.txt"
DATASET_INPUT_FILENAME = "swift.txt"

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
     
def test_encode_decode(tokenizer, test_text):
    print(f"start testing")
    encoded_ids = tokenizer.encode(test_text)
    decoded_chunks = [tokenizer.decode(ids) for ids in encoded_ids]
    decoded_text = "".join(decoded_chunks)
    print(f"{decoded_text} \n {test_text}")
    assert decoded_text == test_text

def test_dataset(tokenizer):
    dataset = TextDecodeDataset(tokenizer=tokenizer,
                                max_len=512,
                                filename=DATASET_INPUT_FILENAME)
    batch_sampler = TokenBatchSampler(dataset=dataset, 
                                batch_size=8)
    collator = PaddingCollator()
    loader = DataLoader(dataset,
                        collate_fn = collator,
                        batch_sampler = batch_sampler,
                        shuffle = False,
                        pin_memory = True)
    for i, batch in enumerate(loader):
        x = batch["input_ids"]
        y = batch["labels"]
        assert torch.equal(x[:, 1:], y[:, :-1])

