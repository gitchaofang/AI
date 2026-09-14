import torch
import random
import numpy as np
import os
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from src.data.regex_tokenizer import RegexTokenizer

LOCAL_FILE_PATH =  Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/")
LOCAL_INDEX_PATH =  Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/index_files")
COLAB_FILE_PATH = Path("/content/AI/LLM_research_platform/src/data/input_data/data/")
COLAB_INDEX_PATH = Path("/content/AI/LLM_research_platform/src/data/input_data/data/index_files")
#seperation pattern
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class TextDecodeDataset(Dataset): # for txt file
    def __init__(self, tokenizer, max_len, filename):
        self.file_dir = LOCAL_FILE_PATH
        self.index_dir = LOCAL_INDEX_PATH
        self.filename = filename
        self.max_len = max_len
        self.tokenizer = tokenizer

        read_path = self.file_dir / filename

        stem = Path(filename).stem

        self.token_path = self.index_dir / f"{stem}_{max_len}_tokens.bin"
        self.index_path = self.index_dir / f"{stem}_{max_len}_index.npy"



        with open(read_path, "r", encoding="utf-8") as f:
            self.text = f.read()
        if not (os.path.exists(self.token_path) and os.path.exists(self.index_path)):
            self.data_prep()

        self.tokens = np.memmap(
             self.token_path,
             dtype = np.int32,
             mode="r"
        )
        self.index = np.load(
             self.index_path,
             mmap_mode = "r"
        )

    def data_prep(self):
        encoded_tokens = self.tokenizer.encode(self.text)
        all_tokens = [item for token_list in encoded_tokens for item in token_list]
        all_token_len = len(all_tokens)
        all_index = []
        offset = 0

        while offset < all_token_len:
            end = min(offset + self.max_len, all_token_len)
            all_index.append([offset, end - offset])
            offset = end

        # Convert to NumPy
        all_tokens = np.asarray(all_tokens,dtype=np.int32)
        index = np.asarray(all_index,dtype=np.int64)

        # Save
        all_tokens.tofile(self.token_path)
        np.save(self.index_path, index)

    def __len__(self):
         return len(self.index)
    
    def __getitem__(self, key):
         offset,length = self.index[key]
         tokens = self.tokens[offset: offset + length]
         data = torch.tensor(tokens, dtype = torch.int64)
         return {
              "input_ids": data[:-1],
              "labels": data[1:]
         }

    def get_length(self, key):
         return self.index[key][1]

        

class TokenBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = list(range(len(dataset)))
        self.batches = []

        self._make_batches()

    def _make_batches(self):
        indices = sorted(
            self.indices,
            key=lambda i: self.dataset.get_length(i)
        )

        for idx in range(0, len(indices), self.batch_size):
            self.batches.append(
                indices[idx:idx + self.batch_size]
            )

        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


class PaddingCollator:
    def __init__(self,token_pad = 0, label_pad = -100,):

        self.token_pad = token_pad
        self.label_pad = label_pad

    def __call__(self, batch):
        batch_size = len(batch)
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = torch.full(
            (batch_size,max_len),
            self.token_pad,
            dtype = torch.int64,
        )

        label_ids = torch.full(
             (batch_size,max_len),
             self.label_pad,
             dtype = torch.int64,
        )

        pad_mask = torch.zeros(
            batch_size,
            max_len,
            dtype = torch.int64,
        )

        for i, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[i][:length] = batch[i]["input_ids"]
            label_ids[i][:length] = batch[i]["labels"]
            pad_mask[i][:length] = 1

        return {
            "input_ids": input_ids,
            "labels": label_ids,
            "pad_mask": pad_mask,
        }
    