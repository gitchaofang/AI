import torch
import random
from totch.utils.data import Dataset
from torch.utils.data import Sampler

class VariableLengthDataset(Dataset):
    def __init__(self,sentences):
        self.sentences = sentences
        chars = set()
        for sentence in sentences:
            chars.update(sentence)
        self.char_set = sorted(chars)
        self.stoi = {
            char: i + 1 for i, char in enumerate(self.char_set)
        }
        self.itos = {
            i + 1: char for i, char in enumerate(self.char_set)
        }

        self.tokens = []

        for sentence in sentences:
            current_sentence = [
                self.stoi[char]
                for char in sentence
            ]
            self.tokens.append(current_sentence)   
        self.tokens = torch.tensor(self.tokens)    

    def __len__(self):
        return len(self.tokens)
    def __getitem__(self,idx):
        sentence = self.tokens[idx]
        return{
            "input_ids": sentence[:-1],
            "labels": sentence[1:]
        }
    def get_length(self,idx):
        return len(self.tokens[idx])

class BucketSampler(Sampler):
    def __init__(self,
                 dataset,
                 max_len,
                 shuffle = True,
                 ):
        self.dataset = dataset
        self.max_len = max_len
        self.shuffle = shuffle

        self.indices = list(range(len(dataset)))

    def __iter__(self):
        indices = sorted(self.indices,
                         key = lambda i:
                            self.dataset.get_length(i))

        batches = []

        for i in range(0,len(indices),self.max_len):
            batch = indices[i: i + self.max_len]
            batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

class TokenBatchSampler(Sampler):
    def __init__(self,
                 dataset,
                 max_token,
                 shuffle = True):
        self.dataset = dataset
        self.max_token = max_token
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        indices = sorted(self.indices,
                         key = lambda i:
                            self.dataset.get_length(i))

        batches = []
        current_batch = []
        current_token = 0

        for idx in indices:
            length = self.dataset.get_length(idx)
            if(length + current_token > self.max_token):
                batches.append(current_batch)
                batches = []
                current_toke = 0
            current_batch.append(i)
            current_token += length

        if len(current_batch) > 0:
            batches.append(current_batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

class PaddingCollator:
    def __init__(self,
            token_pad_ids = 0,
            label_pad_ids = -100,):
        self.token_pad_ids = token_pad_ids
        self.label_pad_ids = label_pad_ids

    def __call__(self, batch):
        batch_size = len(batch)
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = torch.fill(
            (batch_size, max_len),
            self.token_pad_ids,
            dtype = torch.int64,
        )

        label_ids = torch.fill(
            (batch_size, max_len),
            self.label_pad_ids,
            dtype = torch.int64,
        )

        pad_mask = torch.zeros(
            batch_size,
            max_len,
            dtype = torch.int64,
        )

        for i, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[i, :length] = item["input_ids"]
            label_ids[i, :length] = item["labelss"]
            pad_mask[i, : length] = 1


        return {
            "input_ids": input_ids,
            "labels": label_ids,
            "pad_mask": pad_mask,
        }





    


