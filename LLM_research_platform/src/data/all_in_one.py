import torch
import random
from totch.utils.data import Dataset
from torch.utils.data import Sampler

class VariableLengthDataset(Dataset):
    def __init__(self,sentences):
        self.sentences = sentences
    def __len__(self)
        return len(self.dataset)
    def __getitem__(self,idx):
        sentence = self.sentences[idx]
        return{
            "input_id": sentence[:-1],
            "labels": sentence[1:]
        }
    def get_length(self,idx):
        return len(self.sentences[idx])

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
                 shuffle = True);
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



    


