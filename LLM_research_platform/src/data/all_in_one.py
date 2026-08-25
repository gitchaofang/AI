import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import Sampler

class variable_length_dataset(Dataset):
    def __init__(self,sentences):
        self.sentences = sentences
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        return{
            "input_ids": sentence[:-1],
            "labels": sentence[1:]
        }
    def get_length(self,idx):
        return len(self.sentences[idx])

class BucketBatchSampler(Sampler):
    def __init__(self,
                 dataset,
                 max_seq_len,
                 shuffle = True
                 ):
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle

        indices = list(range(len(dataset)))
    def __iter__(self):
        indices = sorted(
            self.indices,
            key=lambda i:
                self.dataset.get_length(i)
        )

        batches = []

        for i in range(0,len(indices),self.max_seq_len):
            batch = indices[i:i + self.max_seq_len]
            batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

class TokenBatchSampler(Sampler):
    def __init__(self,
                 dataset,
                 max_token):
        self.dataset = dataset
        self.max_token = max_token

        self.indices = list(range(len(dataset)))

    def __iter__(self):
        indices = sorted(self.indices,
            key=lambda i:
                self.dataset.get_length(i)
        )
        
        batches = []
        current_batch = []
        current_token = 0
        for idx in indices:
            token_len = self.dataset.get_length(idx)
            if(current_token + token_len > self.max_token and len(current_batch) != 0):
                batches.append(current_batch)
                current_battch = []
                current_token = 0
            current_batch.append(idx)
            current_token += token_len
        if(len(current_batch) > 0):
            batches.append(current_batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch




class PaddingCollator:
    def __init__(self,
                 pad_token_id,
                 pad_label_id):
        self.pad_token_id = pad_token_id
        self.pad_label_id = pad_label_id

    def __call__(self, batch):
        max_seq_len = max(
            len(x["input_ids"])
            for x in batch            
        )

        input_ids = torch.fill(
            (len(batch),max_seq_len),
            self.pad_token_id,
            dtype = torch.int64,
        )

        labels = torch.fill(
            (len(batch),max_seq_len),
            self.pad_label_id,
            dtype = torch.int64
        )

        attention_mask = torch.zeros(
            len(batch),
            max_seq_len,
            dtypq = torch.int64
        )

        for i, item in enumerate(batch):
            input = item["input_ids"]
            label = item["labels"]
            length = len(input)

            input_ids[i, : length] = input
            labels[i, :length] = label
            attention_mask[i, :length] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    


