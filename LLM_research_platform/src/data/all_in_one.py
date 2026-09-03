import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from src.data.tokenizer import Tokenizer
# data dir: /Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data

class VariableLengthDataset(Dataset): # for txt file
    def __init__(self, file_name):
        self.file_name = file_name
        sentences = self._data_prep()
        tokenizer = Tokenizer(file_name)
        token_map = tokenizer.get()
        self.stoi = token_map["stoi"]
        self.itos = token_map["itos"]

        self.tokens = []
        for sentence in sentences:
            current_sentence = [self.stoi[char] for char in sentence]
            self.tokens.append(current_sentence) 
        print(f"batch size = {len(self.tokens)}")
        print(f"token number = {len(self.stoi)}")

    def _data_prep(self): # read data then return a list of lis (a batch of data)
        #path = "/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/" + self.file_name # for local
        path = "/content/AI/LLM_research_platform/src/data/input_data/data/" + self.file_name # for codlab online
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        out = []
        current_batch = []
        current_len = 0
        for char in text:
            current_batch.append(char)
            current_len += 1
            if current_len == 128:
                out.append(current_batch)          
                current_len = 0
                current_batch = []
        return out
        


    def __len__(self):
        return len(self.tokens)
    def __getitem__(self,idx):
        sentence = torch.tensor(self.tokens[idx])  # Convert to tensor here
        return {
            "input_ids": sentence[:-1],
            "labels": sentence[1:]
        }
    def get_vocab_size(self):
        return len(self.stoi)
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
                 batch_size = 2,
                 shuffle = True):
        self.dataset = dataset
        self.max_token = max_token
        self.shuffle = shuffle
        self.batch_size = batch_size
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
            if(length + current_token > self.max_token or len(current_batch) >= self.batch_size):
                batches.append(current_batch)
                current_batche = []
                current_token = 0
            current_batch.append(idx)
            current_token += length

        if len(current_batch) > 0:
            batches.append(current_batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch
            
    def __len__(self):
        indices = sorted(self.indices,
                     key = lambda i: self.dataset.get_length(i))
    
        num_batches = 0
        current_token = 0
    
        for idx in indices:
            length = self.dataset.get_length(idx)
            if length + current_token > self.max_token and current_token > 0:
                num_batches += 1
                current_token = 0
            current_token += length
    
        if current_token > 0:
            num_batches += 1
    
        return num_batches

class PaddingCollator:
    def __init__(self,
            token_pad_ids = 0,
            label_pad_ids = -100,):
        self.token_pad_ids = token_pad_ids
        self.label_pad_ids = label_pad_ids

    def __call__(self, batch):
        batch_size = len(batch)
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = torch.full(
            (batch_size, max_len),
            self.token_pad_ids,
            dtype = torch.int64,
        )

        label_ids = torch.full(
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
            label_ids[i, :length] = item["labels"]
            pad_mask[i, : length] = 1


        return {
            "input_ids": input_ids,
            "labels": label_ids,
            "pad_mask": pad_mask,
        }

#dataset = VariableLengthDataset("novel.txt")
#print(len(dataset))








