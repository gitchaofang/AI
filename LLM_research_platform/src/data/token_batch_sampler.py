import torch
import random
from torch.utils.data import Sampler

class TokenBatchSampler(Sampler):
    def __init__(self,
                 dataset,
                 max_token,
                 batch_size = 4,
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