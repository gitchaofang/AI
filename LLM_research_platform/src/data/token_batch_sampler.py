import torch
from torch.utils.data import Sampler

class TokenBatchSampler(Sampler):
    def __init__(self,
                 dataset,
                 max_token,
                 shuffle = True):
        self.dataset = dataset
        self.max_token = max_token
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
    def __len__(self):
        return len(self.dataset)

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
                current_token = 0
            current_batch.append(idx)
            current_token += length

        if len(current_batch) > 0:
            batches.append(current_batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch