import torch
import random
from torch.utils.data import Sampler

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