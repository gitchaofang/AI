import torch
import random
from torch.utils.data import BatchSampler

class ImageDatasetBatchSampler(BatchSampler):

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            yield batch

    def __len__(self):

        return (
            len(self.dataset) + self.batch_size - 1
        ) // self.batch_size