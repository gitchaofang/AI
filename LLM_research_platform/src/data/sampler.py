import random
from torch.utils.data import Sampler

class BucketBatchSampler(Sampler):
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle=True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indicex = list(range(len(dataset)))

    def __iter__(self):
        indices = sorted(
            self.indices,
            key=lambda i:
                self.dataset.get_length(i)
        )

        # Divide into batches
        batches = []

        for i in range(0,len(indices), self.batch_size,):
            batch = indices[i: i + self.batch_size]
            batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)
    
        for batch in batches:
            yield batch

    def __len__(self):
        return(len(self.indices) + self.batch_size - 1) // self.batch_size