import random
from torch.utils.data import Sampler

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