import torch

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
