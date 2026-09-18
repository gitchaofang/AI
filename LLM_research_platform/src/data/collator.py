import torch

class PaddingCollator:
    def __init__(self,token_pad = 0, label_pad = -100,):

        self.token_pad = token_pad
        self.label_pad = label_pad

    def __call__(self, batch):
        batch_size = len(batch)
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = torch.full(
            (batch_size,max_len),
            self.token_pad,
            dtype = torch.int64,
        )

        label_ids = torch.full(
             (batch_size,max_len),
             self.label_pad,
             dtype = torch.int64,
        )

        pad_mask = torch.zeros(
            batch_size,
            max_len,
            dtype = torch.int64,
        )

        # create 1d positions for text
        positions = torch.arange(max_len,dtype=torch.int64)[None,:, None].expand(batch_size, max_len, 1).clone()


        for i, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[i][:length] = batch[i]["input_ids"]
            label_ids[i][:length] = batch[i]["labels"]
            pad_mask[i][:length] = 1

        return {
            "input_ids": input_ids,
            "labels": label_ids,
            "pad_mask": pad_mask,
            "positions": positions,
        }