import torch
class PaddingCollator:
    def __init__(self, pad_token_id = 0):
        self.pad_token_id = pad_token_id

    def __call__(self,batch):
        max_len = max(
            len(x) for x in batch
        )

        input_ids = torch.fill(
            (len(batch), max_len),
            self.pad_token_id,
            dtype = torch.int64,
        )

        attention_mask = torch.zeros(
            len(batch),
            max_len,
            dype = torch.int64,
        )

        for i, sequence in enumerate(batch):
            length = len(sequence)
            input_ids[i, :length] = sequence
            attention_mask[i, :length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }