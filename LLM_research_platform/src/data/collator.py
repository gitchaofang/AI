import torch
class PaddingCollator:
    def __init__(self, 
                 pad_token_id = 0,
                 label_pad_token_id = -100,):
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self,batch):
        max_len = max(
            len(item["input_ids"]) 
            for item in batch
        )

        batch_size = len(batch)

        input_ids = torch.fill(
            (batch_size, max_len),
            self.pad_token_id,
            dtype = torch.int64,
        )

        labels = torch.full(
            (batch_size,max_len),
            self.label_pad_token_id,
            dtype=torch.int64,
        )

        attention_mask = torch.zeros(
            batch_size,
            max_len,
            dype = torch.int64,
        )

        for i, item in enumerate(batch):
            input_ids_i = item["input_ids"]
            labels_i = item["labels"]

            length = len(input_ids_i)

            input_ids[i, :length] = input_ids_i
            labels[i, :length] = labels_i 

            attention_mask[i, :length] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }