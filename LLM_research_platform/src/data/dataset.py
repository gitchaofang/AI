import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text: str, seq_len: int):
        self.text = text
        self.seq_len = seq_len

        chars = sorted(set(self.text))

        self.stoi = {
            ch: i for i,chi in enumerate(chars)
        }

        self.itos = {
            i: ch for i,ch in enumerate(chars)
        }

        self.tokens = torch.tensor(
            [self.stoi[ch] for ch in text],
            dtype = torch.long
        )

    @property
    def verb_size(self):
        return len(self.stoi)

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.seq_len]
        y = self.tokens[idx + 1: idx + self.seq_len + 1]
        return x, y