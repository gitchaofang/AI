import torch
from totch.utils.data import Dataset

class VariableLengthDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
        

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return {
            "input_ids": sequence[:-1],
            "labels": sequence[1:]
        }
    def get_length(self,idx):
        return len(self.sequences[idx]) - 1
