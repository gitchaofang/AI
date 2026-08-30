import torch
from torch.utils.data import Dataset

class VariableLengthDataset(Dataset):
    def __init__(self,sentences):
        self.sentences = sentences
        chars = set()
        for sentence in sentences:
            chars.update(sentence)
        self.char_set = sorted(chars)
        self.stoi = {
            char: i + 1 for i, char in enumerate(self.char_set)
        }
        self.itos = {
            i + 1: char for i, char in enumerate(self.char_set)
        }

        self.tokens = []

        for sentence in sentences:
            current_sentence = [
                self.stoi[char]
                for char in sentence
            ]
            self.tokens.append(current_sentence)   

    def __len__(self):
        return len(self.tokens)
    def __getitem__(self,idx):
        sentence = torch.tensor(self.tokens[idx])  # Convert to tensor here
        return {
            "input_ids": sentence[:-1],
            "labels": sentence[1:]
        }
    def get_length(self,idx):
        return len(self.tokens[idx])