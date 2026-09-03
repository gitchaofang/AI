import torch
from torch.utils.data import Dataset
from src.data.tokenizer import Tokenizer

class VariableLengthDataset(Dataset): # for txt file
    def __init__(self, file_name):
        self.file_name = file_name
        sentences = self._data_prep()
        tokenizer = Tokenizer(file_name)
        token_map = tokenizer.get()
        self.stoi = token_map["stoi"]
        self.itos = token_map["itos"]

        self.tokens = []
        for sentence in sentences:
            current_sentence = [self.stoi[char] for char in sentence]
            self.tokens.append(current_sentence) 
        print(f"batch size = {len(self.tokens)}")
        print(f"token number = {len(self.stoi)}")

    def _data_prep(self): # read data then return a list of lis (a batch of data)
        #path = "/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/" + self.file_name. # for local
        path = "/content/AI/LLM_research_platform/src/data/input_data/data/" + self.file_name # for codlab online
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        out = []
        current_batch = []
        current_len = 0
        for word in text.split():
            current_batch.append(word)
            current_len += 1
            if current_len == 128:
                out.append(current_batch)          
                current_len = 0
                current_batch = []
        return out
        


    def __len__(self):
        return len(self.tokens)
    def __getitem__(self,idx):
        sentence = torch.tensor(self.tokens[idx])  # Convert to tensor here
        return {
            "input_ids": sentence[:-1],
            "labels": sentence[1:]
        }
    def get_vocab_size(self):
        return len(self.stoi)
    def get_length(self,idx):
        return len(self.tokens[idx])