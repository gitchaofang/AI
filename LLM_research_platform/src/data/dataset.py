import torch
from torch.utils.data import Dataset
from src.data.simple_tokenizer import SimpleTokenizer

class VariableLengthDataset(Dataset): # for txt file
    def __init__(self, file_name):
        self.file_name = file_name
        print(f"dataset is: {file_name}")
        sentences = self._data_prep()
        tokenizer = SimpleTokenizer("novel.txt")
        token_map = tokenizer.get()
        self.stoi = token_map["stoi"]
        self.itos = token_map["itos"]

        self.tokens = []
        cnt = 0
        for sentence in sentences:
            cnt += len(sentence)
            current_sentence = [self.stoi[char] for char in sentence]
            self.tokens.append(current_sentence) 
        print(f"sample size is: {cnt}")
        print(f"token size = {len(self.tokens)}")
        print(f"token number = {len(self.stoi)}")
        print("\n")

    def _data_prep(self): # read data then return a list of lis (a batch of data)
        #path = "/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/" + self.file_name # for local
        path = "/content/AI/LLM_research_platform/src/data/input_data/data/" + self.file_name # for codlab online
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        out = []
        current_batch = []
        current_len = 0
        for char in text:
            current_batch.append(char)
            current_len += 1
            if current_len > 200 and (char == '.' or char == '\n'):
    #        if current_len == 256:
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