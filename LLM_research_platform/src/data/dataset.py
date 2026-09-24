import torch
import os
import json
import yaml
from PIL import Image
from torchvision import transforms
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset
from src.data.simple_tokenizer import SimpleTokenizer

# load config file
LOCAL_YAML_PATH = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/gpt.yaml")
COLAB_YAML_PATH = Path("/content/AI/LLM_research_platform/configs/gpt.yaml")
# load yaml config
with open(COLAB_YAML_PATH,"r") as f:
    gpt_config = yaml.safe_load(f)

LOCAL_FILE_PATH =  Path(gpt_config["data"]["local_file_path"])
LOCAL_INDEX_PATH =  Path(gpt_config["data"]["local_index_path"])
COLAB_FILE_PATH = Path(gpt_config["data"]["colab_file_path"])
COLAB_INDEX_PATH = Path(gpt_config["data"]["colab_index_path"])

# text
class TextDecodeDataset(Dataset): # for txt file
    def __init__(self, tokenizer, max_len, filename):
        self.file_dir = LOCAL_FILE_PATH
        self.index_dir = LOCAL_INDEX_PATH
        self.filename = filename
        self.max_len = max_len
        self.tokenizer = tokenizer

        read_path = self.file_dir / filename

        stem = Path(filename).stem

        self.token_path = self.index_dir / f"{stem}_{max_len}_tokens.bin"
        self.index_path = self.index_dir / f"{stem}_{max_len}_index.npy"



        with open(read_path, "r", encoding="utf-8") as f:
            self.text = f.read()
        if not (os.path.exists(self.token_path) and os.path.exists(self.index_path)):
            self.data_prep()

        self.tokens = np.memmap(
             self.token_path,
             dtype = np.int32,
             mode="r"
        )
        self.index = np.load(
             self.index_path,
             mmap_mode = "r"
        )

    def data_prep(self):
        encoded_tokens = self.tokenizer.encode(self.text)
        all_tokens = [item for token_list in encoded_tokens for item in token_list]
        all_token_len = len(all_tokens)
        all_index = []
        offset = 0

        while offset < all_token_len:
            end = min(offset + self.max_len, all_token_len)
            all_index.append([offset, end - offset])
            offset = end

        # Convert to NumPy
        all_tokens = np.asarray(all_tokens,dtype=np.int32)
        index = np.asarray(all_index,dtype=np.int64)

        # Save
        all_tokens.tofile(self.token_path)
        np.save(self.index_path, index)

    def __len__(self):
         return len(self.index)
    
    def __getitem__(self, key):
         offset,length = self.index[key]
         tokens = self.tokens[offset: offset + length]
         data = torch.tensor(tokens, dtype = torch.int64)
         return {
              "input_ids": data[:-1],
              "labels": data[1:]
         }

    def get_length(self, key):
         return self.index[key][1]

# -------------------------------------------------
# image dala loading
# I use google drive as disk for store data. The path is:
# extracted data: /content/drive/MyDrive/VLM_DATA/CC3M/normal 
# shareds data: /content/drive/MyDrive/VLM_DATA/CC3M/shards
# index_map{index, image-data name}
# use inex_map to fetch data from google drive during training.

# load config file
LOCAL_YAML_PATH = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/vit.yaml")
COLAB_YAML_PATH = Path("/content/AI/LLM_research_platform/configs/vit.yaml")
# load yaml config
with open(COLAB_YAML_PATH,"r") as f:
    vit_config = yaml.safe_load(f)

EXTRACTED_PATH  = Path(vit_config["data"]["extracted_path"]) 
SHARED_PATH = Path(vit_config["data"]["shared_path"])

class ImageTextEncode(Dataset):
    def __init__(self, file_dir=EXTRACTED_PATH):
        self.file_dir = Path(file_dir)
        self.index_path = self.file_dir / "index.json"
        self.transform = transforms.ToTensor()

        # Load existing index
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                self.stems = json.load(f)
        # Build index if it doesn't exist
        else:
            self.stems = []
            for file_path in sorted(self.file_dir.glob("*.txt")):
                self.stems.append(file_path.stem)
            with open(self.index_path, "w") as f:
                json.dump(self.stems, f)

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, key):
        assert 0 <= key < len(self.stems), f"key {key} is out of range"
        stem = self.stems[key]
        image_path = self.file_dir / f"{stem}.jpg"
        meta_data_path = self.file_dir / f"{stem}.json"

        # Load image
        image = Image.open(image_path).convert("RGB")
        # PIL → Tensor
        image = self.transform(image)
        # Load metadata
        with open(meta_data_path, "r") as f:
            meta_data = json.load(f)
        return {
            "image": image, #[B,C,H,W]
            "meta_data": meta_data, #"caption", "url", "key", "status", "error_message", "width", "height", "exif", "original_width", "original_height"
        }
    def get_length(self):
        return len(self.stems)
