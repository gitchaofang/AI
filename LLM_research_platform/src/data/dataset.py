import torch
import os
import json
import yaml
import tarfile
from PIL import Image
from io import BytesIO
from .helper import patchify
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
# image data loading
# use google drive as disk for store data. The path is:
# extracted data: /content/drive/MyDrive/VLM_DATA/CC3M/normal 
# shareds data: /content/drive/MyDrive/VLM_DATA/CC3M/shards
# use inex_map to fetch data from google drive during training: index_map{index, image-data name}

# load config file
LOCAL_YAML_PATH = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/vit.yaml")
COLAB_YAML_PATH = Path("/content/AI/LLM_research_platform/configs/vit.yaml")
# load yaml config
with open(COLAB_YAML_PATH,"r") as f:
    vit_config = yaml.safe_load(f)

class ImageTextEncode(Dataset):
    def __init__(self, data_dir, tokenizer = None, image_only = vit_config["data"]["image_only"]):
        self.data_dir = Path(data_dir)
        self.index_path = self.data_dir/f"index.json"
        self.transform = transforms.ToTensor()
        self.image_only =  image_only
        self.tokenizer = tokenizer
        assert (self.tokenizer is None and self.image_only) or (self.tokenizer is not None and not self.image_only)

        '''
        Load index dict:
            key: sample name
            value: address in google drive
        '''
        
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                index = json.load(f)
        # Build index if it doesn't exist
        else:
            raise FileNotFoundError(
                f"Index file does not exist: {self.index_path}"
            )
        # build a list of tuples(file_name ("00015"), tar_file_name("cc3m-train_0565"))
        self.id_tar_pair = list(index.items())

    def __len__(self):
        return len(self.id_tar_pair)

    def __getitem__(self, key):
        assert 0 <= key < len(self.id_tar_pair), f"key {key} is out of range"
        sample_id, tar_name = self.id_tar_pair[key]
        tar_path = self.data_dir / "training" / tar_name
        image_name = f"{sample_id}.jpg"
        meta_data_name = f"{sample_id}.json"

        # Load image and meta_data
        with tarfile.open(tar_path, "r") as tar:
            image_file = tar.extractfile(image_name)
            text_file = tar.extractfile(meta_data_name)
            # Check if any of image_file and text_file is None
            if image_file is None:
                raise FileNotFoundError(f"{image_name} not found in {tar_path}")
            if text_file is None:
                raise FileNotFoundError(f"{meta_data_name} not found in {tar_path}")

            image_bytes = image_file.read()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            meta_data = json.loads(text_file.read().decode("utf-8"))

        # PIL → Tensor
        image = self.transform(image)
        patchify_res = patchify(image = image) 

        # images
        patches = patchify_res["patches"]          # [N,  C * patch_size * patch_size]
        patch_positions = patchify_res["positions"]   # [N, 2]
                                
        # process text
        if not self.image_only:
            text = meta_data["caption"]
            encoded_tokens = self.tokenizer.encode(text)
            all_tokens = [item for token_list in encoded_tokens for item in token_list]
            caption_ids = torch.tensor(all_tokens,dtype=torch.int64)
            return {
                "patches": patches,            # [N, C * patch_size * patch_size]
                "patch_positions": patch_positions,    # [N, 2]
                "caption_ids": caption_ids,     # [len(all_tokens)]
                "meta_data": meta_data,        # "caption", "url", "key", "status", "error_message", "width", "height", "exif", "original_width", "original_height"
            }

        # if only image is needed
        return {
            "patches": patches, #[N,C * patch_size * patch_size]
            "patch_positions": patch_positions, # [N, 2]
            "meta_data": meta_data, #json: "caption", "url", "key", "status", "error_message", "width", "height", "exif", "original_width", "original_height"
        } 
    def get_length(self):
        return len(self.id_tar_pair)
    
    """
    these are the data shape before going to the model:
        patched_input: [B, N_max, C*P*P]
        patch_positions: [B, N_max, 2]
        pad_mask_patch: [B, N_max]
        caption_ids: [B, T_max]
        pad_mask_text: [B, T_max]
        meta_data: list[B]
    """
