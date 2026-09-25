import torch
import random
import json
import numpy as np
import os
from pathlib import Path
from PIL import Image
from .helper import patchify
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler
from src.data.regex_tokenizer import RegexTokenizer

LOCAL_FILE_PATH =  Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/")
LOCAL_INDEX_PATH =  Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/index_files")
COLAB_FILE_PATH = Path("/content/AI/LLM_research_platform/src/data/input_data/data/")
COLAB_INDEX_PATH = Path("/content/AI/LLM_research_platform/src/data/input_data/data/index_files")
#seperation pattern
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class TextDecodeDataset(Dataset): # for txt file
    def __init__(self, tokenizer, max_len, filename):
        print(f"build dataset")
        self.file_dir = COLAB_FILE_PATH
        self.index_dir = COLAB_INDEX_PATH
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
        print(f"tokens size: {len(encoded_tokens)}")
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

        

class TokenBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = list(range(len(dataset)))
        self.batches = []

        self._make_batches()

    def _make_batches(self):
        indices = sorted(
            self.indices,
            key=lambda i: self.dataset.get_length(i)
        )

        for idx in range(0, len(indices), self.batch_size):
            self.batches.append(
                indices[idx:idx + self.batch_size]
            )
        print(f"batches size is: {len(self.batches)}")
        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


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

            #-----------   image ------------

# load config file
LOCAL_YAML_PATH = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/vit.yaml")
COLAB_YAML_PATH = Path("/content/AI/LLM_research_platform/configs/vit.yaml")
# load yaml config
with open(COLAB_YAML_PATH,"r") as f:
    vit_config = yaml.safe_load(f)

EXTRACTED_PATH  = Path(vit_config["data"]["extracted_path"]) 
SHARED_PATH = Path(vit_config["data"]["shared_path"])

class ImageTextEncode(Dataset):
    def __init__(self, file_dir=EXTRACTED_PATH, tokenizer = None, image_only = vit_config["data"]["image_only"]):
        self.file_dir = Path(file_dir)
        self.index_path = self.file_dir / "index.json"
        self.transform = transforms.ToTensor()
        self.image_only =  image_only
        self.tokenizer = tokenizer
        assert (self.tokenizer is None and self.image_only) or (self.tokenizer is not None and not self.image_only)

        # Load existing index: [image name stems]
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
        patchify_res = patchify(image = image) 

        # images
        patches = patchify_res["patches"]          # [N,  C * patch_size * patch_size]
        patch_positions = patchify_res["positions"]   # [N, 2]

        # meta data
        with open(meta_data_path, "r") as f:
            meta_data = json.load(f)

         # process text
        if not self.image_only:
            text = meta_data["caption"]
            encoded_tokens = self.tokenizer.encode(text)
            all_tokens = [item for token_list in encoded_tokens for item in token_list]
            return {
                "patches": patches,            # [N,C * patch_size * patch_size]
                "patch_positions": patch_positions,    # [N, 2]
                "caption_ids": all_tokens,     # [len(all_tokens)]
                "meta_data": meta_data,        # "caption", "url", "key", "status", "error_message", "width", "height", "exif", "original_width", "original_height"
            }

        # if only image is needed
        return {
            "patches": patches, #[N,C * patch_size * patch_size]
            "patch_positions": patch_positions, # [N, 2]
            "meta_data": meta_data, #json: "caption", "url", "key", "status", "error_message", "width", "height", "exif", "original_width", "original_height"
        } 
    def get_length(self):
        return len(self.stems)

class ImageDatasetBatchSampler(BatchSampler):

    def __init__(self, dataset, batch_size, shuffle=vit_config["data"]["shuffle"]):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            yield batch

    def __len__(self):

        return (
            len(self.dataset) + self.batch_size - 1
        ) // self.batch_size
    
class VitCollator:
    def __init__(self, token_pad = 0, image_pad = 0.0,label_pad = -100, image_only = vit_config["data"]["image_only"]):
        self.token_pad = token_pad
        self.image_pad = image_pad
        self.label_pad = label_pad
        self.image_only = image_only

    def __call__(self, batch):
        batch_size = len(batch)
        max_len_patch = max(len(x["patches"]) for x in batch)
        patch_d = vit_config["data"]["color"]*vit_config["data"]["patch_size"]*vit_config["data"]["patch_size"]

        patched_input = torch.full(
            (batch_size, max_len_patch, patch_d),
            self.image_pad,
            dtype = batch[0]["patches"].dtype,
        )

        if not self.image_only:
            max_len_text = max(len(x["caption_ids"]) for x in batch)
            caption_ids = torch.full(
                (batch_size, max_len_text),
                self.token_pad,
                dtype = torch.int64,
            )
            pad_mask_text = torch.zeros(
                batch_size,
                max_len_text,
                dtype = torch.int64,
            )

        patch_positions = torch.zeros(
            batch_size,
            max_len_patch,
            2,
            dtype = torch.int64,
        )
        
        pad_mask_patch = torch.zeros(
            batch_size,
            max_len_patch,
            dtype = torch.int64,
        )

        meta_data = [] # list of dict

        for i, item in enumerate(batch):
            meta_data.append(item["meta_data"])
            # patches
            length_patches = len(item["patches"]) 
            patched_input[i,:length_patches] = item["patches"] # patched input
            patch_positions[i,:length_patches] = item["patch_positions"] # patch coordinates for RoPE
            pad_mask_patch[i,:length_patches] = 1 # pad maskes for patched input

            # text
            if not self.image_only:
                length_text = len(item["caption_ids"])
                caption_ids[i,:length_text] = item["caption_ids"]
                pad_mask_text[i,:length_text] = 1
           

        if not self.image_only:
            return {
                "patched_input": patched_input, #[B, max_len_patch ,C * patch_size * patch_size]
                "patch_positions": patch_positions, # [B, max_len_patch,2]
                "pad_mask_patch": pad_mask_patch,# [B, max_len_patch]
                "caption_ids": caption_ids, # [B, max_len_text]
                "pad_mask_text": pad_mask_text, # [B, max_len_text]
                "meta_data": meta_data, # list of dict. B dicts
            }

        return {
            "patched_input": patched_input, #[B, max_len_patch ,C * patch_size * patch_size]
            "patch_positions": patch_positions, # [B, max_len_patch, 2]
            "pad_mask_patch": pad_mask_patch,# [B, max_len_patch]
            "meta_data": meta_data, # list of dict. B dicts
        } 
    """
        these are the data shape before going to the model:
            patched_input: [B, N_max, C*P*P]
            patch_positions: [B, N_max, 2]
            pad_mask_patch: [B, N_max]
            caption_ids: [B, T_max]
            pad_mask_text: [B, T_max]
            meta_data: list[B]
    """