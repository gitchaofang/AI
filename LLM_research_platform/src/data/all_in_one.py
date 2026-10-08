import torch
import random
import json
import numpy as np
import yaml
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
        all_token_len = len(encoded_tokens)
        all_index = []
        offset = 0

        while offset < all_token_len:
            end = min(offset + self.max_len, all_token_len)
            all_index.append([offset, end - offset])
            offset = end

        # Convert to NumPy
        all_tokens = np.asarray(encoded_tokens,dtype=np.int32)
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

# -------------------------------------------------
# image data loading
# use google drive as disk for store data. The path is:
# extracted data: /content/drive/MyDrive/VLM_DATA/CC3M/normal 
# shareds data: /content/drive/MyDrive/VLM_DATA/CC3M/shards
# use inex_map to fetch data from google drive during training: index_map{index, image-data name}

# load config file
LOCAL_YAML_PATH = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/vit.yaml")
COLAB_YAML_PATH = Path("/content/AI/LLM_research_platform/configs/vit.yaml")
CACHE_RESERVE_GB = 50
# load yaml config
with open(COLAB_YAML_PATH,"r") as f:
    vit_config = yaml.safe_load(f)

class ImageDataset(Dataset):
    def __init__(self, data_dir, tokenizer = None, image_only = vit_config["data"]["image_only"], for_training=True):
        self.data_dir = Path(data_dir)
        if for_training:
            self.index_path = self.data_dir/f"training"/f"index.json"
        else:
            self.index_path = self.data_dir/f"validation"/f"index.json"
        self.transform = transforms.ToTensor()
        self.image_only =  image_only
        self.tokenizer = tokenizer
        self.colab_cache_path = Path("/content/cache")
        self.colab_cache_path.mkdir(parents=True, exist_ok=True)
        self.drive_cache_path = self.data_dir / "cache"
        self.drive_cache_path.mkdir(parents=True, exist_ok=True)
        self.for_training = for_training
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

    def _atomic_write_bytes(self,path, data):
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, path)


    def _atomic_write_text(self, path, text, encoding="utf-8"):
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with open(tmp_path, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp_path, path)


    def _atomic_write_json(self, path, data):
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)

    def __len__(self):
        return len(self.id_tar_pair)

    def __getitem__(self, key):
        assert 0 <= key < len(self.id_tar_pair), f"key {key} is out of range"
        sample_id, tar_name = self.id_tar_pair[key]
        shard_type = Path(tar_name).stem.split("-")[1]
        if self.for_training:
            tar_path = self.data_dir / "training" / tar_name
        else:
            tar_path = self.data_dir / "validation" / tar_name

        image_name = f"{sample_id}.jpg"
        if shard_type == "train":
            meta_data_name = f"{sample_id}.json"
        elif shard_type == "validation":
            text_data_name = f"{sample_id}.txt"

        '''
            Load image and metadata with 3 options (only one will be applied):
                1. check if image and metadata can be loaded from colab cache
                2. check if image and metadata can be loaded from google drive cach
                3. It not cached, extract them from .tar shard
        '''
 
        # Option 1: check if image and metadata can be loaded from colab cache
        colab_image_path = self.colab_cache_path/f"{image_name}"
        drive_image_path = self.drive_cache_path/f"{image_name}"
        if shard_type == "train":
            colab_meta_path = self.colab_cache_path/f"{meta_data_name}"
            drive_meta_path = self.drive_cache_path/f"{meta_data_name}"
        elif shard_type == "validation":
            colab_text_path = self.colab_cache_path/f"{text_data_name}"
            drive_text_path = self.drive_cache_path/f"{text_data_name}"

        # -------------------------------------------------
        # Determine cache hits
        # -------------------------------------------------
        if shard_type == "train":
            colab_cache_hit = (colab_image_path.is_file() and colab_meta_path.is_file())
            drive_cache_hit = (drive_image_path.is_file() and drive_meta_path.is_file())
        elif shard_type == "validation":
            colab_cache_hit = (colab_image_path.is_file() and colab_text_path.is_file())
            drive_cache_hit = (drive_image_path.is_file() and drive_text_path.is_file())

        # -------------------------------------------------
        # Option 1: Colab cache
        # -------------------------------------------------
        if colab_cache_hit:
            with open(colab_image_path, "rb") as f:
                image = Image.open(f).convert("RGB")
            if shard_type == "train":
                with open(colab_meta_path, "r") as f:
                    meta_data = json.load(f)
            elif shard_type == "validation":
                with open(colab_text_path, "r", encoding="utf-8") as f:
                    text_data = f.read()
        # -------------------------------------------------
        # Option 2: Google Drive cache
        # -------------------------------------------------
        elif drive_cache_hit:
            with open(drive_image_path, "rb") as f:
                image = Image.open(f).convert("RGB")
            if shard_type == "train":
                with open(drive_meta_path, "r") as f:
                    meta_data = json.load(f)
            elif shard_type == "validation":
                with open(drive_text_path, "r", encoding="utf-8") as f:
                    text_data = f.read()
        # -------------------------------------------------
        # Option 3: Load from .tar
        # -------------------------------------------------
        else:
            with tarfile.open(tar_path, "r") as tar:
                image_file = tar.extractfile(image_name)
                if shard_type == "train":
                    meta_file = tar.extractfile(meta_data_name)
                elif shard_type == "validation":
                    text_file = tar.extractfile(text_data_name)
                # Check if any of image_file and
                # text_file is None
                if image_file is None:
                    raise FileNotFoundError(f"{image_name} not found in {tar_path}")
                if (shard_type == "train" and meta_file is None):
                    raise FileNotFoundError(f"{meta_data_name} not found in {tar_path}")
                elif (shard_type == "validation"and text_file is None):
                    raise FileNotFoundError(f"{text_data_name} not found in {tar_path}")
                
                # read image
                image_bytes = image_file.read()
                image = Image.open(BytesIO(image_bytes)).convert("RGB")

                # read metadata or txt
                if shard_type == "train":
                    meta_data = json.load(meta_file)
                elif shard_type == "validation":
                    text_data = (text_file.read().decode("utf-8"))
                # Free disk space on colab (G)
                free_colab = (shutil.disk_usage(self.colab_cache_path).free / 1024**3)

                # cache image and metadata.
                # Leave 50G on each disk
                required_gb = (len(image_bytes) / 1024**3)
                if (free_colab - required_gb > CACHE_RESERVE_GB):
                    self._atomic_write_bytes(colab_image_path, image_bytes)
                    if shard_type == "train":
                        self._atomic_write_json(colab_meta_path, meta_data)
                    elif shard_type == "validation":
                        self._atomic_write_text(colab_text_path, text_data)
                else:
                    # Free disk space on google drive (G)
                    free_drive = (shutil.disk_usage(self.drive_cache_path).free / 1024**3)
                    if (free_drive - required_gb > CACHE_RESERVE_GB):
                        self._atomic_write_bytes(drive_image_path, image_bytes)
                        if shard_type == "train":
                            self._atomic_write_json(drive_meta_path, meta_data)
                        elif shard_type == "validation":
                            self._atomic_write_text(drive_text_path, text_data)

        '''
            Read and patchfy image
            Process text caption
        '''
        image = self.transform(image)
        # pachify
        patchify_res = patchify(image = image) 
        patches = patchify_res["patches"]          # [N,  C * patch_size * patch_size]
        patch_positions = patchify_res["positions"]   # [N, 2]
                                
        # process text
        if not self.image_only:
            if shard_type == "train":
                text = meta_data["caption"]
            elif shard_type == "validation":
                text = text_data
            encoded_tokens = self.tokenizer.encode(text)
            all_tokens = [item for token_list in encoded_tokens for item in token_list]
            caption_ids = torch.tensor(all_tokens,dtype=torch.int64)
            # if caption_ids has less than 2 tokens, causal LLM can't work
            if len(caption_ids) < 2:
                raise ValueError(f"Sample {sample_id} has fewer than 2 tokens")
            if shard_type == "train":
                return {
                    "patches": patches,            # [N, C * patch_size * patch_size]
                    "patch_positions": patch_positions,    # [N, 2]
                    "caption_ids": caption_ids[:-1],     # [len(all_tokens) - 1]
                    "caption_ids_label": caption_ids[1:,],   # [len(all_tokens) - 1]
                    "meta_data": meta_data,        # "caption", "url", "key", "status", "error_message", "width", "height", "exif", "original_width", "original_height"
                }
            elif shard_type == "validation":
                return {
                    "patches": patches,            # [N, C * patch_size * patch_size]
                    "patch_positions": patch_positions,    # [N, 2]
                    "caption_ids": caption_ids[:-1],     # [len(all_tokens) - 1]
                    "caption_ids_label": caption_ids[1:,],   # [len(all_tokens) - 1]
                }

        # if only image is needed
        return {
            "patches": patches, #[N,C * patch_size * patch_size]
            "patch_positions": patch_positions, # [N, 2]
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


class ImageDatasetBatchSampler(BatchSampler):

    def __init__(self, dataset, batch_size, shuffle=True):
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
        return (len(self.dataset) + self.batch_size - 1) // self.batch_sizee
    
class VitCollator:
    def __init__(self, token_pad = 0, image_pad = 0.0, label_pad = -100, image_only = False, for_training = True):
        self.token_pad = token_pad
        self.label_pad = label_pad
        self.image_pad = image_pad
        self.image_only = image_only
        self.for_training = for_training

    def __call__(self, batch):
        batch_size = len(batch)
        max_len_patch = max(len(x["patches"]) for x in batch)
        patch_d = batch[0]["patches"].shape[1]

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
            caption_ids_label = torch.full(
                (batch_size,max_len_text),
                self.label_pad,
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

        if self.for_training:
            meta_data = [] # list of dict

        for i, item in enumerate(batch):
            if self.for_training:
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
                caption_ids_label[i, :length_text] = item["caption_ids_label"]
                pad_mask_text[i,:length_text] = 1
          
        if not self.image_only:
            # create 1d positions for text
            positions = torch.arange(max_len_text,dtype=torch.int64).view(1, max_len_text, 1).expand(batch_size, -1, -1)
            res_dict = {
                "patched_input": patched_input, #[B, max_len_patch ,C * patch_size * patch_size]
                "patch_positions": patch_positions, # [B, max_len_patch, 2]
                "pad_mask_patch": pad_mask_patch,# [B, max_len_patch]
                "caption_ids": caption_ids, # [B, max_len_text]
                "caption_ids_label": caption_ids_label, #[B, max_len_text]
                "pad_mask_text": pad_mask_text, # [B, max_len_text]
                "positions_text": positions, #[B, max_len_text,1]
            }
            if self.for_training:
                res_dict["meta_data"] = meta_data # list of dict. B dicts
            return res_dict

        return {
            "patched_input": patched_input, #[B, max_len_patch ,C * patch_size * patch_size]
            "patch_positions": patch_positions, # [B, max_len_patch, 2]
            "pad_mask_patch": pad_mask_patch,# [B, max_len_patch]
        } 