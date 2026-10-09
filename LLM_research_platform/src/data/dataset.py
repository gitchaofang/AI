import torch
import os
import uuid
import json
import yaml
import tarfile
import shutil
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
    def __init__(self, data_dir, saved_samples_set,tokenizer = None, image_only = vit_config["data"]["image_only"], for_training=True):
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
        self.saved_samples_set = saved_samples_set
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

        # initialize shared-set
        self._load_cached_samples()

    def _load_cached_samples(self):
        for cache_dir in (self.colab_cache_path, self.drive_cache_path):
            for image_path in cache_dir.glob("*.jpg"):
                sample_id = image_path.stem
                
                if self.for_training:
                    companion_path = cache_dir / f"{sample_id}.json"
                else:
                    companion_path = cache_dir / f"{sample_id}.txt"

                if companion_path.is_file():
                    self.saved_samples_set.add(sample_id)
         
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

                # cache files
                # everytime opening a tar, try to go through all the files that have not been cached.
                free_colab = (shutil.disk_usage(self.colab_cache_path).free / 1024**3)
                free_drive = (shutil.disk_usage(self.drive_cache_path).free / 1024**3)
                if free_colab < CACHE_RESERVE_GB or free_drive < CACHE_RESERVE_GB:          
                    for member in tar:
                        if member.isfile() and member.name.endswith(".jpg"):
                            stem = Path(member.name).stem
                            if stem in self.saved_samples_set:
                                continue

                            image_name = f"{stem}.jpg"
                            image_file = tar.extractfile(image_name)
                            if image_file is None:
                                raise FileNotFoundError(f"{image_name} not found in {tar_path}")
                            image_bytes_cache = image_file.read()
                            if shard_type == "train":
                                meta_data_name = f"{stem}.json"
                                meta_file = tar.extractfile(meta_data_name)
                                if meta_file is None:
                                    raise FileNotFoundError(f"{meta_data_name} not found in {tar_path}")
                                meta_data_cache = json.load(meta_file)
                            elif shard_type == "validation":
                                text_data_name = f"{stem}.txt"
                                text_file = tar.extractfile(text_data_name)
                                if text_file is None:
                                    raise FileNotFoundError(f"{text_data_name} not found in {tar_path}")
                                text_data_cache = (text_file.read().decode("utf-8"))

                            # Free disk space on colab (G)
                            free_colab = (shutil.disk_usage(self.colab_cache_path).free / 1024**3)

                            # cache image and metadata.
                            # Leave 50G on each disk
                            required_gb = (len(image_bytes_cache) / 1024**3)
                            if (free_colab - required_gb > CACHE_RESERVE_GB):
                                # file paths
                                colab_image_path = self.colab_cache_path/f"{image_name}"
                                self._atomic_write_bytes(colab_image_path, image_bytes_cache)
                                if shard_type == "train":
                                    colab_meta_path = self.colab_cache_path/f"{meta_data_name}"
                                    self._atomic_write_json(colab_meta_path, meta_data_cache)
                                elif shard_type == "validation":
                                    colab_text_path = self.colab_cache_path/f"{text_data_name}"
                                    self._atomic_write_text(colab_text_path, text_data_cache)
                                self.saved_samples_set.add(stem)
                            else:
                                # Free disk space on google drive (G)
                                free_drive = (shutil.disk_usage(self.drive_cache_path).free / 1024**3)
                                if (free_drive - required_gb > CACHE_RESERVE_GB):
                                    drive_image_path = self.drive_cache_path/f"{image_name}"
                                    self._atomic_write_bytes(drive_image_path, image_bytes_cache)
                                    if shard_type == "train":
                                        drive_meta_path = self.drive_cache_path/f"{meta_data_name}"
                                        self._atomic_write_json(drive_meta_path, meta_data_cache)
                                    elif shard_type == "validation":
                                        drive_text_path = self.drive_cache_path/f"{text_data_name}"
                                        self._atomic_write_text(drive_text_path, text_data_cache)
                                    self.saved_samples_set.add(stem)

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
            caption_ids = torch.tensor(encoded_tokens,dtype=torch.int64)
            # if caption_ids has less than 2 tokens, causal LLM can't work
            if len(caption_ids) < 2:
                raise ValueError(f"Sample {sample_id} has fewer than 2 tokens")
            if shard_type == "train":
                return {
                    "patches": patches,            # [N, C * patch_size * patch_size]
                    "patch_positions": patch_positions,    # [N, 2]
                    "caption_ids": caption_ids,     # [len(all_tokens)]
                    "caption_ids_label": caption_ids,   # [len(all_tokens)]
                    "meta_data": meta_data,        # "caption", "url", "key", "status", "error_message", "width", "height", "exif", "original_width", "original_height"
                }
            elif shard_type == "validation":
                return {
                    "patches": patches,            # [N, C * patch_size * patch_size]
                    "patch_positions": patch_positions,    # [N, 2]
                    "caption_ids": caption_ids,     # [len(all_tokens)]
                    "caption_ids_label": caption_ids,   # [len(all_tokens)]
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
