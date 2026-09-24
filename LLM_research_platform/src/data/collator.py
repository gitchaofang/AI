import torch
import yaml
from pathlib import Path
from .helper import patchify

# load config file
LOCAL_YAML_PATH = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/vit.yaml")
COLAB_YAML_PATH = Path("/content/AI/LLM_research_platform/configs/vit.yaml")
# load yaml config
with open(COLAB_YAML_PATH,"r") as f:
    vit_config = yaml.safe_load(f)

EXTRACTED_PATH  = Path(vit_config["data"]["extracted_path"]) 
SHARED_PATH = Path(vit_config["data"]["shared_path"])


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
    
class VitCollator:
    def __init__(self, token_pad = 0, image_pad = 0.0, label_pad = -100, image_only = vit_config["data"]["image_only"]):
        self.token_pad = token_pad
        self.label_pad = label_pad
        self.image_pad = image_pad
        self.image_only = image_only

    def __call__(self, batch):
        batch_size = len(batch)
        max_len_patch = max(len(x["patches"]) for x in batch)
        max_len_text = max(len(x["meta_data"]["caption"]) for x in batch)
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
            patched_input[i][:length_patches] = item["patches"] # patched input
            patch_positions[i][:length_patches] = item["patch_positions"] # patch coordinates for RoPE
            pad_mask_patch[i][:length_patches] = 1 # pad maskes for patched input

            # text
            if not self.image_only:
                length_text = len(item["caption_ids"])
                caption_ids[i][:length_text] = item["caption_ids"]
                pad_mask_text[i][:length_text] = 1
          
            

        if not self.image_only:
            return {
                "patched_input": patched_input, #[B, max_len_patch ,C * patch_size * patch_size]
                "patch_positions": patch_positions, # [B, max_len_patch]
                "pad_mask_patch": pad_mask_patch,# [B, max_len_patch,2]
                "caption_ids": caption_ids, # [B, max_len_text]
                "pad_mask_text": pad_mask_text, # [B, max_len_text]
                "meta_data": meta_data, # list of dict. B dicts
            }

        return {
            "patched_input": patched_input, #[B, max_len_patch ,C * patch_size * patch_size]
            "patch_positions": patch_positions, # [B, max_len_patch]
            "pad_mask_patch": pad_mask_patch,# [B, max_len_patch]
            "meta_data": meta_data, # list of dict. B dicts
        } 