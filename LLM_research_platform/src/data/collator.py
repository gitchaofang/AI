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
    def __init__(self, token_pad = 0, label_pad = -100, image_only = vit_config["data"]["image_only"]):
        self.token_pad = token_pad
        self.label_pad = label_pad
        self.image_only = image_only

    def __call__(self, batch):
        batch_size = len(batch)
        max_len = max(len(x["patches"]) for x in batch)
        patch_d = vit_config["data"]["color"]*vit_config["data"]["patch_size"]*vit_config["data"]["patch_size"]

        patched_input = torch.full(
            (batch_size, max_len, patch_d),
            self.token_pad,
            dtype = torch.int64,
        )
        
        pad_mask = torch.zeros(
            batch_size,
            max_len,
            dtype = torch.int64,
        )

        for i, item in enumerate(batch):
            length = len(item["patches"]) # item["patches"]: [N, C * patch_size * patch_size]
            patched_input[i][:length] = item["patches"]
            pad_mask[i][:length] = 1