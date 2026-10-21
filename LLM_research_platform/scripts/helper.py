import torch
import math
import os
from pathlib import Path
from torch.optim.lr_scheduler import LambdaLR
import yaml
import random
import numpy as np
from torch.utils.data import DataLoader
from src.data.regex_tokenizer_with_lib import RegexTokenizer

import random
from src.data.dataset import ImageDatasetLocal
from src.data.collator import VitCollator
from src.data.image_dataset_sampler import ImageDatasetBatchSampler  
from src.models.GPT import GPT
from src.models.ViT import ViT
from src.models.multimodalGPT import MultimodalGPT
from src.training.trainer_multimodalGPT import Trainer

def eval(loader_eval, trainer, epoch):
    #eval:
    print(f"eval loader size: {len(loader_eval)}")
    model.eval()
    eval_loss_accu = 0.0
    eval_cnt = 0.0
    eval_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(loader_eval):
            patched_items = {"patched_input": batch["patched_input"].to(device,non_blocking=True),
                             "patched_pad_mask": batch["patched_pad_mask"].to(device,non_blocking=True),
                                 "patched_positions": batch["patched_positions"].to(device,non_blocking=True)}
            text_x = batch["caption_ids"].to(device,non_blocking=True)
            text_pad_mask = batch["text_pad_mask"].to(device,non_blocking=True)
            text_positions = batch["text_positions"].to(device,non_blocking=True)
            y = batch[ "caption_ids_label"].to(device,non_blocking=True)
    
            num_tokens = text_x.sum().item()
            eval_tokens += num_tokens
    
            eval_loss_accu += trainer.train_step(
                patched_items = patched_items, 
                text_x = text_x, 
                text_pad_mask = text_pad_mask,
                text_positions = text_positions, 
                is_prefill = False, 
                is_generate = False, 
                y=y,).item()
            eval_cnt += 1
    avg_eval_loss = eval_loss_accu / eval_cnt
    print(f"epoch | {epoch} | eval error: {avg_eval_loss}")
    wandb.log({
            "eval/loss": avg_eval_loss,
            "eval/tokens": eval_tokens,
            "epoch": epoch,
        })
    model.train()
    return avg_eval_loss

