import torch
import time
import math
import pytest
import os
import regex as re
import yaml
import random
import numpy as np
from pathlib import Path
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from torch.utils.data import DataLoader
#from src.data.regex_tokenizer import RegexTokenizer
from src.data.regex_tokenizer_with_lib import RegexTokenizer
from src.data.all_in_one import TextDecodeDataset
from src.data.dataset import ImageDataset
from src.data.dataset import ImageDatasetLocal
from src.data.image_dataset_sampler import ImageDatasetBatchSampler
from src.data.all_in_one import TokenBatchSampler
from src.data.all_in_one import PaddingCollator
from src.data.collator import VitCollator
from src.base.self_attention import SelfAttention
from src.models.ViT import ViT
from src.models.GPT import GPT
from src.models.multimodalGPT import MultimodalGPT
from src.training.trainer_multimodalGPT import Trainer
from scripts.helper import eval

LOCAL_YAML_PATH_GPT = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/gpt.yaml")
COLAB_YAML_PATH_GPT = Path("/content/AI/LLM_research_platform/configs/gpt.yaml")

# load yaml config
with open(COLAB_YAML_PATH_GPT,"r") as f:
    config_gpt = yaml.safe_load(f)

LOCAL_YAML_PATH_VIT = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/vit.yaml")
COLAB_YAML_PATH_VIT = Path("/content/AI/LLM_research_platform/configs/vit.yaml")

# load yaml config
with open(COLAB_YAML_PATH_VIT,"r") as f:
    config_vit = yaml.safe_load(f)

SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model(device,vocab_size,dropout):
    vit = ViT(
            d_model=config_vit["model"]["d_model"], 
            n_heads=config_vit["model"]["n_heads"],
            n_layers=config_vit["model"]["n_layers"], 
            in_channels=config_vit["data"]["in_channels"], 
            max_seq_len=config_vit["data"]["max_len"],
            patch_size=config_vit["data"]["patch_size"],
            mlp_ratio=config_vit["model"]["mlp_ratio"],
            dropout=dropout,
            RoPE=config_vit["model"]["rope"],
            rope_dims=config_vit["model"]["rope_dims"],
            num_class=config_vit["model"]["num_class"], 
            cls_enabled=config_vit["model"]["cls"],
            sdpa=True,
            causal=False,
            cross_attention_enabled=False,
        ).to(device)
                    
    gpt = GPT(
            vocab_size=vocab_size,
            d_model=config_gpt["model"]["d_model"],
            max_seq_len=config_gpt["data"]["max_len"],
            n_layers=config_gpt["model"]["n_layers"],
            n_heads=config_gpt["model"]["n_heads"],
            cross_attention=config_gpt["model"]["cross_attention"],
            d_model_ca=config_vit["model"]["d_model"],
            causal=config_gpt["model"]["causal"],
            RoPE=config_gpt["model"]["rope"],
            rope_dims=config_gpt["model"]["rope_dims"],
            dropout=dropout,
            cls_enabled=config_gpt["model"]["cls"],
            sdpa=True,
        ).to(device)
                    
    model = torch.compile(MultimodalGPT(vit = vit, gpt = gpt))
    return model


#--------tuning for vit-gpt model------------------
# learning rate
if __name__ == "__main__":
    def decay_and_dropout():
        #testing parameters setup:
        weight_decays = [0.0, 0.01, 0.1]
        dropouts = [0.0, 0.1, 0.2]
        epochs = 20

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device:", device)

        #training dataset
        tokenizer = RegexTokenizer()
        tokenizer.train()
        vocab_size = tokenizer.get_vocab_size()
        print(f"tokenizer is trained. The vocab size is {vocab_size}")

        # training dataloader
        dataset_train = ImageDatasetLocal(tokenizer=tokenizer,
                                        data_dir=config_vit["data"]["data_path_local"], 
                                        image_only=config_vit["data"]["image_only"],
                                        for_training=True)
        collator_train = VitCollator(token_pad = 0,
                                    image_pad = 0.0,
                                    label_pad = -100,
                                    image_only = config_vit["data"]["image_only"], 
                                    for_training = True)
        sampler_train = ImageDatasetBatchSampler(dataset = dataset_train,
                                                 batch_size=config_vit["data"]["batch_size"],
                                                 shuffle=False)
        
        #evaluation dataset
        dataset_eval = ImageDatasetLocal(tokenizer=tokenizer,
                                                data_dir=config_vit["data"]["data_path_local"], 
                                                image_only=config_vit["data"]["image_only"],
                                                for_training=False,)
        
        sampler_eval = ImageDatasetBatchSampler(dataset=dataset_eval,
                                                batch_size=config_vit["data"]["validation_batch_size"],
                                                shuffle=False,)
                        
        collator_eval = VitCollator(token_pad = 0,
                                    image_pad = 0.0,
                                    label_pad = -100,
                                    image_only = config_vit["data"]["image_only"], 
                                    for_training = False)
                        
        loader_eval = DataLoader(dataset=dataset_eval,
                                collate_fn = collator_eval,
                                batch_sampler = sampler_eval,
                                shuffle = False,
                                num_workers=4,
                                pin_memory=(device == "cuda"),)

        # set up constants
        accumulation_steps = config_vit["training"]["accumulation_steps"]
        min_lr_ratio = 0.10

        # best record
        best_val_loss = float("inf")
        best_weight_decay = 0.0
        best_dropout = 0.0

        # record from each test
        record_best_loss = []
        record_weight_decay = []
        record_dropouts = []
        record_best_epoch = []
        record_elapsed_time = []
        
        print(f"{len(dataset_train)} samples. {len(dataset_eval)} samples")
        for weight_decay in weight_decays:
            for dropout in dropouts:
                    print(f"Testing: weight_decay={weight_decay} | dropout={dropout}")
                    set_seed(SEED) 
                    model = build_model(device=device, vocab_size=vocab_size,dropout=dropout)
                    loader_train = DataLoader(dataset=dataset_train,
                                            collate_fn = collator_train,
                                            batch_sampler = sampler_train,
                                            shuffle = False,
                                            num_workers=4,
                                            pin_memory=(device == "cuda"),)
                    
    
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr = config_vit["training"]["learning_rate"],
                        weight_decay = weight_decay,
                        betas=(0.9, 0.95),
                    )
                    trainer = Trainer(
                        model,
                        device,
                    )

                    # ----------set up for learning-scheduler-------    
                    updates_per_epoch = math.ceil(
                        len(loader_train) / accumulation_steps
                    )
                    total_steps = epochs * updates_per_epoch

                    warmup_ratio = 0.05
                    warmup_steps = int(total_steps * warmup_ratio)
                    def lr_lambda(step):
                        # Linear warmup.
                        if warmup_steps > 0 and step < warmup_steps:
                            return (step + 1) / warmup_steps

                        # Cosine decay after warmup.
                        decay_steps = max(1, total_steps - warmup_steps)
                        progress = min(1.0,max(0.0,(step - warmup_steps) / decay_steps,),)
                        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                        return min_lr_ratio + (
                            1.0 - min_lr_ratio
                        ) * cosine
                    scheduler = LambdaLR(optimizer, lr_lambda)

                    global_step = 0
                    total_token = 0

                    # training loop:
                    best_epoch = 0
                    best_loss = float("inf")
                    best_recorded_elapsed_time = 0.0
                    for epoch in range(epochs):
                        model.train()
                        optimizer.zero_grad()
                        loss_accu = 0.0
                        cnt = 0.0
                        epoch_tokens = 0
                        num_batches = len(loader_train)
                        # start elaped time
                        epoch_time_start = time.perf_counter()
                        for i, batch in enumerate(loader_train):
                            patched_items = {
                                "patched_input": batch["patched_input"].to(device,non_blocking=True),
                                "patched_pad_mask": batch["patched_pad_mask"].to(device,non_blocking=True),
                                "patched_positions": batch["patched_positions"].to(device,non_blocking=True)
                            }
                            text_x = batch["caption_ids"].to(device,non_blocking=True)
                            text_pad_mask = batch["text_pad_mask"].to(device,non_blocking=True)
                            text_positions = batch["text_positions"].to(device,non_blocking=True)
                            y = batch[ "caption_ids_label"].to(device,non_blocking=True)

                            num_tokens = text_pad_mask.sum().item()
                            epoch_tokens += num_tokens
                            # calculate the accumulation step size to avoid the shortage of last a few batchs
                            window_start = (i // accumulation_steps) * accumulation_steps
                            window_size = min(accumulation_steps,num_batches - window_start,)
                            raw_loss = trainer.train_step(
                                patched_items = patched_items, 
                                text_x = text_x, 
                                text_pad_mask = text_pad_mask,
                                text_positions = text_positions, 
                                is_prefill = False, 
                                is_generate = False, 
                                y=y,
                            )

                            (raw_loss / window_size).backward()

                            loss_accu += raw_loss.item()
                            cnt += 1

                            text_seq_len = text_x.size(1)
                            if(i % 500 == 0):
                                print(
                                    f"epoch {epoch} | "
                                    f"batch {i} | "
                                    f"batch_size {config_vit["data"]["batch_size"]} | "
                                    f"seq_len {text_seq_len} | "
                                    f"tokens {num_tokens} | "
                                    f"loss {raw_loss.item() :.4f}"
                                )

                            if (i + 1) % accumulation_steps == 0 or i == len(loader_train) - 1:
                                #----add gradient clipping
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    max_norm=1.0,
                                )

                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad(set_to_none=True)

                                global_step += 1

                            avg_train_loss = loss_accu / cnt
                        # epoch time
                        epoch_elapsed_time = time.perf_counter() - epoch_time_start
                        total_token += epoch_tokens

                        print(
                            f"epoch {epoch} | "
                            f"train_loss={avg_train_loss:.4f} | "
                            f"lr={optimizer.param_groups[0]['lr']:.2e}"
                        )
                        # Validate at the end of the epoch.
                        val_loss = eval(trainer=trainer,loader_eval = loader_eval,epoch = epoch,model=model,device=device)
                        # Save the best checkpoint when validation loss improves.
                        if val_loss < best_loss:
                            best_epoch = epoch
                            best_loss = val_loss
                            best_recorded_elapsed_time = epoch_elapsed_time
                        
                    if best_loss < best_val_loss:
                        best_val_loss = best_loss
                        best_weight_decay = weight_decay
                        best_dropout = dropout
                        best_epoch_elapsed_time = best_recorded_elapsed_time
                    # update records
                    record_best_loss.append(best_loss)
                    record_weight_decay.append(weight_decay)
                    record_dropouts.append(dropout)
                    record_best_epoch.append(best_epoch)
                    record_elapsed_time.append(best_recorded_elapsed_time)
                    
                
        # output
        print( "decay   dropout.  best_epoch   epoch_time_sec   best_val_loss")

        for i in range(len(record_best_epoch)):
            print(
                f"{record_weight_decay[i]:.2e}   "
                f"{record_dropouts[i]}   "
                f"{record_best_epoch[i] + 1}   "
                f"{record_elapsed_time[i]:.2f}   "
                f"{record_best_loss[i]:.4f}"
            )

        print("------------ optimal setup -----------")
        print(f"Weight decay: {best_weight_decay:.2e}")
        print(f"Dropout : {best_dropout}")
        print(f"Best epoch training time: {best_epoch_elapsed_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.4f}")
    decay_and_dropout()


                        
                    




            



