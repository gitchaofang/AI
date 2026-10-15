import torch
import os
from pathlib import Path
import yaml
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

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

def eval(loader_eval, trainer, epoch):
    #eval:
        model.eval()
        eval_loss_accu = 0.0
        eval_cnt = 0.0
        eval_tokens = 0
        with torch.no_grad():
            for i, batch in enumerate(loader_eval):
                patched_items = {"patched_input": batch["patched_input"].to(device),
                                 "patched_pad_mask": batch["patched_pad_mask"].to(device),
                                 "patched_positions": batch["patched_positions"].to(device)}
                text_x = batch["caption_ids"].to(device)
                text_pad_mask = batch["text_pad_mask"].to(device)
                text_positions = batch["text_positions"].to(device)
                y = batch[ "caption_ids_label"].to(device)
    
                num_tokens = text_x.sum().item()
                eval_tokens += num_tokens
    
                eval_loss_accu += trainer.train_step(patched_items = patched_items, 
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
        
# Training scripts
# weight and bias setup 
import wandb
wandb.login()
wandb.init(
    project="my-image-text-multimodal",
    config={
        "d_model": config_vit["model"]["d_model"],
        "n_layers": config_vit["model"]["n_layers"],
        "n_heads": config_vit["model"]["n_heads"],
        "max_seq_len": config_vit["data"]["max_len"],
        "learning_rate": config_vit["training"]["learning_rate"],
        "epochs": config_vit["training"]["epochs"],
        "accumulation_steps": config_vit["training"]["accumulation_steps"],
    }
)
#traiing dataset
tokenizer = RegexTokenizer()
tokenizer.train()
vocab_size = tokenizer.get_vocab_size()
print(f"tokenizer is trained. The vocab size is {vocab_size}")
dataset_train = ImageDatasetLocal(tokenizer=tokenizer,
                                  data_dir=config_vit["data"]["data_path_local"], 
                                  image_only=config_vit["data"]["image_only"],
                                  for_training=True)

sampler_train = ImageDatasetBatchSampler(dataset = dataset_train,
                                         batch_size=config_vit["data"]["batch_size"],
                                         shuffle=config_vit["data"]["shuffle"])

collator_train = VitCollator(token_pad = 0,
                       image_pad = 0.0,
                       label_pad = -100,
                       image_only = config_vit["data"]["image_only"], 
                       for_training = True)

loader_train = DataLoader(
    dataset=dataset_train,
    collate_fn = collator_train,
    batch_sampler = sampler_train,
    shuffle = False,
    pin_memory = False,
    num_workers=4,
)

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

loader_eval = DataLoader(
    dataset=dataset_eval,
    collate_fn = collator_eval,
    batch_sampler = sampler_eval,
    shuffle = False,
    pin_memory = False,
    num_workers=4,
)


vit = ViT(
    d_model=config_vit["model"]["d_model"], 
    n_heads=config_vit["model"]["n_heads"],
    n_layers=config_vit["model"]["n_layers"], 
    in_channels=config_vit["data"]["in_channels"], 
    max_seq_len=config_vit["data"]["max_len"],
    patch_size=config_vit["data"]["patch_size"],
    mlp_ratio=config_vit["model"]["mlp_ratio"],
    dropout=config_vit["model"]["dropout"],
    RoPE=config_vit["model"]["rope"],
    rope_dims=config_vit["model"]["rope_dims"],
    num_class=config_vit["model"]["num_class"], 
    cls_enabled=config_vit["model"]["cls"],
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
    dropout=config_gpt["model"]["dropout"],
    cls_enabled=config_gpt["model"]["cls"],
).to(device)

model = MultimodalGPT(vit = vit, gpt = gpt)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = config_vit["training"]["learning_rate"],
)


trainer = Trainer(
    model,
    device,
)

epoches = config_vit["training"]["epochs"]
accumulation_steps = config_vit["training"]["accumulation_steps"]
for epoch in range(epoches):
    model.train()
    optimizer.zero_grad()
    loss_accu = 0.0
    cnt = 0.0
    epoch_tokens = 0
    
    '''
    patch_items,        #vit
    text_x,             #gpt
    text_pad_mask,      #gpt
    text_positions,     #gpt
    is_prefill=False,   #gpt
    is_generate=False): #gpt
    '''
    #train
    print(f"{len(loader_train)}")
    for i, batch in enumerate(loader_train):
        patched_items = {
            "patched_input": batch["patched_input"].to(device),
            "patched_pad_mask": batch["patched_pad_mask"].to(device),
            "patched_positions": batch["patched_positions"].to(device)
        }
        text_x = batch["caption_ids"].to(device)
        text_pad_mask = batch["text_pad_mask"].to(device)
        text_positions = batch["text_positions"].to(device)
        y = batch[ "caption_ids_label"].to(device)

        num_tokens = text_pad_mask.sum().item()
        epoch_tokens += num_tokens

        loss = trainer.train_step(
            patched_items = patched_items, 
            text_x = text_x, 
            text_pad_mask = text_pad_mask,
            text_positions = text_positions, 
            is_prefill = False, 
            is_generate = False, 
            y=y,
        )/ accumulation_steps
        loss.backward()

        loss_accu += (loss.item() * accumulation_steps)

        if(i % 100 == 0):
            batch_size = text_x.size(0)
            text_seq_len = text_x.size(1)

            wandb.log({
                "train/batch_loss": loss.item(),
                "train/batch_size": batch_size,
                "train/seq_len": text_seq_len,
                "train/tokens": num_tokens,
                "epoch": epoch,
                "batch": i,
            })

        if(i % 250 == 0):
             print(
                            f"epoch {epoch} | "
                            f"batch {i} | "
                            f"batch_size {batch_size} | "
                            f"seq_len {text_seq_len} | "
                            f"tokens {num_tokens} | "
                            f"loss {loss.item() * accumulation_steps:.4f}"
                        )

        if (i + 1) % accumulation_steps == 0 or i == len(loader_train) - 1:
            optimizer.step()
            optimizer.zero_grad()
        cnt += 1

        # eval
        if(i % 250 == 0):
             eval_res = eval(trainer=trainer,loader_eval = loader_eval,epoch = epoch)
            
    avg_train_loss = loss_accu / cnt
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch} | ave_loss: {avg_train_loss}")
    wandb.log({
                "train/epoch_loss": avg_train_loss,
                "epoch": epoch,
            })
    wandb.log({
        "train/epoch_tokens": epoch_tokens,
        "train/learning_rate": current_lr,
        "epoch": epoch,
    })

wandb.finish()