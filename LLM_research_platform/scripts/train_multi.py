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
with open(LOCAL_YAML_PATH_GPT,"r") as f:
    config_gpt = yaml.safe_load(f)

LOCAL_YAML_PATH_VIT = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/vit.yaml")
COLAB_YAML_PATH_VIT = Path("/content/AI/LLM_research_platform/configs/vit.yaml")

# load yaml config
with open(LOCAL_YAML_PATH_VIT,"r") as f:
    config_vit = yaml.safe_load(f)

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
dataset_train = ImageDatasetBatchSampler(tokenizer=tokenizer,
                                         data_dir=config_vit["data"]["data_path_local"], 
                                         max_len=config_vit["data"]["max_len"],
                                         image_only=config_vit["data"]["image_only"],
                                         data=config_vit["training"]["data_path_local"],
                                         for_training=True)
sampler_train = ImageDatasetBatchSampler(dataset = dataset_train,
                                         batch_size=config_vit["data"]["batch_size"],
                                         shuffle=config_vit["data"]["shuffle"])
collator = VitCollator(token_pad = 0,
                       image_pad = 0.0,
                       label_pad = -100,
                       image_only = config_vit["data"]["image_only"], 
                       for_training = True)

loader_train = DataLoader(
    dataset=dataset_train,
    collate_fn = collator,
    batch_sampler = sampler_train,
    shuffle = False,
    pin_memory = True,
    num_workers=4,
)

#evaluation dataset
dataset_eval = ImageDatasetBatchSampler(tokenizer=tokenizer,
                                        data_dir=config_vit["data"]["data_path_local"], 
                                        max_len=config_vit["data"]["max_len"],
                                        image_only=config_vit["data"]["image_only"],
                                        dataset=config_vit["training"]["data_path_local"],
                                        for_training=False,)

sampler_eval = ImageDatasetBatchSampler(dataset=dataset_train,
                                        batch_size=config_vit["data"]["validation_batch_size:"],
                                        shuffle=False,)

loader_eval = DataLoader(
    dataset=dataset_eval,
    collate_fn = collator,
    batch_sampler = sampler_eval,
    shuffle = False,
    pin_memory = True,
    num_workers=4,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

gpt = GPT(
    vocab_size=vocab_size,
    d_model=config_gpt["model"]["d_model"],
    max_seq_len=config_gpt["data"]["max_len"],
    n_layers=config_gpt["model"]["n_layers"],
    n_heads=config_gpt["model"]["n_heads"],
    cross_attention=config_gpt["model"]["cross_attention"],
    causal=config_gpt["model"]["causal"],
    RoPE=config_gpt["model"]["rope"],
    rope_dims=config_gpt["model"]["rope_dims"],
    dropout=config_gpt["model"]["dropout"],
    cls_enabled=config_gpt["model"]["cls"],
).to(device)

vit = ViT(d_model=config_vit["model"]["d_model"], 
          n_heads=config_vit["model"]["n_heads"], 
          in_channels=config_vit["model"]["in_channel"], 
          max_len=config_vit["model"]["max"],
          patch_size=config_vit["model"]["patch_size"],
          n_layers=config_vit["model"]["patch_size"],
          mlp_ratio=config_vit["model"]["mlp_ratio"],
          dropout=config_vit["model"]["dropout"],
          RoPE=config_vit["model"]["rope"],
          num_class=None,
          rope_dims=["model"]["rope_dim"], 
          cls_enabled=True,
          causal=False,
          cross_attention_enabled=False)

model = MultimodalGPT(vit = vit, gpt = gpt)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = config_vit["training"]["training_rate"],
)


trainer = Trainer(
    model,
    optimizer,
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
    for i, batch in enumerate(loader_train):
        patched_items = {"patched_input": batch["patched_input"],
                       "pad_mask_patch": batch["pad_mask_patch"],
                       "patched_positions": batch["patch_positions"]}.to(device)
        text_x = batch["caption_ids"].to(device)
        text_pad_mask = batch["pad_mask_text"].to(device)
        text_positions = batch["positions_text"].to(device)
        y = batch[ "caption_ids_label"].to(device)

        num_tokens = text_pad_mask.sum().item()
        epoch_tokens += num_tokens

        loss = trainer.train_step(patched_items = patched_items, 
                                  text_x = text_x, 
                                  text_pad_mask = text_pad_mask,
                                  text_positions = text_positions, 
                                  is_prefill = False, 
                                  is_generate = False, 
                                  y=y,).item() / accumulation_steps
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
    avg_train_loss = loss_accu / cnt
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"epoch {epoch} | ave_loss: {avg_train_loss}")
    wandb.log({
        "train/epoch_tokens": epoch_tokens,
        "train/learning_rate": current_lr,
        "epoch": epoch,
    })

    #eval:
    model.eval()
    eval_loss_accu = 0.0
    eval_cnt = 0.0
    eval_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(loader_eval):
            patched_items = {"patched_input": batch["patched_input"],
                             "pad_mask_patch": batch["pad_mask_patch"],
                             "patched_positions": batch["patch_positions"]}.to(device)
            text_x = batch["caption_ids"].to(device)
            text_pad_mask = batch["pad_mask_text"].to(device)
            text_positions = batch["positions_text"].to(device)
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
        "train/epoch_loss": avg_train_loss,
        "epoch": epoch,
    })
    wandb.log({
            "eval/tokens": eval_tokens,
            "epoch": epoch,
        })

wandb.finish()