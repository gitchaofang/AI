import torch
from torch.utils.data import DataLoader

import random
from src.data.dataset import VariableLengthDataset
from src.data.collator import PaddingCollator
from src.data.token_batch_sampler import TokenBatchSampler
from src.models.GPT import GPT
from src.training.trainer import Trainer

# Training scripts
import wandb
wandb.login()
wandb.init(
    project="my-gpt",
    config={
        "d_model": 384,
        "n_layers": 6,
        "n_heads": 6,
        "max_seq_len": 256,
        "batch_tokens": 3000,
        "learning_rate": 3e-4,
        "epochs": 200,
        "accumulation_steps": 8,
    }
)
#traiing dataset
dataset_train = VariableLengthDataset("novel_train.txt") 
vocab_size = dataset_train.get_vocab_size()
print(f"vocab_size is: {vocab_size}")
collator = PaddingCollator()
sampler_train = TokenBatchSampler(dataset_train,3000)

loader_train = DataLoader(
    dataset_train,
    collate_fn = collator,
    batch_sampler = sampler_train,
    shuffle = False,
    pin_memory = True,
)

#evaluation dataset
dataset_eval = VariableLengthDataset("novel_eval.txt") 
sampler_eval = TokenBatchSampler(dataset_eval,16000)

loader_eval = DataLoader(
    dataset_eval,
    collate_fn = collator,
    batch_sampler = sampler_eval,
    shuffle = False,
    pin_memory = True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

model = GPT(
    vocab_size = vocab_size,
    d_model = 384,
    max_seq_len = 256,
    n_layers = 6,
    n_heads = 6,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 3e-4,
)


trainer = Trainer(
    model,
    optimizer,
    device,
)

epoches = 200
accumulation_steps = 8
for epoch in range(epoches):
    model.train()
    optimizer.zero_grad()
    loss_accu = 0.0
    cnt = 0.0
    epoch_tokens = 0
    #train
    for i, batch in enumerate(loader_train):
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        num_tokens = pad_mask.sum().item()
        epoch_tokens += num_tokens

        loss = trainer.train_step(x, y, pad_mask) / accumulation_steps
        loss.backward()

        loss_accu += (loss.item() * accumulation_steps)

        if(i % 100 == 0):
            batch_size = x.size(0)
            seq_len = x.size(1)

           

            wandb.log({
                "train/batch_loss": loss.item(),
                "train/batch_size": batch_size,
                "train/seq_len": seq_len,
                "train/tokens": num_tokens,
                "epoch": epoch,
                "batch": i,
            })

        if(i % 250 == 0):
             print(
                            f"epoch {epoch} | "
                            f"batch {i} | "
                            f"batch_size {batch_size} | "
                            f"seq_len {seq_len} | "
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
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            num_tokens = pad_mask.sum().item()
            eval_tokens += num_tokens

            eval_loss_accu += trainer.train_step(x, y, pad_mask).item()
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