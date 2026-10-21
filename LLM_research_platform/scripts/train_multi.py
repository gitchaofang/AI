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
from .helper import eval

# Set this before importing torch or initializing CUDA.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_seed(seed: int = 42, deterministic: bool = False):
    """Set random seeds for reproducible training."""

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN behavior
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    # Optionally enforce deterministic algorithms
    torch.use_deterministic_algorithms(deterministic)

    print(f"Random seed: {seed}")
    print(f"Deterministic algorithms: {deterministic}")


SEED = 42
set_seed(SEED, deterministic=False)

# ---------- Checkpoint save/load ----------
CHECKPOINT_DIR = Path("/content/drive/MyDrive/protected/data/cc3m/checkpoints")
#CHECKPOINT_DIR = Path("content/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def get_model_to_save(model):
    """Return the original model when wrapped by torch.compile."""
    return getattr(model, "_orig_mod", model)

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    loader_generator,
    best_val_loss=float("inf"),
):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state_dict": get_model_to_save(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "loader_generator_state": loader_generator.get_state(),
    }

    path = Path(path)
    temp_path = Path(str(path) + ".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.replace(path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    loader_generator,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    get_model_to_save(model).load_state_dict(
        checkpoint["model_state_dict"]
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    loader_generator.set_state(checkpoint["loader_generator_state"].cpu())

    # Restore RNG states last, after constructing/loading all objects.
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

    if (
        torch.cuda.is_available()
        and checkpoint["cuda_rng_state_all"] is not None
    ):
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])

    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]
    best_val_loss = checkpoint["best_val_loss"]

    print(
        f"Resuming from epoch {start_epoch}, "
        f"global_step {global_step}"
    )
    return start_epoch, global_step, best_val_loss


# for dataloader
def seed_worker(worker_id):
    # PyTorch assigns each worker a seed derived from
    # the DataLoader generator.
    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)
loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)



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

'''
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
'''
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
    num_workers=4,
    pin_memory=(device == "cuda"),
    worker_init_fn=seed_worker,
    generator=loader_generator,
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
    num_workers=4,
    pin_memory=(device == "cuda"),
    worker_init_fn=seed_worker,
    generator=loader_generator,
)

if config_vit["model"]["sdpa"]:
    print(f"using sdpa for Vit")
if  config_gpt["model"]["sdpa"]:
    print(f"using sdpa for GPT")
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
    sdpa=config_gpt["model"]["sdpa"],
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
    sdpa=config_gpt["model"]["sdpa"],
).to(device)

model = torch.compile( MultimodalGPT(vit = vit, gpt = gpt))

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = config_vit["training"]["learning_rate"],
    weight_decay = 0.01,
    betas=(0.9, 0.95),
)


trainer = Trainer(
    model,
    device,
)

epochs = config_vit["training"]["epochs"]
accumulation_steps = config_vit["training"]["accumulation_steps"]

# ----------set up for learning-scheduler-------
updates_per_epoch = math.ceil(
    len(loader_train) / accumulation_steps
)
total_steps = epochs * updates_per_epoch

warmup_ratio = 0.01
warmup_steps = int(total_steps * warmup_ratio)

min_lr_ratio = 0.10
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

# ---------- Resume from latest checkpoint, if available ----------
checkpoint_path = CHECKPOINT_DIR / "latest.pt"
start_epoch = 0
global_step = 0
best_val_loss = float("inf")

if checkpoint_path.exists():
    start_epoch, global_step, best_val_loss = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loader_generator=loader_generator,
        device=device,
    )
else:
    print("No checkpoint found. Starting training from scratch.")

for epoch in range(start_epoch, epochs):
    model.train()
    optimizer.zero_grad()
    loss_accu = 0.0
    cnt = 0.0
    epoch_tokens = 0
    num_batches = len(loader_train)
    
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
        window_start = (
            i // accumulation_steps
        ) * accumulation_steps

        window_size = min(
            accumulation_steps,
            num_batches - window_start,
        )

        if config_vit["training"]["bp16"]:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                raw_loss = trainer.train_step(
                    patched_items = patched_items, 
                    text_x = text_x, 
                    text_pad_mask = text_pad_mask,
                    text_positions = text_positions, 
                    is_prefill = False, 
                    is_generate = False, 
                    y=y,
                )
        else:
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
        
        if(i % 100 == 0):
            batch_size = text_x.size(0)
            text_seq_len = text_x.size(1)

            wandb.log({
                "train/batch_loss": raw_loss.item(),
                "train/batch_size": batch_size,
                "train/seq_len": text_seq_len,
                "train/tokens": num_tokens,
                "epoch": epoch,
                "batch": i,
            })

        if(i % 1000 == 0):
             print(
                            f"epoch {epoch} | "
                            f"batch {i} | "
                            f"batch_size {batch_size} | "
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

            wandb.log({
                "train/grad_norm": grad_norm.item(),
                "train/learning_rate": (
                    optimizer.param_groups[0]["lr"]
                ),
                "train/global_step": global_step,
                "epoch": epoch,
            })
        

        # eval
     #   if(i % 5000 == 0):
     #       eval(trainer=trainer,loader_eval = loader_eval,epoch = epoch)
            
    avg_train_loss = loss_accu / cnt
    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"epoch {epoch} | "
        f"train_loss={avg_train_loss:.4f} | "
        f"lr={optimizer.param_groups[0]['lr']:.2e}"
    )

    # Validate at the end of the epoch.
    val_loss = eval(
        trainer=trainer,
        loader_eval = loader_eval,
        epoch = epoch
    )

    wandb.log({
                "train/epoch_loss": avg_train_loss,
                "epoch": epoch,
            })
    wandb.log({
        "train/epoch_tokens": epoch_tokens,
        "train/learning_rate": current_lr,
        "epoch": epoch,
    })

    # Save the best checkpoint when validation loss improves.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
            path=CHECKPOINT_DIR / "best.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            loader_generator=loader_generator,
            best_val_loss=best_val_loss,
        )
        print(f"New best validation loss: {best_val_loss:.4f}")

    # Always save the most recently completed epoch for resuming.
    save_checkpoint(
        path=CHECKPOINT_DIR / "latest.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        loader_generator=loader_generator,
        best_val_loss=best_val_loss,
    )
wandb.finish()