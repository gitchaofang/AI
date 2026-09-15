import random
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.regex_tokenizer import RegexTokenizer
from src.data.all_in_one import TextDecodeDataset
from src.data.all_in_one import TokenBatchSampler
from src.data.all_in_one import PaddingCollator

#device = "cuda" if torch.cuda.is_available() else "cpu"
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TRAINING_FILENAME = "training.txt"
VALIDATION_FILENAME = "validation.txt"
MAX_LEN = 256
HIDDEN_DIM = 384
BATCH_SIZE = 8
ACCUMULATION_STEPS = 8
# self attention
class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, pad_token=0, dropout = 0.2):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.atten_dropout = nn.Dropout(dropout)
        self.out_drouout = nn.Dropout(dropout)
        mask = torch.tril(
            torch.ones((self.max_seq_len, self.max_seq_len), dtype=torch.int64)
        )
        self.register_buffer(
            "causal_mask",
            mask.view(1, 1, self.max_seq_len, self.max_seq_len),
        )

        self.kv_cache = None
        self.kv_mask = None
        self.pad_token = pad_token

    def reset_cache(self):
        self.kv_cache = None
        self.kv_mask = None

    def _dot_product(self, q, k, v, combined_mask):
        B, T_q, D = q.shape
        T_k = k.shape[1]

        q = q.view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-1, -2)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(combined_mask == 0, -1e10)

        attention = torch.softmax(scores, dim=-1)

        # dropout on attention
        attention = self.atten_dropout(attention)
        out = attention @ v
        out = out.transpose(1, 2).contiguous().view(B, T_q, D)

        out = self.out_proj(out)
        # dropout on out project
        out  = self.out_drouout(out)
        return out


    def forward(self, x, pad_mask, is_prefill=False, is_generate=False):
        B, T_q, D = x.shape
        pad_mask = pad_mask.to(device=x.device, dtype=torch.float32)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if is_generate:
            if self.kv_cache is None:
                self.kv_cache = (k, v)
                self.kv_mask = pad_mask
            else:
                k_full = torch.cat([self.kv_cache[0], k], dim=1)
                v_full = torch.cat([self.kv_cache[1], v], dim=1)
                self.kv_cache = (k_full, v_full)
                self.kv_mask = torch.cat([self.kv_mask, pad_mask], dim=1)

            k_full, v_full = self.kv_cache
            T_k = k_full.size(1)

            valid_mask = self.kv_mask.unsqueeze(-1) @ self.kv_mask.unsqueeze(-2)
            valid_mask = valid_mask.unsqueeze(1)

            causal_mask = self.causal_mask[:, :, :T_k, :T_k].to(x.device).float()
            combined_mask = causal_mask * valid_mask
            combined_mask = combined_mask[:, :, -T_q:, :]

            return self._dot_product(q, k_full, v_full, combined_mask)

        if is_prefill:
            self.kv_cache = (k, v)
            self.kv_mask = pad_mask

        pad_mask = pad_mask.unsqueeze(-1)
        pad_mask = pad_mask @ pad_mask.transpose(-1, -2)
        combined_mask = self.causal_mask[:, :, :T_q, :T_q].to(x.device).float() * pad_mask.unsqueeze(1)

        return self._dot_product(q, k, v, combined_mask)
        

# transformer:
class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio = 4, dropout = 0.2):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model, n_heads, max_seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x, pad_mask, is_prefill=False, is_generate=False):
        x = x + self.attention(self.norm1(x), pad_mask, is_prefill, is_generate)
        x = x + self.ffn(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        max_seq_len: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size + 1 ,d_model,)
        self.position_embedding = nn.Embedding(max_seq_len,d_model,)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model,)
        self.lm_linear = nn.Linear(d_model,vocab_size + 1,)
        self.position_offset = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_cache(self):
        """
        Clear all KV caches and reset logical position tracking.
        Call this before starting a new generation session.
        """
        self.position_offset = None

        for block in self.blocks:
            block.attention.reset_cache()

    def forward(
        self,
        x,
        pad_mask,
        is_prefill=False,
        is_generate=False,
    ):
        B, T_q = x.shape

        assert T_q <= self.max_seq_len

        assert not (
            is_prefill and is_generate
        ), "is_prefill and is_generate cannot both be True"

        if not is_prefill and not is_generate:
            position_ids = torch.arange(T_q,device=x.device,).unsqueeze(0).expand(B,T_q,)
        elif is_prefill:
            position_ids = torch.arange(T_q,device=x.device,).unsqueeze(0).expand(B,T_q,)
            self.position_offset = (pad_mask.to(torch.int64).sum(dim=1))
        else:
            assert is_generate
            assert self.position_offset is not None, (
                "Generation requires a previous prefill"
            )
            relative_positions = torch.arange(T_q,device=x.device,).unsqueeze(0)
            position_ids = (self.position_offset.unsqueeze(1)+ relative_positions)

        assert torch.all(position_ids >= 0)
        assert torch.all(position_ids < self.max_seq_len), ("Position ID exceeds max_seq_len")

        x = (self.token_embedding(x)+ self.position_embedding(position_ids))
        for block in self.blocks:
            x = block(
                x,
                pad_mask,
                is_prefill=is_prefill,
                is_generate=is_generate,
            )
        x = self.norm(x)
        logits = self.lm_linear(x)
        if is_generate:
            self.position_offset += (pad_mask.to(torch.int64).sum(dim=1))

        return logits


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_step(self, x, y, mask):

        logits = self.model(x, mask, is_prefill=False, is_generate=False)

        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            y.reshape(B * T),
            ignore_index=-100,
        )

        return loss


# Training scripts
# weight and bias setup 
import wandb
wandb.login()
wandb.init(
    project="my-gpt",
    config={
        "d_model": HIDDEN_DIM,
        "n_layers": 6,
        "n_heads": 6,
        "max_seq_len": MAX_LEN,
        "batch_tokens": 3000,
        "learning_rate": 3e-4,
        "epochs": 200,
        "accumulation_steps": ACCUMULATION_STEPS,
    }
)
#traiing dataset
tokenizer = RegexTokenizer()
tokenizer.train()
print(f"tokenizer is trained")
dataset_train = TextDecodeDataset(tokenizer=tokenizer,
                                max_len=MAX_LEN,
                                filename=TRAINING_FILENAME)
sampler_train = TokenBatchSampler(dataset = dataset_train,
                                  batch_size=8)
collator = PaddingCollator()

vocab_size = tokenizer.get_vocab_size()
print(f"vocab_size is: {vocab_size}")

loader_train = DataLoader(
    dataset_train,
    collate_fn = collator,
    batch_sampler = sampler_train,
    shuffle = False,
    pin_memory = True,
)

#evaluation dataset
dataset_eval = TextDecodeDataset(tokenizer=tokenizer,
                                 max_len = MAX_LEN,
                                 filename = VALIDATION_FILENAME) 
sampler_eval = TokenBatchSampler(dataset=dataset_eval,
                                 batch_size=1)

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
    d_model = HIDDEN_DIM,
    max_seq_len = MAX_LEN,
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
accumulation_steps = ACCUMULATION_STEPS
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




    
