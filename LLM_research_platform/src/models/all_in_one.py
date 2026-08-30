import random
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.data.all_in_one import VariableLengthDataset
from src.data.all_in_one import TokenBatchSampler
from src.data.all_in_one import PaddingCollator

# self attention
class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
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

  

    def forward(self, x, combined_mask: torch.Tensor):
        B,T,D = x.shape

        assert combined_mask.shape[-1] <= self.max_seq_len
        # build attentions [B,T,D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # multi-heead [B,T,D] -> [B,H,T,Dh]
        q = q.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        k = k.view(B,T,self.n_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.n_heads,self.head_dim).transpose(1,2)

        # scores
        scores = q @ k.transpose(-1,-2)
    #    print(f"mask: {combined_mask}")
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            combined_mask == 0,
            -1e10
            #float("-inf"),
        )
        
        # attention
        attention = torch.softmax(scores, dim = -1)
#        print(f"attention: {attention}")

        # value: [B,H,T,T] @ [B,H,T,Dh] -> [B,T,T,Dh]
        out = attention @ v
        # [B,H,T,Dh] ->  [B,T,D]
        out = out.transpose(1, 2).contiguous()
        out = out.view(B,T,D)
#        print(f"attention block output: {torch.nonzero(torch.isnan(out))}")

        return self.out_proj(out)

# transformer:
class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio = 4):
        super().__init__()
        hidden = d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model)
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads:int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(d_model, n_heads, max_seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)
    def forward(self, x: torch.Tensor, mask):
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self,
            vocab_size :int,
            d_model: int,
            n_layers: int,
            n_heads: int,
            max_seq_len: int,
        ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.max_seq_len = max_seq_len
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                max_seq_len,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_linear = nn.Linear(d_model, vocab_size)

        #build causal mask
        mask = torch.tril(
            torch.ones(
                size = (max_seq_len,max_seq_len),
                dtype = torch.int)
        )

        self.register_buffer(
            "causal_mask",
            mask.view(1,1,max_seq_len,max_seq_len)
        )

    def forward(self,x: torch.Tensor, pad_mask: torch.Tensor):
        B,T = x.shape
        assert T <= self.max_seq_len
#        print(f"pad mask {torch.nonzero(pad_mask)}")
        # build combined mask: causal mask [1,1,max_seq_len, max_seq_len] + pad_mask[B,T] -> [B,1,T,T]
        pad_mask = pad_mask.unsqueeze(-1)
        pad_mask = pad_mask @ pad_mask.transpose(-1,-2)
        combined_mask = self.causal_mask[:,:,:T,:T] * pad_mask.unsqueeze(1)

        # embedding
        pos_seq = torch.arange(
            T,
            dtype = torch.int64,
            device = x.device
        )
        x = self.token_embedding(x) + self.position_embedding(pos_seq)

#        print(f"x after embedding: {torch.nonzero(torch.isnan(x))}")

        # transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x, combined_mask)
#            print(f"{i}th block: {torch.nonzero(torch.isnan(x))}")

#        print(f"x: {x}")
        # forward
        x = self.norm(x)
        logits = self.lm_linear(x)

        return logits


class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    def train_step(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)

#        print(f"x: {x}")
#        print(f"y: {y}")
        mask = mask.to(self.device)

        self.optimizer.zero_grad()

        logits = self.model(x, mask)
#        print(f"logits: {logits}")

        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            y.reshape(B * T),
            ignore_index = -100,
        )

        loss.backward()
        self.optimizer.step()

        return loss.item()


# Training scripts

# generate senteces
subject_pool = [
    "The researcher", "A student", "This model", "Every engineer",
    "Our lab", "The system", "A scientist", "The team"
]

verb_pool = [
    "tests", "improves", "trains", "evaluates", "refines", "compares",
    "optimizes", "studies", "scales", "debugs"
]

object_pool = [
    "a new transformer", "the experiment", "the dataset", "the baseline",
    "the optimizer", "the attention layer", "the language model",
    "the training loop"
]

adverb_pool = [
    "carefully", "quickly", "consistently", "efficiently", "robustly",
    "reproducibly", "smoothly", "accurately"
]

sentence_templates = [
    "{subject} {verb} {object}.",
    "{subject} {verb} {object} {adverb}.",
    "{subject} {verb} {object} during training.",
    "{subject} {verb} {object} with confidence.",
]

def make_sentence():
    subject = random.choice(subject_pool)
    verb = random.choice(verb_pool)
    obj = random.choice(object_pool)
    adverb = random.choice(adverb_pool)
    template = random.choice(sentence_templates)
    return template.format(subject=subject, verb=verb, object=obj, adverb=adverb)

sentences = [make_sentence() for _ in range(10000)]
chars = sorted(set(" ".join(sentences)))
vocab_size = len(chars) + 1   # +1 because your mapping often reserves 0 for padding
print(f"vocab_size is {vocab_size}")

dataset = VariableLengthDataset(sentences)
collator = PaddingCollator()
sampler = TokenBatchSampler(dataset,512)

loader = DataLoader(
    dataset,
    collate_fn = collator,
    batch_sampler = sampler,
    shuffle = False,
    pin_memory = True,
)

model = GPT(
    vocab_size = vocab_size,
    d_model = 128,
    max_seq_len = 512,
    n_layers = 4,
    n_heads = 4,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 3e-4,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

trainer = Trainer(
    model,
    optimizer,
    device,
)

epoches = 10

for epoch in range(epoches):
    loss_accu = 0.0
    for batch in loader:
        x = batch["input_ids"]
        y = batch["labels"]
        pad_mask = batch["pad_mask"]

        loss = trainer.train_step(x,y,pad_mask)
        loss_accu += loss
    print(f"epoch {epoch} | ave_loss: {loss_accu / len(loader)}")
#        print(f"epoch {epoch} | ave_loss: {loss_accu}")