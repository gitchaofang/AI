import torch
from torch.utils.data import DataLoader

from src.data.dataset import TextDataset
from src.data.variable_length_dataset import VariableLengthDataset
from src.data.collator import PaddingCollator
from src.data.sampler import BucketBatchSampler
from src.data.token_batch_sampler import TokenBatchSampler
from src.models.GPT import GPT
from src.training.trainer import Trainer

text = """
Machine learning research requires careful experiments.
Good engineering makes research reproducible and scalable.
"""


# variable set dataset
sequences = [
    torch.randint(
        0, 1000, (torch.randint(3,512,(1,)).item(),)
    )
    for _ in range(10000)
]
dataset = VariableLengthDataset(sequences)
collator = PaddingCollator(pad_token_id=0)
sampler = BucketBatchSampler(dataset= dataset,
                             batch_size=32,
                             shuffle = True,)

#dataset = TextDataset(text * 1000, seq_len = 64)

loader = DataLoader(
    dataset,
    batch_sampler = sampler,
    collate_fn=collator,
    shuffle = True,
)

model = GPT(
    verb_size = dataset.verb_size,
    d_model = 128,
    n_layerrs = 4,
    n_heads = 4,
)

optimizer = torch.optim_AdamW(
    model.parameters(),
    lr - 3e-4,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

trainer = Trainer(
    model,
    optimizer,
    device,
)

for epoch in range(5):
    for batch in loader:
        loss = trainer.train_step(batch["input_ids"],batch["attention_mask"],batch["mask"])

    print(f"epoch={epoch}, loss = {loss: .4f}")
    