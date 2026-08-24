import torch
from torch.utils.data import DataLoader

from src.data.dataset import TextDataset
from src.models.gpt import GPT
from src.training.trainer import Trainer

text = """
Machine learning research requires careful experiments.
Good engineering makes research reproducible and scalable.
"""

dataset = TextDataset(text * 1000, seq_len = 64)

loader = DataLoader(
    dataset,
    batch_size = 32,
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
    for x , y in loader:
        loss = trainer.train_step(x,y)

    print(f"epoch={epoch}, loss = {loss: .4f}")
    