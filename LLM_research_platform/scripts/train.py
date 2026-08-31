import torch
from torch.utils.data import DataLoader

import random
from src.data.dataset import VariableLengthDataset
from src.data.collator import PaddingCollator
from src.data.token_batch_sampler import TokenBatchSampler
from src.models.GPT import GPT
from src.training.trainer import Trainer

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

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

model = GPT(
    vocab_size = vocab_size,
    d_model = 128,
    max_seq_len = 512,
    n_layers = 2,
    n_heads = 4,
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

epoches = 10
accumulation_steps = 16
for epoch in range(epoches):
    model.train()
    optimizer.zero_grad()
    loss_accu = 0.0
    step_count = 0

    for i, batch in enumerate(loader):
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        pad_mask = batch["pad_mask"].to(device)

        loss = trainer.train_step(x, y, pad_mask) / accumulation_steps
        loss.backward()

        loss_accu += loss.item() * accumulation_steps
        step_count += 1

        if(i % 50 == 0):
            print(f"epoch {epoch} | batch {i} | batch_size {len(batch)} | loss: {loss.item() * accumulation_steps}")

        if step_count % accumulation_steps == 0 or i == len(loader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            step_count = 0

    print(f"epoch {epoch} | ave_loss: {loss_accu / len(loader)}")
