import torch
import torch.nn.functional as F

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