import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    def train_step(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()

        logits = self.model(x, mask)

        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            y.reshape(B * T),
            ignore_index = -100,
        )

        loss.backwards()
        self.optimizer.step()

        return loss.item()