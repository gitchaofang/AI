import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_step(self, x, y, mask):
        x = x.to(self.device)
        y = y.to(self.device)
        mask = mask.to(self.device)
        with torch.autocast(
            device_type = self.device,
            dtype = torch.bfloat16,
        ):
            logits = self.model(x, mask)

            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                y.reshape(B * T),
                ignore_index=-100,
            )

        return loss