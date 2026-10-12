import torch
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train_step(self, patched_items, text_x, text_pad_mask,text_positions, is_prefill, is_generate, y,):

        logits = self.model(patched_items=patched_items,
                            text_x=text_x,
                            text_pad_mask=text_pad_mask,
                            text_positions=text_positions,
                            is_prefill=is_prefill,
                            is_generate=is_generate,)

        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            y.reshape(B * T),
            ignore_index=-100,
        )

        return loss