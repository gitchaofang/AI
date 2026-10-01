import torch
from torch import nn


class MultimodalGPT(nn.Module):
    def __init__(self,vit, gpt,):
        super().__init__()
        self.vit = vit
        self.gpt = gpt
        self.encoder_out = None

    def reset_cache(self):
        self.encoder_out = None
        self.gpt.reset_cache()

    def forward(self, 
                patch_items, 
                text_x, 
                text_pad_mask, 
                text_positions,
                is_prefill=False, 
                is_generate=False):
        '''
        text_x: [B, T_q]
        text_pad_mask: [B, T_q]
        patch_items:
            "patched_input": [B, T_kv, in_channel * patch_size * patch_size]
            "pad_mask_patch": [B, T_kv]
        encoder_out:
            "patch_seq": patch_out: [B, T_kv, d_model_ca]
            "pad_mask": pad_mask_out: [B, T_kv]
        '''
        if (not is_generate) or (self.encoder_out is None):
            self.encoder_out = self.vit(
                patch_items=patch_items
            )
        content = {"patches": self.encoder_out["patch_seq"],
                   "mask": self.encoder_out["pad_mask"]}
        logits = self.gpt(
            x=text_x,
            pad_mask=text_pad_mask,
            positions=text_positions,
            content=content,
            is_prefill=is_prefill,
            is_generate=is_generate
        )
        return logits