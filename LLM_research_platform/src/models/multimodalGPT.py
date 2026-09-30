import torch
from torch import nn


'''
output from dataloader: 
        "patched_input": patched_input, #[B, max_len_patch ,C * patch_size * patch_size]
        "patch_positions": patch_positions, # [B, max_len_patch, 2]
        "pad_mask_patch": pad_mask_patch,# [B, max_len_patch]
        "caption_ids": caption_ids, # [B, max_len_text]
        "pad_mask_text": pad_mask_text, # [B, max_len_text]
        "positions_text": positions, #[B, max_len_text,1]
        "meta_data": meta_data, # list of dict. B dicts
        '''
class MultimodalGPT(nn.Module):
    def __init__(self,vit, gpt, is_prefill=False, is_generate=False):
        super().__init__()
        self.vit = vit
        self.gpt = gpt
        self.is_prefill=is_prefill
        self.is_generate=is_generate
    def forward(self, patch_items, text_x, text_pad_mask, text_positions):
        '''
        text_x: [B, T_q]
        text_pad_mask: [B, T_q]
        patch_items:
            "patched_input": [B, T_q, in_channel * patch_size * patch_size]
            "pad_mask_patch": [B, T_q, 2]
        encoder_out:
            "patch_seq": patch_out: [B, T_kv, d_model_ca]
            "pad_mask": pad_mask_out: [B, T_kv]
        '''
        encoder_out = self.vit(
            patch_items=patch_items
        )
        content = {"patches": encoder_out["patch_seq"],
                   "mask": encoder_out["pad_mask"]}
        logits = self.gpt(
            x=text_x,
            pad_mask=text_pad_mask,
            positions=text_positions,
            content=content,
            is_prefill=self.is_prefill,
            is_generate=self.is_generate
        )
        return logits