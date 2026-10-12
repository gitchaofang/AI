import torch
from torch import nn


class MultimodalGPT(nn.Module):
    def __init__(self,vit, gpt,):
        super().__init__()
        self.vit = vit
        self.gpt = gpt
        self.encoder_out = None # most recent encoder out from vit. 
                                # It is usually used for generation 

    def reset_cache(self):
        self.encoder_out = None
        self.gpt.reset_cache()

    def forward(self, 
                patch_items,        #vit
                text_x,             #gpt
                text_pad_mask,      #gpt
                text_positions,     #gpt
                is_prefill=False,   #gpt
                is_generate=False): #gpt
        '''
        text_x: [B, T_q]
        text_pad_mask: [B, T_q]
        patch_items:
            "patched_input": [B, T_kv, in_channel * patch_size * patch_size]
            "pad_mask_patch": [B, T_kv]
            "patched_positions": [B, T_kv, 2]
        encoder_out:
            "patch_seq": patch_out: [B, T_kv, d_model_ca]
            "pad_mask": pad_mask_out: [B, T_kv]
        '''
        if is_generate and self.encoder_out is not None:
            # Reuse visual features for the same generation session.
            encoder_out = self.encoder_out
        else:
            assert is_generate, "generate without vision input"
            # Compute fresh visual features.
            encoder_out = self.vit(patch_items=patch_items)
            # Cache for subsequent generation calls.
            if is_prefill:
                self.encoder_out = encoder_out

        content = {
            "patches": encoder_out["patch_seq"],
            "mask": encoder_out["pad_mask"],
        }

        return self.gpt(
            x=text_x,
            pad_mask=text_pad_mask,
            positions=text_positions,
            content=content,
            is_prefill=is_prefill,
            is_generate=is_generate
        )