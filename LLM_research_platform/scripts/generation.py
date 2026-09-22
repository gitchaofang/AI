import torch
from src.models.self_attention import SelfAttention
from src.models.GPT import GPT
from src.models.all_in_one import GPT

@torch.no_grad()
def generate(
    model,
    input_ids,  # [B, T_prompt], right-padded prompts
    pad_mask, # 1: valid, 0 padding
    max_new_tokens, # maximum number tokens to generate
    eos_token_id=None,
):  # return [B, T_prompt + max_new_tokens]
   
    model.eval()

    model.reset_cache()

    input_ids = input_ids.to(next(model.parameters()).device)
    pad_mask = pad_mask.to(input_ids.device)

    B, T_prompt = input_ids.shape

    logits = model(
        input_ids,
        pad_mask,
        is_prefill=True,
        is_generate=False,
    )

    lengths = pad_mask.long().sum(dim=1)

    last_indices = lengths - 1

    batch_indices = torch.arange(
        B,
        device=input_ids.device,
    )

    next_logits = logits[
        batch_indices,
        last_indices,
        :,
    ]

    finished = torch.zeros(
        B,
        dtype=torch.bool,
        device=input_ids.device,
    ) # [B]


    for _ in range(max_new_tokens):

        next_token = torch.argmax(
            next_logits,
            dim=-1,
        ) #[B]

        next_token = next_token.unsqueeze(1) #[B,1]

        if eos_token_id is not None:

            # Sequences that were already finished should
            # generate EOS repeatedly.
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(
                    next_token,
                    eos_token_id,
                ),
                next_token,
            )

            finished = finished | (
                next_token.squeeze(1) == eos_token_id
            )


        output_ids = torch.cat(
            [output_ids, next_token],
            dim=1,
        )


        next_pad_mask = torch.ones(
            B,
            1,
            dtype=pad_mask.dtype,
            device=pad_mask.device,
        )

        logits = model(
            next_token,
            next_pad_mask,
            is_prefill=False,
            is_generate=True,
        )

        next_logits = logits[:, -1, :]

        if eos_token_id is not None and torch.all(finished):
            break

    return output_ids