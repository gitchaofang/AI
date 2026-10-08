import torch
import pytest
import os
import regex as re
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.regex_tokenizer import RegexTokenizer
from src.data.all_in_one import TextDecodeDataset
from src.data.dataset import ImageDataset
from src.data.image_dataset_sampler import ImageDatasetBatchSampler
from src.data.all_in_one import TokenBatchSampler
from src.data.all_in_one import PaddingCollator
from src.data.collator import VitCollator
from src.base.self_attention import SelfAttention
from src.models import GPT

DATA_DIR =  "/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/src/data/input_data/data/"
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
ENCODE_DECODE_FILENAME = "swift.txt"
DATASET_INPUT_FILENAME = "swift.txt"
B = 10
T = 26
D = 32

# load config file
LOCAL_YAML_PATH = Path("/Users/chaofang/Documents/coding_playground/GitHub/AI/LLM_research_platform/configs/vit.yaml")
COLAB_YAML_PATH = Path("/content/AI/LLM_research_platform/configs/vit.yaml")
CACHE_RESERVE_GB = 50
# load yaml config
with open(COLAB_YAML_PATH,"r") as f:
    vit_config = yaml.safe_load(f)

@pytest.fixture
def tokenizer():
    print(f"train tokenizer")
    tokenizer = RegexTokenizer()
    tokenizer.train()
    return tokenizer

@pytest.fixture
def test_text():
    print(f"build text_text")
    read_path = os.path.join(DATA_DIR, ENCODE_DECODE_FILENAME)
    with open(read_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

@pytest.fixture
# model input package:
# x: input_tensor
# positions: tentor for RoPE
# pad_mask: padding mask
# rope_dims: dims for RoPE
def model_param():
    print(f"build model params")
    # x [B,T,D]
    # pad_mask [B,T]
    x = torch.randint(2,255,(B,T,D), dtype = torch.int64)
    pad_mask = torch.ones((B,T), dtype = torch.int)
    # add padding
    x[:,-10:,:] = 0
    pad_mask[:,-10:] = 0

    # positions: [B,T,M] M = 3
    positions = torch.full((B,T,3), -1, dtype = torch.int64)
    
    for j in range(B):
        coord = torch.randint(0,224,(3,), dtype = torch.int)
        positions[j,:-10,:] = coord

    # rope_dims
    rope_dims = [4,6,6]

    return {"ids": x,
            "pos": positions,
            "mask": pad_mask,
            "rdim": rope_dims}

# ------dataset---------
def test_encode_decode(tokenizer, test_text):
    print(f"start testing")
    encoded_ids = tokenizer.encode(test_text)
    decoded_chunks = [tokenizer.decode(ids) for ids in encoded_ids]
    decoded_text = "".join(decoded_chunks)
    print(f"{decoded_text} \n {test_text}")
    assert decoded_text == test_text


def test_dataset_shift(tokenizer):
    dataset = TextDecodeDataset(
        tokenizer=tokenizer,
        max_len=512,
        filename=DATASET_INPUT_FILENAME
    )

    for i in range(len(dataset)):
        item = dataset[i]

        x = item["input_ids"]
        y = item["labels"]

        assert len(x) == len(y), f"Length mismatch at {i}"

        if not torch.equal(x[1:], y[:-1]):
            print(f"FAILED at dataset index {i}")

            for j in range(min(len(x[1:]), len(y[:-1]))):
                if x[1:][j] != y[:-1][j]:
                    print(
                        f"Mismatch at position {j}: "
                        f"x={x[1:][j].item()}, "
                        f"y={y[:-1][j].item()}"
                    )
                    break

            raise AssertionError

def test_collator(tokenizer):
    dataset = TextDecodeDataset(
        tokenizer=tokenizer,
        max_len=512,
        filename=DATASET_INPUT_FILENAME
    )

    collator = PaddingCollator()

    samples = [
        dataset[0],
        dataset[1],
    ]

    batch = collator(samples)

    x = batch["input_ids"]
    y = batch["labels"]
    mask = batch["pad_mask"]

    for i, item in enumerate(samples):
        length = len(item["input_ids"])

        assert torch.equal(
            x[i, :length],
            item["input_ids"]
        )

        assert torch.equal(
            y[i, :length],
            item["labels"]
        )

def test_dataloader(tokenizer):
    dataset = TextDecodeDataset(
        tokenizer=tokenizer,
        max_len=512,
        filename=DATASET_INPUT_FILENAME
    )

    batch_sampler = TokenBatchSampler(
        dataset=dataset,
        batch_size=8,
        shuffle=False
    )

    collator = PaddingCollator()

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        pin_memory=True
    )

    for batch_indices, batch in zip(batch_sampler, loader):

        x = batch["input_ids"]
        y = batch["labels"]
        mask = batch["pad_mask"]

        # Check every sample against the original Dataset
        for row, dataset_idx in enumerate(batch_indices):
            original = dataset[dataset_idx]

            length = len(original["input_ids"])

            assert torch.equal(
                x[row, :length],
                original["input_ids"]
            )

            assert torch.equal(
                y[row, :length],
                original["labels"]
            )

def test_text_dataset(tokenizer):
    dataset = TextDecodeDataset(tokenizer=tokenizer,
                                max_len=512,
                                filename=DATASET_INPUT_FILENAME)
    batch_sampler = TokenBatchSampler(dataset=dataset, 
                                batch_size=8,
                                shuffle=False)
    collator = PaddingCollator()
    loader = DataLoader(dataset=dataset,
                        collate_fn = collator,
                        batch_sampler = batch_sampler,
                        pin_memory = True)

    for batch in loader:
        x = batch["input_ids"]
        y = batch["labels"]
        mask = batch["pad_mask"]

        for i in range(x.size(0)):
            valid_len = mask[i].sum().item()
            if valid_len > 1:
                assert torch.equal(x[i, 1:valid_len],y[i,:valid_len - 1])

def test_image_dataset(tokenizer):
    print("image test starts")

    # ============================================================
    # 1. Dataset
    # ============================================================

    dataset = ImageDataset(
        data_dir=vit_config["data"]["data_path"],
        tokenizer=tokenizer,
        image_only=vit_config["data"]["image_only"],
        for_training=True,
    )

    print(f"dataset done: {len(dataset)} samples")

    assert len(dataset) > 0

    # ============================================================
    # 2. Batch sampler
    # ============================================================

    batch_sampler = ImageDatasetBatchSampler(
        dataset=dataset,
        batch_size=vit_config["data"]["batch_size"],
        shuffle=vit_config["data"]["shuffle"],
    )

    print("sampler done")

    # ============================================================
    # 3. Collator
    # ============================================================

    collator = VitCollator(
        token_pad=0,
        image_pad=0.0,
        label_pad=-100,
        image_only=vit_config["data"]["image_only"],
    )

    print("collator done")

    # ============================================================
    # 4. DataLoader
    # ============================================================

    loader = DataLoader(
        dataset=dataset,
        collate_fn=collator,
        batch_sampler=batch_sampler,
        pin_memory=True,
    )

    print("iteration starts")

    # Only test one batch.
    # Don't iterate through the entire CC3M dataset in a unit test.
    batch = next(iter(loader))

    print("batch loaded")

    # ============================================================
    # 5. Check batch keys
    # ============================================================

    expected_keys = {
        "patched_input",
        "patch_positions",
        "pad_mask_patch",
        "caption_ids",
        "caption_ids_label",
        "pad_mask_text",
        "positions_text",
        "meta_data",
    }

    assert set(batch.keys()) == expected_keys, (
        f"Unexpected batch keys: {batch.keys()}"
    )

    # ============================================================
    # 6. Get batch tensors
    # ============================================================

    patched_input = batch["patched_input"]
    patch_positions = batch["patch_positions"]
    pad_mask_patch = batch["pad_mask_patch"]

    caption_ids = batch["caption_ids"]
    caption_ids_label = batch["caption_ids_label"]
    pad_mask_text = batch["pad_mask_text"]
    positions_text = batch["positions_text"]

    meta_data = batch["meta_data"]

    # ============================================================
    # 7. Check image tensor dimensions
    # ============================================================

    assert patched_input.ndim == 3, (
        f"patched_input should be [B, N, D], "
        f"got {patched_input.shape}"
    )

    B, N, D = patched_input.shape

    print(
        f"image: B={B}, max_patches={N}, patch_dim={D}"
    )

    assert B > 0
    assert N > 0
    assert D > 0

    # patch_positions:
    # [B, N, 2]
    assert patch_positions.shape == (B, N, 2), (
        f"patch_positions shape mismatch: "
        f"{patch_positions.shape}"
    )

    # image padding mask:
    # [B, N]
    assert pad_mask_patch.shape == (B, N), (
        f"pad_mask_patch shape mismatch: "
        f"{pad_mask_patch.shape}"
    )

    # ============================================================
    # 8. Check image tensor dtypes
    # ============================================================

    assert patched_input.dtype == torch.float32, (
        f"patched_input dtype should be float32, "
        f"got {patched_input.dtype}"
    )

    assert patch_positions.dtype == torch.long, (
        f"patch_positions dtype should be long, "
        f"got {patch_positions.dtype}"
    )

    assert pad_mask_patch.dtype == torch.bool, (
        f"pad_mask_patch dtype should be bool, "
        f"got {pad_mask_patch.dtype}"
    )

    # ============================================================
    # 9. Check image values
    # ============================================================

    assert torch.isfinite(patched_input).all(), (
        "patched_input contains NaN or Inf"
    )

    assert torch.isfinite(
        patch_positions.float()
    ).all(), (
        "patch_positions contains NaN or Inf"
    )

    # ============================================================
    # 10. Every image must contain real patches
    # ============================================================

    real_patch_count = (~pad_mask_patch).sum(dim=1)

    assert torch.all(real_patch_count > 0), (
        "At least one image contains only padding"
    )

    print(
        f"real patches per image: "
        f"min={real_patch_count.min().item()}, "
        f"max={real_patch_count.max().item()}"
    )

    # ============================================================
    # 11. Check image padding values
    # ============================================================

    if pad_mask_patch.any():
        padded_values = patched_input[pad_mask_patch]

        assert torch.all(padded_values == 0.0), (
            "Image padding is not zero"
        )

    # ============================================================
    # 12. Check text tensor dimensions
    # ============================================================

    assert caption_ids.ndim == 2, (
        f"caption_ids should be [B, T], "
        f"got {caption_ids.shape}"
    )

    B_text, T = caption_ids.shape

    print(
        f"text: B={B_text}, max_length={T}"
    )

    # Text batch size must equal image batch size.
    assert B_text == B, (
        f"Image batch size {B} != "
        f"text batch size {B_text}"
    )

    # labels
    assert caption_ids_label.shape == (B, T), (
        f"caption_ids_label shape mismatch: "
        f"{caption_ids_label.shape}"
    )

    # text mask
    assert pad_mask_text.shape == (B, T), (
        f"pad_mask_text shape mismatch: "
        f"{pad_mask_text.shape}"
    )

    # text positions
    assert positions_text.shape == (B, T), (
        f"positions_text shape mismatch: "
        f"{positions_text.shape}"
    )

    # ============================================================
    # 13. Check text dtypes
    # ============================================================

    assert caption_ids.dtype == torch.int64, (
        f"caption_ids dtype should be long, "
        f"got {caption_ids.dtype}"
    )

    assert caption_ids_label.dtype == torch.int64, (
        f"caption_ids_label dtype should be long, "
        f"got {caption_ids_label.dtype}"
    )

    assert positions_text.dtype == torch.int64, (
        f"positions_text dtype should be long, "
        f"got {positions_text.dtype}"
    )

    # ============================================================
    # 14. Check text values
    # ============================================================

    assert torch.isfinite(
        caption_ids.float()
    ).all(), (
        "caption_ids contains NaN or Inf"
    )

    assert torch.isfinite(
        caption_ids_label.float()
    ).all(), (
        "caption_ids_label contains NaN or Inf"
    )

    # ============================================================
    # 15. Check token ID range
    # ============================================================

    valid_tokens = caption_ids[~pad_mask_text]

    assert valid_tokens.numel() > 0, (
        "There are no valid text tokens"
    )

    vocab_size = tokenizer.get_vocab_size()

    assert valid_tokens.min() >= 1, (
        f"Invalid token ID: {valid_tokens.min().item()}"
    )

    assert valid_tokens.max() < vocab_size, (
        f"Token ID {valid_tokens.max().item()} "
        f">= vocab size {vocab_size}"
    )

    # ============================================================
    # 16. Check text padding
    # ============================================================

    if pad_mask_text.any():

        # PAD token should be 0.
        padded_input_tokens = caption_ids[pad_mask_text]

        assert torch.all(
            padded_input_tokens == 0
        ), (
            "caption_ids contains non-zero values "
            "at padded positions"
        )

        # Labels at padding positions should be -100
        # so CrossEntropyLoss ignores them.
        padded_labels = caption_ids_label[pad_mask_text]

        assert torch.all(
            padded_labels == -100
        ), (
            "caption_ids_label contains values other "
            "than -100 at padded positions"
        )

    # ============================================================
    # 17. Check positions_text
    # ============================================================

    expected_positions = torch.arange(
        T,
        device=positions_text.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(B, T)

    assert torch.equal(
        positions_text,
        expected_positions,
    ), (
        f"positions_text is incorrect:\n"
        f"expected:\n{expected_positions}\n"
        f"got:\n{positions_text}"
    )

    # ============================================================
    # 18. Check metadata
    # ============================================================

    assert len(meta_data) == B, (
        f"metadata length {len(meta_data)} "
        f"!= batch size {B}"
    )

    for item in meta_data:
        assert isinstance(item, dict), (
            f"metadata item should be dict, "
            f"got {type(item)}"
        )

        # These are the fields you described in your CC3M metadata.
        required_fields = {
            "caption",
            "url",
            "key",
        }

        assert required_fields.issubset(item.keys()), (
            f"Missing metadata fields. "
            f"Expected {required_fields}, "
            f"got {item.keys()}"
        )

    # ============================================================
    # 19. Final summary
    # ============================================================

    print(
        f"batch test passed: "
        f"B={B}, "
        f"image patches={N}, "
        f"patch dim={D}, "
        f"text length={T}"
    )
        

# ----------model--------------
def test_Selfattention(model_param):
    x = model_param["ids"]
    positions = model_param["pos"]
    pad_mask = model_param["mask"]
    rope_dims = model_param["rdim"]

    self_attention = SelfAttention(d_model = D,
                                   n_heads = 2,
                                   max_seq_len=T,
                                   rope_dims= rope_dims,
                                   RoPE = True)

    result_attention = self_attention(x, pad_mask, positions)
    assert result_attention.shape == x.shape
    assert torch.isfinite(result_attention).all()

