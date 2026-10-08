
from pathlib import Path
import json

import yaml
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


PAD_ID = 0
EOS_ID = 1

PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|eos|>"

FILE_DIR = Path("/content/token_train")
COLAB_YAML_PATH = Path(
    "/content/AI/LLM_research_platform/configs/gpt.yaml"
)

with COLAB_YAML_PATH.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


class RegexTokenizer:

    def __init__(
        self,
        file_dir=FILE_DIR,
        pattern=None,
        vocab_size=None,
    ):
        self.file_dir = Path(file_dir)
        self.file_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer_path = self.file_dir / "tokenizer.json"

        self.vocab_size = (
            vocab_size
            if vocab_size is not None
            else config["tokenizer"]["vocab_size"]
        )

        self.tokenizer = None

        print(f"Vocabulary size: {self.vocab_size}")

    def get_vocab_size(self):
        if self.tokenizer is None:
            return self.vocab_size

        return self.tokenizer.get_vocab_size()

    def train(self):
        # Load an existing tokenizer instead of retraining.
        if self.tokenizer_path.is_file():
            self.tokenizer = Tokenizer.from_file(
                str(self.tokenizer_path)
            )

            print(f"Loaded tokenizer from {self.tokenizer_path}")

            assert self.tokenizer.token_to_id(PAD_TOKEN) == PAD_ID
            assert self.tokenizer.token_to_id(EOS_TOKEN) == EOS_ID
            return

        files = sorted(self.file_dir.glob("*.txt"))

        if not files:
            raise FileNotFoundError(
                f"No .txt training files found in {self.file_dir}"
            )

        print(f"Training on {len(files)} text files")

        # Create a byte-level BPE tokenizer.
        self.tokenizer = Tokenizer(
            BPE(unk_token=None)
        )

        self.tokenizer.pre_tokenizer = ByteLevel(
            add_prefix_space=False
        )
        self.tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            initial_alphabet=ByteLevel.alphabet(),
            special_tokens=[
                AddedToken(PAD_TOKEN, special=True),
                AddedToken(EOS_TOKEN, special=True),
            ],
            show_progress=True,
        )

        self.tokenizer.train(
            files=[str(path) for path in files],
            trainer=trainer,
        )

        # Verify the special token IDs expected by the model.
        assert self.tokenizer.token_to_id(PAD_TOKEN) == PAD_ID
        assert self.tokenizer.token_to_id(EOS_TOKEN) == EOS_ID

        self.tokenizer.save(str(self.tokenizer_path))

        print(f"Training complete")
        print(f"Actual vocabulary size: {self.get_vocab_size()}")
        print(f"Saved tokenizer to {self.tokenizer_path}")

    def encode(self, text):
        if self.tokenizer is None:
            raise RuntimeError("Call train() before encode().")

        # Return a flat list of token IDs, followed by EOS.
        ids = self.tokenizer.encode(text).ids

        # Do not duplicate EOS if the text already ends in EOS.
        if not ids or ids[-1] != EOS_ID:
            ids.append(EOS_ID)

        return ids

    def decode(self, ids):
        if self.tokenizer is None:
            raise RuntimeError("Call train() before decode().")

        # Stop at EOS; omit padding.
        cleaned_ids = []

        for idx in ids:
            if idx == EOS_ID:
                break
            if idx == PAD_ID:
                continue
            cleaned_ids.append(int(idx))

        return self.tokenizer.decode(
            cleaned_ids,
            skip_special_tokens=True,
        )
