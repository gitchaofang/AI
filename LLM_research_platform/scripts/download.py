from pathlib import Path
import tarfile
import time
from google.colab import drive # only run in colab
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download

SHARD_PATH = Path("/content/drive/MyDrive/protected/data/cc3m/shards")
TRAIN_PATH = Path("/content/drive/MyDrive/protected/data/cc3m/training")
VALIDATION_PATH= Path("/content/drive/MyDrive/protected/data/cc3m/validation")

# step 1: download all ".tar" shards
if False:
    snapshot_download(
        repo_id="pixparse/cc3m-wds",
        repo_type="dataset",
        local_dir=SHARD_PATH,
        allow_patterns=[
            "cc3m-train-*.tar",
            "cc3m-validation-*.tar",
        ]
    )

# step 2: build index dict and seperate ".tar" files in training and validation files:
cnt = 0
if True:
    for file_path in SHARD_PATH.iterdir():
        type = file_path.split('.')[0].split('-')[1]
        num = file_path.split('.')[0].split('-')[2]
        if file_path.is_file():
            print(f"{type}{num}")