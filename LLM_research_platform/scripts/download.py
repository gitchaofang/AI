from pathlib import Path
import tarfile
import time
from google.colab import drive # only run in colab
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download

SHARD_PATH = Path( "/content/drive/MyDrive/protected/data/cc3m/shards")

snapshot_download(
    repo_id="pixparse/cc3m-wds",
    repo_type="dataset",
    local_dir=SHARD_PATH,
    allow_patterns="cc3m-train-*.tar",
)