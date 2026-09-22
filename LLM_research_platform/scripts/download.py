from google.colab import drive # only run in colab
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download


DATA_DIR = Path("/content/drive/MyDrive/VLM_DATA/CC3M/shards")
DATA_DIR.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="pixparse/cc3m-wds",
    repo_type="dataset",
    local_dir=DATA_DIR,
)

DEST_DIR = Path("/content/drive/MyDrive/VLM_DATA/CC3M/normal")

DEST_DIR.mkdir(parents=True, exist_ok=True)

tar_files = sorted(DATA_DIR.glob("*.tar"))

print(f"Found {len(tar_files)} TAR files")
print(f"Destination: {DEST_DIR}")

for i, tar_path in enumerate(tar_files, start=1):

    print(f"\n[{i}/{len(tar_files)}] Extracting {tar_path.name}")

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(
            path=DEST_DIR,
            filter="data",
        )

    print("Done")

print("\nAll shards extracted.")
