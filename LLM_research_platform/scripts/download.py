from pathlib import Path
import tarfile
import time
from google.colab import drive # only run in colab
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download

DATA_DIR = Path("/content/drive/MyDrive/VLM_DATA/CC3M/shards")
DEST_DIR = Path("/content/drive/MyDrive/VLM_DATA/CC3M/normal")

DEST_DIR.mkdir(parents=True, exist_ok=True)

snapshot_download( 
    repo_id="pixparse/cc3m-wds", 
    repo_type="dataset", 
    local_dir=DATA_DIR, 
)

tar_files = sorted(DATA_DIR.glob("*.tar"))

progress_file = DEST_DIR / "progress.log"


def log(message):
    print(message, flush=True)
    with open(progress_file, "a") as f:
        f.write(message + "\n")
        f.flush()


log(f"Found {len(tar_files)} TAR files")

for i, tar_path in enumerate(tar_files, start=1):

    marker = DEST_DIR / f".{tar_path.name}.done"

    # Already completely extracted
    if marker.exists():
        log(f"[{i}/{len(tar_files)}] SKIP {tar_path.name}")
        continue

    log(f"[{i}/{len(tar_files)}] START {tar_path.name}")

    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(
                path=DEST_DIR,
                filter="data",
            )

        # IMPORTANT:
        # Only mark complete after extraction succeeds.
        marker.touch()

        log(f"[{i}/{len(tar_files)}] DONE {tar_path.name}")

    except Exception as e:
        log(f"[{i}/{len(tar_files)}] FAILED {tar_path.name}: {repr(e)}")
        log("Stopping so the failed shard can be retried.")
        raise

log("All shards completed.")
