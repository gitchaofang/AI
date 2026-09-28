```python
from pathlib import Path
import tarfile
import time
import json
import shutil

from huggingface_hub import HfApi, hf_hub_download


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(
    "/content/drive/MyDrive/VLM_DATA/CC3M/shards"
)

DEST_DIR = Path(
    "/content/drive/MyDrive/VLM_DATA/CC3M/normal"
)

VLM_DATA_DIR = Path(
    "/content/drive/MyDrive/VLM_DATA"
)

# Temporary files stay on Colab local disk
TMP_DIR = Path("/content/cc3m_tmp")

# Progress log stays on local disk
PROGRESS_FILE = TMP_DIR / "progress.log"


# ============================================================
# Create directories
# ============================================================

DATA_DIR.mkdir(parents=True, exist_ok=True)
DEST_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Logging
# ============================================================

def log(message):

    print(message, flush=True)

    try:
        with open(PROGRESS_FILE, "a") as f:
            f.write(message + "\n")
            f.flush()
    except OSError:
        pass


# ============================================================
# Get list of TAR files from Hugging Face
# ============================================================

log("Getting CC3M shard list from Hugging Face...")

api = HfApi()

files = api.list_repo_files(
    repo_id="pixparse/cc3m-wds",
    repo_type="dataset",
)

tar_files = sorted(
    f for f in files
    if f.endswith(".tar")
)

log(f"Found {len(tar_files)} TAR files.")


# ============================================================
# Load existing index
# ============================================================

index_file = VLM_DATA_DIR / "cc3m_index.json"

if index_file.exists():

    with open(index_file, "r") as f:
        index = json.load(f)

else:

    index = {}


# ============================================================
# Determine existing sample count
#
# This lets batch numbering continue after a restart.
# ============================================================

sample_count = len(index)

log(f"Existing samples in index: {sample_count:,}")


# ============================================================
# Process each TAR
# ============================================================

for i, repo_tar in enumerate(tar_files, start=1):

    tar_name = Path(repo_tar).name

    drive_tar = DATA_DIR / tar_name

    done_marker = DEST_DIR / f".{tar_name}.done"


    # --------------------------------------------------------
    # Already completely processed
    # --------------------------------------------------------

    if done_marker.exists():

        log(
            f"[{i}/{len(tar_files)}] "
            f"SKIP {tar_name}"
        )

        continue


    log(
        f"[{i}/{len(tar_files)}] "
        f"START {tar_name}"
    )


    # ========================================================
    # Download TAR to local Colab disk
    # ========================================================

    if not drive_tar.exists():

        log(
            f"Downloading {tar_name} "
            f"to local Colab disk..."
        )

        local_tar = hf_hub_download(
            repo_id="pixparse/cc3m-wds",
            repo_type="dataset",
            filename=repo_tar,
            cache_dir=TMP_DIR,
        )

        local_tar = Path(local_tar)


        # ----------------------------------------------------
        # Copy completed TAR to Google Drive
        #
        # Use .partial so an interrupted copy is never
        # mistaken for a completed TAR.
        # ----------------------------------------------------

        partial_tar = DATA_DIR / f"{tar_name}.partial"

        log(
            f"Copying {tar_name} "
            f"to Google Drive..."
        )

        shutil.copy2(
            local_tar,
            partial_tar,
        )

        partial_tar.rename(drive_tar)

        log(
            f"{tar_name} saved to Google Drive."
        )


    else:

        log(
            f"{tar_name} already exists "
            f"on Google Drive."
        )


    # ========================================================
    # Extract
    # ========================================================

    log(
        f"Extracting {tar_name}..."
    )

    start_time = time.time()


    try:

        with tarfile.open(
            drive_tar,
            "r",
        ) as tar:

            # ------------------------------------------------
            # Group TAR members by sample stem
            #
            # Example:
            #
            # abc123.jpg
            # abc123.txt
            #
            # become one sample.
            # ------------------------------------------------

            samples = {}

            for member in tar:

                if not member.isfile():
                    continue

                stem = Path(member.name).stem

                if stem not in samples:
                    samples[stem] = []

                samples[stem].append(member)


            # ------------------------------------------------
            # Extract samples
            # ------------------------------------------------

            for stem, members in samples.items():

                batch_number = sample_count // 500 + 1

                batch_dir = (
                    DEST_DIR
                    / f"batch{batch_number}"
                )

                batch_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )


                # --------------------------------------------
                # Extract image + metadata into same directory
                # --------------------------------------------

                for member in members:

                    output_path = (
                        batch_dir
                        / Path(member.name).name
                    )

                    if output_path.exists():
                        continue

                    tar.extract(
                        member,
                        path=batch_dir,
                        filter="data",
                    )


                # --------------------------------------------
                # Add sample to index
                # --------------------------------------------

                index[stem] = {
                    "batch": f"batch{batch_number}",
                    "path": str(batch_dir),
                }


                sample_count += 1


                # --------------------------------------------
                # Save index every 500 samples
                # --------------------------------------------

                if sample_count % 500 == 0:

                    with open(
                        index_file,
                        "w",
                    ) as f:

                        json.dump(
                            index,
                            f,
                            indent=2,
                        )

                    log(
                        f"  "
                        f"{sample_count:,} "
                        f"total samples processed"
                    )


        # ====================================================
        # Save index after TAR completes
        # ====================================================

        with open(index_file, "w") as f:

            json.dump(
                index,
                f,
                indent=2,
            )


        # ====================================================
        # Mark TAR complete
        #
        # IMPORTANT: this happens only after extraction
        # and index update succeed.
        # ====================================================

        done_marker.touch()

        elapsed = time.time() - start_time

        log(
            f"[{i}/{len(tar_files)}] "
            f"DONE {tar_name} "
            f"({elapsed / 60:.1f} min)"
        )


    except Exception as e:

        log(
            f"[{i}/{len(tar_files)}] "
            f"FAILED {tar_name}: {repr(e)}"
        )

        log(
            "Restart the script to resume."
        )

        raise


log(
    f"ALL SHARDS COMPLETED. "
    f"Total samples: {sample_count:,}"
)
```

