from pathlib import Path
import tarfile
import shutil
import time

from huggingface_hub import HfApi, hf_hub_download


# ============================================================
# Paths
# ============================================================

SHARD_DIR = Path(
    "/content/drive/MyDrive/VLM_DATA/CC3M/shards"
)

DEST_DIR = Path(
    "/content/drive/MyDrive/VLM_DATA/CC3M/normal"
)

TMP_DIR = Path("/content/cc3m_tmp")

SHARD_DIR.mkdir(parents=True, exist_ok=True)
DEST_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)


# Local log -- NOT Google Drive
LOG_FILE = TMP_DIR / "progress.log"


def log(msg):
    print(msg, flush=True)

    try:
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")
    except OSError:
        pass


# ============================================================
# Get list of TAR files
# ============================================================

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
# Process one TAR at a time
# ============================================================

for i, repo_tar in enumerate(tar_files, 1):

    tar_name = Path(repo_tar).name

    drive_tar = SHARD_DIR / tar_name

    done = DEST_DIR / f".{tar_name}.done"


    # --------------------------------------------------------
    # Already extracted
    # --------------------------------------------------------

    if done.exists():

        log(
            f"[{i}/{len(tar_files)}] "
            f"SKIP {tar_name}"
        )

        continue


    # --------------------------------------------------------
    # Download TAR if it isn't already on Drive
    # --------------------------------------------------------

    if not drive_tar.exists():

        log(
            f"[{i}/{len(tar_files)}] "
            f"Downloading {tar_name}..."
        )

        local_tar = hf_hub_download(
            repo_id="pixparse/cc3m-wds",
            repo_type="dataset",
            filename=repo_tar,
            cache_dir=TMP_DIR,
        )

        local_tar = Path(local_tar)

        log(
            f"Copying {tar_name} to Google Drive..."
        )

        # Copy to .partial first
        partial = SHARD_DIR / f"{tar_name}.partial"

        shutil.copy2(
            local_tar,
            partial,
        )

        # Rename only after copy finishes
        partial.rename(drive_tar)

        log(
            f"{tar_name} saved."
        )

    else:

        log(
            f"[{i}/{len(tar_files)}] "
            f"{tar_name} already downloaded."
        )


    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    log(
        f"Extracting {tar_name}..."
    )

    start = time.time()

    extracted = 0
    skipped = 0

    try:

        with tarfile.open(
            drive_tar,
            "r",
        ) as tar:

            for member in tar:

                target = DEST_DIR / member.name


                # Already extracted before disconnect?
                if target.exists():

                    skipped += 1

                else:

                    tar.extract(
                        member,
                        path=DEST_DIR,
                        filter="data",
                    )

                    extracted += 1


                # Print progress every 1,000 files
                if (
                    extracted + skipped
                ) % 1000 == 0:

                    log(
                        f"  "
                        f"{extracted + skipped:,} "
                        f"members processed "
                        f"({extracted:,} new, "
                        f"{skipped:,} existing)"
                    )


        # ----------------------------------------------------
        # Only now mark the shard complete
        # ----------------------------------------------------

        done.touch()

        log(
            f"DONE {tar_name} "
            f"({time.time() - start:.1f} sec)"
        )


    except Exception as e:

        log(
            f"FAILED {tar_name}: {repr(e)}"
        )

        log(
            "Restart the script to resume."
        )

        raise


log("ALL SHARDS COMPLETED!")