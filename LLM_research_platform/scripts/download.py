from pathlib import Path
import tarfile
import time
import shutil
import os

from huggingface_hub import HfApi, hf_hub_download


# ============================================================
# Configuration
# ============================================================

REPO_ID = "pixparse/cc3m-wds"
REPO_TYPE = "dataset"

DRIVE_ROOT = Path(
    "/content/drive/MyDrive/VLM_DATA/CC3M"
)

SHARD_DIR = DRIVE_ROOT / "shards"
DEST_DIR = DRIVE_ROOT / "normal"

# Temporary local disk
LOCAL_DIR = Path("/content/cc3m_tmp")

SHARD_DIR.mkdir(parents=True, exist_ok=True)
DEST_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_FILE = DEST_DIR / "progress.log"


# ============================================================
# Logging
# ============================================================

def log(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"

    print(msg, flush=True)

    with open(PROGRESS_FILE, "a") as f:
        f.write(msg + "\n")
        f.flush()


# ============================================================
# Get list of TAR files from Hugging Face
# ============================================================

log("Getting CC3M shard list from Hugging Face...")

api = HfApi()

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
)

tar_names = sorted(
    f for f in files
    if f.endswith(".tar")
)

log(f"Found {len(tar_names)} TAR shards on Hugging Face.")


# ============================================================
# Process one shard at a time
# ============================================================

for shard_index, repo_tar_name in enumerate(tar_names, start=1):

    tar_name = Path(repo_tar_name).name

    drive_tar = SHARD_DIR / tar_name

    done_marker = DEST_DIR / f".{tar_name}.done"

    extracting_marker = DEST_DIR / f".{tar_name}.extracting"

    local_tar = LOCAL_DIR / tar_name


    log("")
    log("=" * 70)
    log(
        f"SHARD {shard_index}/{len(tar_names)}: "
        f"{tar_name}"
    )
    log("=" * 70)


    # ========================================================
    # 1. Already completely extracted
    # ========================================================

    if done_marker.exists():

        log(
            f"SKIP {tar_name}: "
            f"extraction already completed."
        )

        continue


    # ========================================================
    # 2. Make sure TAR exists on Google Drive
    # ========================================================

    if drive_tar.exists():

        size_gb = drive_tar.stat().st_size / (1024 ** 3)

        log(
            f"TAR already exists on Drive: "
            f"{size_gb:.2f} GB"
        )

    else:

        # ====================================================
        # Download to LOCAL Colab disk first
        # ====================================================

        log(
            f"Downloading {tar_name} "
            f"to local Colab disk..."
        )

        download_start = time.time()

        try:

            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=repo_tar_name,

                # Hugging Face cache on local disk
                cache_dir="/content/huggingface_cache",
            )

            downloaded_path = Path(downloaded_path)

            elapsed = time.time() - download_start

            size_gb = (
                downloaded_path.stat().st_size
                / (1024 ** 3)
            )

            log(
                f"Download completed: "
                f"{size_gb:.2f} GB "
                f"in {elapsed / 60:.1f} min"
            )


            # =================================================
            # Copy completed TAR to Google Drive
            #
            # IMPORTANT:
            # Copy to .partial first.
            # Only rename after copy completes.
            # =================================================

            partial_tar = SHARD_DIR / f"{tar_name}.partial"

            log(
                f"Copying TAR to Google Drive..."
            )

            copy_start = time.time()

            shutil.copy2(
                downloaded_path,
                partial_tar,
            )

            # Verify size before considering it complete

            source_size = downloaded_path.stat().st_size
            copied_size = partial_tar.stat().st_size

            if source_size != copied_size:

                raise RuntimeError(
                    f"Copy verification failed: "
                    f"source={source_size}, "
                    f"destination={copied_size}"
                )

            # Atomic-ish rename within same Drive directory
            partial_tar.replace(drive_tar)

            elapsed = time.time() - copy_start

            log(
                f"TAR successfully stored on Drive "
                f"({elapsed / 60:.1f} min)"
            )


        except Exception as e:

            log(
                f"DOWNLOAD/COPY FAILED: {repr(e)}"
            )

            log(
                "This shard will be retried next time."
            )

            raise


    # ========================================================
    # 3. Verify TAR
    # ========================================================

    log(
        f"Checking TAR integrity: {tar_name}"
    )

    try:

        with tarfile.open(
            drive_tar,
            mode="r",
        ) as tar:

            # Do not extract yet.
            # Just make sure the archive can be opened.

            member_count = 0

            while True:

                member = tar.next()

                if member is None:
                    break

                member_count += 1

                if member_count % 10000 == 0:

                    log(
                        f"  TAR scan: "
                        f"{member_count:,} members..."
                    )

        log(
            f"TAR check OK: "
            f"{member_count:,} members"
        )

    except Exception as e:

        log(
            f"TAR integrity check FAILED: {repr(e)}"
        )

        log(
            "Deleting invalid TAR. "
            "It will be downloaded again."
        )

        drive_tar.unlink(missing_ok=True)

        raise


    # ========================================================
    # 4. Extract
    # ========================================================

    log(
        f"Starting extraction of {tar_name}"
    )

    extracting_marker.touch()

    extraction_start = time.time()

    extracted_count = 0
    skipped_count = 0

    try:

        with tarfile.open(
            drive_tar,
            mode="r",
        ) as tar:

            while True:

                member = tar.next()

                if member is None:
                    break

                # ------------------------------------------------
                # Determine target path
                # ------------------------------------------------

                target = DEST_DIR / member.name

                # ------------------------------------------------
                # If this file was already extracted before a
                # Colab disconnect, don't extract it again.
                # ------------------------------------------------

                if target.exists():

                    skipped_count += 1

                else:

                    # Python's data filter prevents dangerous
                    # paths such as ../../something
                    tar.extract(
                        member,
                        path=DEST_DIR,
                        filter="data",
                    )

                    extracted_count += 1


                # ------------------------------------------------
                # VERY IMPORTANT:
                #
                # Print progress periodically.
                #
                # This prevents a slow Google Drive operation
                # from looking like Colab has frozen.
                # ------------------------------------------------

                total_processed = (
                    extracted_count
                    + skipped_count
                )

                if total_processed % 1000 == 0:

                    elapsed = (
                        time.time()
                        - extraction_start
                    )

                    rate = (
                        total_processed / elapsed
                        if elapsed > 0
                        else 0
                    )

                    log(
                        f"  Extraction progress: "
                        f"{total_processed:,} members | "
                        f"new={extracted_count:,} | "
                        f"existing={skipped_count:,} | "
                        f"{rate:.1f} files/sec"
                    )


        # ====================================================
        # Extraction completed successfully
        # ====================================================

        elapsed = time.time() - extraction_start

        done_marker.touch()

        extracting_marker.unlink(
            missing_ok=True
        )

        log(
            f"EXTRACTION COMPLETE: {tar_name}"
        )

        log(
            f"  New files:      {extracted_count:,}"
        )

        log(
            f"  Existing files: {skipped_count:,}"
        )

        log(
            f"  Time:            {elapsed / 60:.1f} min"
        )


    except Exception as e:

        # ----------------------------------------------------
        # DO NOT create .done
        #
        # Therefore the shard will be resumed/retried.
        # ----------------------------------------------------

        extracting_marker.unlink(
            missing_ok=True
        )

        log(
            f"EXTRACTION FAILED: {repr(e)}"
        )

        log(
            "The shard is NOT marked as complete."
        )

        log(
            "Restarting the script will resume it."
        )

        raise


# ============================================================
# All shards finished
# ============================================================

log("")
log("=" * 70)
log("ALL CC3M SHARDS COMPLETED")
log("=" * 70)