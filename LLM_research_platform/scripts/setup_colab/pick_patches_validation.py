
import shutil
from pathlib import Path
from PIL import Image

PATCH_SIZE = 16
MAX_PATCHES = 2000

source_dir = Path("/content/cache/validation/")
target_dir = source_dir / "large_images"

target_dir.mkdir(parents=True, exist_ok=True)

moved_count = 0
skipped_count = 0
processed_count = 0

for image_path in source_dir.glob("*.jpg"):
    processed_count += 1

    sample_id = image_path.stem
    text_path = source_dir / f"{sample_id}.txt"

    # Read the image dimensions safely.
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except (OSError, ValueError) as exc:
        print(f"Cannot read {image_path.name}: {exc}")
        skipped_count += 1
        continue

    # Estimate the number of non-overlapping patches.
    seq_len = (width // PATCH_SIZE) * (height // PATCH_SIZE)

    # Ignore images within the threshold.
    if seq_len <= MAX_PATCHES:
        continue

    print(
        f"Large image: {sample_id}, "
        f"size={width}x{height}, patches={seq_len}"
    )

    # Require both files to exist.
    associated_files = [image_path, text_path]

    missing_files = [
        path.name for path in associated_files
        if not path.is_file()
    ]

    if missing_files:
        print(f"  Skipped: missing files {missing_files}")
        skipped_count += 1
        continue

    # Avoid partially moving an existing fileset.
    destinations = [
        target_dir / path.name for path in associated_files
    ]

    existing_destinations = [
        path.name for path in destinations
        if path.exists()
    ]

    if existing_destinations:
        print(
            f"  Skipped: destination already contains "
            f"{existing_destinations}"
        )
        skipped_count += 1
        continue

    # Move both files.
    try:
        for src, dst in zip(associated_files, destinations):
            shutil.move(str(src), str(dst))
            print(f"  Moved: {src.name}")

        moved_count += 1

    except OSError as exc:
        print(f"  Error moving {sample_id}: {exc}")
        print("  Check both directories for a partially moved fileset.")

print("\n--- Summary ---")
print(f"Images processed: {processed_count}")
print(f"Filesets moved: {moved_count}")
print(f"Filesets skipped: {skipped_count}")
print(f"Destination: {target_dir}")







