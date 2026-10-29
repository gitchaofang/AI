import json
import shutil
from pathlib import Path

PATCH_SIZE = 16  # Replace with your actual patch size
MAX_PATCHES = 2032

source_dir = Path("/content/cache/training")
target_dir = Path("/content/cache/training/large_images")

# Create destination directory if it doesn't exist.
target_dir.mkdir(parents=True, exist_ok=True)

moved_count = 0
total = 0

for json_file in source_dir.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    width = int(data["width"])
    height = int(data["height"])

    patch_seq_len = (
        (width // PATCH_SIZE) * (height // PATCH_SIZE)
    )

    total += 1

    if patch_seq_len > MAX_PATCHES:
        file_id = json_file.stem

        associated_files = [
            json_file,
            source_dir / f"{file_id}.jpg",
            source_dir / f"{file_id}.txt",
        ]

        print(
            f"Moving {file_id}: "
            f"{width}x{height}, patches={patch_seq_len}"
        )

        for file in associated_files:
            if file.exists():
                destination = target_dir / file.name

                if destination.exists():
                    print(f"  Skipped (destination exists): {file.name}")
                    continue

                shutil.move(str(file), str(destination))
                print(f"  Moved: {file.name}")

        moved_count += 1

print(f"\nTotal JSON files checked: {total}")
print(f"Filesets processed: {moved_count}")
print(f"Destination: {target_dir}")
