import requests
import sys
from pathlib import Path


def download_file(url, download_path):
	download_path = "./data/"+download_path
	response = requests.get(url, stream=True)
	response.raise_for_status()

	path = Path(download_path)
	path.parent.mkdir(parents=True, exist_ok=True)

	with open(path, "wb") as f:
		for chunk in response.iter_content(chunk_size=1024 * 1024):
			if chunk:
				f.write(chunk)

	print(f"Downloaded to: {path}")


if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage:")
		print("  python download.py <url> <download_path>")
		sys.exit(1)

	url = sys.argv[1]
	download_path = sys.argv[2]

	download_file(url, download_path)
