from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_DATASET_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Dataset already exists at {destination}. Skipping download.")
        return destination
    print(f"Downloading dataset from {url} to {destination}...")
    urlretrieve(url, destination)
    return destination


def extract_archive(archive_path: Path, target_dir: Path) -> Path:
    dataset_dirs = [
        target_dir / "2750",
        target_dir / "EuroSAT",
        target_dir / "EuroSAT" / "2750",
        target_dir / "EuroSAT_RGB",
    ]
    if any((dataset_dir / "AnnualCrop").is_dir() for dataset_dir in dataset_dirs):
        print(f"Extraction target {target_dir} already exists. Leaving it in place.")
        return target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(target_dir)
    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the EuroSAT RGB satellite dataset.")
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL, help="Dataset archive URL")
    parser.add_argument("--output", default="data/EuroSAT.zip", help="Where to save the downloaded archive")
    parser.add_argument("--extract-to", default="data", help="Directory to extract the archive into")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    output_path = (base_dir / args.output).resolve()
    extract_dir = (base_dir / args.extract_to).resolve()

    try:
        archive_path = download_file(args.dataset_url, output_path)
        extract_archive(archive_path, extract_dir)
        print(f"Dataset ready. Check the files under: {extract_dir}")
    except Exception as exc:
        print(f"Automatic download failed: {exc}", file=sys.stderr)
        print("\nManual download instructions:", file=sys.stderr)
        print("1. Visit https://github.com/phelber/eurosat", file=sys.stderr)
        print("2. Download the EuroSAT RGB archive from the official project page.", file=sys.stderr)
        print(f"3. Save it as {output_path} and re-run this script.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
