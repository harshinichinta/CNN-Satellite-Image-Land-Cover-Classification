from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image
from sklearn.model_selection import train_test_split


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_dataset_root(dataset_root: Path, candidates: Iterable[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if dataset_root.exists():
        return dataset_root
    raise FileNotFoundError(
        f"Dataset directory not found. Looked for: {', '.join(str(path) for path in [dataset_root, *candidates])}"
    )


def collect_image_paths(dataset_root: Path):
    records = []
    corrupt_images = []
    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.glob("*.*")):
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                with Image.open(image_path) as img:
                    img.verify()
                with Image.open(image_path) as img:
                    width, height = img.size
                records.append({
                    "image_path": str(image_path),
                    "label": class_dir.name,
                    "width": width,
                    "height": height,
                })
            except Exception:
                corrupt_images.append(str(image_path))
                print(f"Skipping corrupted image: {image_path}")
    return records, corrupt_images


def summarize_dimensions(records):
    widths = [record["width"] for record in records]
    heights = [record["height"] for record in records]
    return {
        "min_width": min(widths) if widths else 0,
        "max_width": max(widths) if widths else 0,
        "min_height": min(heights) if heights else 0,
        "max_height": max(heights) if heights else 0,
        "unique_sizes": sorted({(record["width"], record["height"]) for record in records}),
    }


def split_records(records, test_size=0.2, val_size=0.2, random_state=42):
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be positive and sum to less than 1.")

    image_names = [record["image_path"] for record in records]
    labels = [record["label"] for record in records]
    holdout_size = test_size + val_size

    train_files, holdout_files, _, holdout_labels = train_test_split(
        image_names,
        labels,
        test_size=holdout_size,
        stratify=labels,
        random_state=random_state,
    )
    val_files, test_files = train_test_split(
        holdout_files,
        test_size=test_size / holdout_size,
        stratify=holdout_labels,
        random_state=random_state,
    )

    return train_files, val_files, test_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and split the EuroSAT dataset.")
    parser.add_argument("--dataset-dir", default="data/2750", help="Dataset directory with class folders")
    parser.add_argument("--output-dir", default="data/processed", help="Directory to save manifest files")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    dataset_root = (base_dir / args.dataset_dir).resolve()
    dataset_root = find_dataset_root(
        dataset_root,
        [
            base_dir / "data" / "2750",
            base_dir / "data" / "EuroSAT",
            base_dir / "data" / "EuroSAT" / "2750",
            base_dir / "data" / "EuroSAT_RGB",
        ],
    )
    output_dir = (base_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, corrupt_images = collect_image_paths(dataset_root)
    train_files, val_files, test_files = split_records(records, random_state=args.random_seed)

    distribution = dict(sorted(Counter(record["label"] for record in records).items()))
    class_names = list(distribution.keys())
    class_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}

    for record in records:
        record["label_index"] = class_to_index[record["label"]]

    manifest = {
        "dataset_dir": str(dataset_root),
        "classes": class_names,
        "class_to_index": class_to_index,
        "random_seed": args.random_seed,
        "class_distribution": distribution,
        "corrupt_images": corrupt_images,
        "image_summary": summarize_dimensions(records),
        "train_count": len(train_files),
        "val_count": len(val_files),
        "test_count": len(test_files),
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
    }

    manifest_path = output_dir / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Prepared dataset manifest at {manifest_path}")
    print(f"Classes: {class_names}")
    print(f"Class distribution: {distribution}")
    print(f"Corrupted images skipped: {len(corrupt_images)}")
    print(f"Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")


if __name__ == "__main__":
    main()
