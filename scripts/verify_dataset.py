from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.prepare_dataset import collect_image_paths, find_dataset_root, summarize_dimensions


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify EuroSAT dataset integrity, counts, and class distribution.")
    parser.add_argument("--dataset-dir", default="data/2750", help="EuroSAT dataset directory")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for reports and figures")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    dataset_root = find_dataset_root(
        (base_dir / args.dataset_dir).resolve(),
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
    by_label = {}
    for record in records:
        by_label[record["label"]] = by_label.get(record["label"], 0) + 1

    summary = {
        "dataset_dir": str(dataset_root),
        "class_names": sorted(by_label),
        "class_distribution": {key: by_label[key] for key in sorted(by_label)},
        "image_count": len(records),
        "corrupted_images": corrupt_images,
        "image_dimensions": summarize_dimensions(records),
    }

    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    class_names = list(summary["class_distribution"]) 
    counts = list(summary["class_distribution"].values())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(class_names, counts)
    ax.set_title("EuroSAT class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image count")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=200)
    plt.close(fig)

    sample_rows = min(2, len(class_names))
    sample_cols = max(1, min(5, len(class_names)))
    fig, axes = plt.subplots(sample_rows, sample_cols, figsize=(sample_cols * 3, sample_rows * 3))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax_idx, class_name in enumerate(class_names[: sample_rows * sample_cols]):
        paths = [record["image_path"] for record in records if record["label"] == class_name]
        if not paths:
            continue
        image = plt.imread(paths[0])
        axes[ax_idx].imshow(image)
        axes[ax_idx].set_title(class_name)
        axes[ax_idx].axis("off")

    for idx in range(len(class_names[: sample_rows * sample_cols]), len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "sample_images.png", dpi=200)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Summary saved to {output_dir / 'dataset_summary.json'}")
    print(f"Class distribution plot saved to {output_dir / 'class_distribution.png'}")
    print(f"Sample image grid saved to {output_dir / 'sample_images.png'}")


if __name__ == "__main__":
    main()