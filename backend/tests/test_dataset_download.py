from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path

from scripts.download_dataset import extract_archive
from scripts.prepare_dataset import find_dataset_root, split_records


def test_extract_archive_when_target_directory_already_exists(tmp_path: Path) -> None:
    target_dir = tmp_path / "data"
    target_dir.mkdir()
    (target_dir / "README.md").write_text("Existing project file", encoding="utf-8")
    archive_path = tmp_path / "EuroSAT.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("2750/AnnualCrop/sample.jpg", b"image data")

    extract_archive(archive_path, target_dir)

    assert (target_dir / "README.md").is_file()
    assert (target_dir / "2750" / "AnnualCrop" / "sample.jpg").read_bytes() == b"image data"


def test_extract_archive_skips_when_dataset_is_already_present(tmp_path: Path) -> None:
    target_dir = tmp_path / "data"
    dataset_class_dir = target_dir / "2750" / "AnnualCrop"
    dataset_class_dir.mkdir(parents=True)
    archive_path = tmp_path / "EuroSAT.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("2750/AnnualCrop/sample.jpg", b"new image")

    extract_archive(archive_path, target_dir)

    assert not (dataset_class_dir / "sample.jpg").exists()


def test_extract_archive_recognizes_eurosat_rgb_layout(tmp_path: Path) -> None:
    target_dir = tmp_path / "data"
    (target_dir / "EuroSAT_RGB" / "AnnualCrop").mkdir(parents=True)
    archive_path = tmp_path / "EuroSAT.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("EuroSAT_RGB/AnnualCrop/sample.jpg", b"new image")

    extract_archive(archive_path, target_dir)

    assert not (target_dir / "EuroSAT_RGB" / "AnnualCrop" / "sample.jpg").exists()


def test_find_dataset_root_finds_eurosat_rgb_inside_existing_data_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    dataset_root = data_dir / "EuroSAT_RGB"
    (dataset_root / "AnnualCrop").mkdir(parents=True)
    (data_dir / "README.md").write_text("Existing project file", encoding="utf-8")

    assert find_dataset_root(
        data_dir / "2750",
        [
            data_dir / "2750",
            data_dir / "EuroSAT",
            data_dir / "EuroSAT" / "2750",
            data_dir / "EuroSAT_RGB",
        ],
    ) == dataset_root


def test_split_records_keeps_stratification_labels_aligned() -> None:
    records = [
        {"image_path": f"{class_name}/{index}.jpg", "label": class_name}
        for class_name, count in (("AnnualCrop", 20), ("Forest", 30), ("River", 50))
        for index in range(count)
    ]

    train_files, val_files, test_files = split_records(records)
    labels_by_path = {record["image_path"]: record["label"] for record in records}

    assert Counter(labels_by_path[path] for path in train_files) == {
        "AnnualCrop": 12,
        "Forest": 18,
        "River": 30,
    }
    assert Counter(labels_by_path[path] for path in val_files) == {
        "AnnualCrop": 4,
        "Forest": 6,
        "River": 10,
    }
    assert Counter(labels_by_path[path] for path in test_files) == {
        "AnnualCrop": 4,
        "Forest": 6,
        "River": 10,
    }
