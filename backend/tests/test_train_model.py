from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf
from PIL import Image

from scripts.train_model import fit_model, load_split_dataset


def test_training_dataset_has_shapes_keras_can_fit(tmp_path: Path) -> None:
    annual_crop = tmp_path / "AnnualCrop"
    forest = tmp_path / "Forest"
    annual_crop.mkdir()
    forest.mkdir()
    train_image = annual_crop / "train.jpg"
    second_train_image = annual_crop / "train_2.jpg"
    val_image = forest / "val.jpg"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(train_image)
    Image.new("RGB", (8, 8), color=(20, 30, 40)).save(second_train_image)
    Image.new("RGB", (8, 8), color=(30, 20, 10)).save(val_image)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "class_to_index": {"AnnualCrop": 0, "Forest": 1},
                "train_files": [str(train_image), str(second_train_image)],
                "val_files": [str(val_image)],
            }
        ),
        encoding="utf-8",
    )

    train_ds = load_split_dataset(manifest_path, "train", image_size=4, batch_size=1)
    val_ds = load_split_dataset(manifest_path, "val", image_size=4, batch_size=1)
    assert train_ds.element_spec[0].shape == (None, 4, 4, 3)
    assert train_ds.element_spec[1].shape == (None,)

    quick_ds = load_split_dataset(
        manifest_path,
        "train",
        image_size=4,
        batch_size=1,
        max_samples_per_class=1,
    )
    assert sum(int(labels.shape[0]) for _, labels in quick_ds) == 1

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4, 4, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    history = fit_model(model, train_ds, val_ds, epochs=1, callbacks=[])

    assert len(history.history["loss"]) == 1
