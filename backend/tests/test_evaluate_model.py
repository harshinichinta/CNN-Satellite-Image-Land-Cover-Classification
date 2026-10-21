from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from scripts import evaluate_model


def test_load_dataset_decodes_jpeg_with_known_tensor_shapes(tmp_path: Path) -> None:
    annual_crop = tmp_path / "AnnualCrop"
    forest = tmp_path / "Forest"
    annual_crop.mkdir()
    forest.mkdir()
    annual_image = annual_crop / "annual.jpg"
    forest_image = forest / "forest.jpg"
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(annual_image)
    Image.new("RGB", (8, 8), color=(60, 40, 20)).save(forest_image)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "class_to_index": {"AnnualCrop": 0, "Forest": 1},
                "test_files": [str(annual_image), str(forest_image)],
            }
        ),
        encoding="utf-8",
    )

    images, labels = next(
        iter(evaluate_model.load_dataset(manifest_path, "test", image_size=4, batch_size=2))
    )

    assert images.shape == (2, 4, 4, 3)
    assert labels.numpy().tolist() == [0, 1]


class FakeModel:
    input_shape = (None, 4, 4, 3)
    output_shape = (None, 2)

    def predict(self, images, verbose: int = 0) -> np.ndarray:
        return np.tile(np.array([[0.8, 0.2]], dtype=np.float32), (len(images), 1))


def test_evaluate_model_runs_with_current_keras_predict_signature(
    tmp_path: Path, monkeypatch
) -> None:
    model_path = tmp_path / "model.keras"
    model_path.write_bytes(b"test model")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "classes": ["AnnualCrop", "Forest"],
                "class_to_index": {"AnnualCrop": 0, "Forest": 1},
                "test_files": ["AnnualCrop/a.jpg", "Forest/f.jpg", "AnnualCrop/b.jpg"],
            }
        ),
        encoding="utf-8",
    )
    dataset = [
        (tf.zeros((2, 4, 4, 3)), tf.constant([0, 1])),
        (tf.zeros((1, 4, 4, 3)), tf.constant([0])),
    ]
    requested_image_sizes = []

    def load_dataset_with_recorded_size(manifest, split_name, image_size, batch_size, **kwargs):
        requested_image_sizes.append(image_size)
        return dataset

    monkeypatch.setattr(evaluate_model, "load_dataset", load_dataset_with_recorded_size)
    monkeypatch.setattr(
        evaluate_model.tf.keras.models,
        "load_model",
        lambda *args, **kwargs: FakeModel(),
    )

    metrics = evaluate_model.evaluate_model(model_path, manifest_path, batch_size=2)

    assert metrics["num_test_samples"] == 3
    assert metrics["mean_inference_latency_ms"] >= 0
    assert requested_image_sizes == [4]
    assert len(metrics["model_sha256"]) == 64
    assert (tmp_path / "evaluation_metrics.json").is_file()
