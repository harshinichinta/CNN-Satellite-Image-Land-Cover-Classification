from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from backend.app.config import CONFIDENCE_WARNING_THRESHOLD
from backend.app.services.probabilities import apply_temperature


def decode_image(path: str, label: int, image_size: int, use_preprocess: bool = False):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(
        image,
        [image_size, image_size],
        method=tf.image.ResizeMethod.BILINEAR,
    )
    image = tf.cast(image, tf.float32)
    if use_preprocess:
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    else:
        image = image / 255.0
    return image, label


def load_dataset(manifest_path: Path, split_name: str, image_size: int, batch_size: int, use_preprocess: bool = False):
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    class_to_index = manifest["class_to_index"]
    file_paths = manifest[f"{split_name}_files"]
    labels = [class_to_index[Path(path).parent.name] for path in file_paths]
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    dataset = dataset.map(
        lambda path, label: decode_image(
            path,
            label,
            image_size,
            use_preprocess,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def evaluate_model(
    model_path: Path,
    manifest_path: Path,
    image_size: int | None = None,
    batch_size: int = 32,
):
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    class_names = manifest["classes"]
    if manifest["class_to_index"] != {
        class_name: index for index, class_name in enumerate(class_names)
    }:
        raise ValueError("Manifest classes and class_to_index have different orders.")
    metadata_path = model_path.with_name(f"{model_path.stem}_metadata.json")
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    model_classes = metadata.get("classes")
    if model_classes is not None and model_classes != class_names:
        raise ValueError("Model metadata class order does not match the dataset manifest.")
    preprocessing = metadata.get("preprocessing")
    if preprocessing not in (None, "mobilenet_v2", "rescale_0_1"):
        raise ValueError(f"Unsupported model preprocessing metadata: {preprocessing}")
    use_preprocess = (
        preprocessing == "mobilenet_v2"
        if preprocessing
        else "mobilenet" in model_path.name.lower()
    )
    model = tf.keras.models.load_model(model_path, compile=False)
    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"Model outputs {model.output_shape[-1]} classes, but manifest defines {len(class_names)}."
        )
    model_image_size = image_size or model.input_shape[1]
    if model_image_size is None:
        raise ValueError("Model input image size must be fixed or provided with --image-size.")
    test_ds = load_dataset(
        manifest_path,
        "test",
        model_image_size,
        batch_size,
        use_preprocess=use_preprocess,
    )
    calibration_temperature = float(metadata.get("calibration_temperature", 1.0))
    if calibration_temperature <= 0:
        raise ValueError("Model calibration temperature must be positive.")

    y_true = []
    y_pred = []
    y_probabilities = []
    raw_confidences = []
    inference_latencies = []
    for images, labels in test_ds:
        start = time.perf_counter()
        predictions = model.predict(images, verbose=0)
        inference_latencies.append((time.perf_counter() - start) * 1000 / len(images))
        raw_confidences.extend(np.max(predictions, axis=1).tolist())
        predictions = apply_temperature(predictions, calibration_temperature)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(predictions, axis=1).tolist())
        y_probabilities.extend(predictions.tolist())

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    probability_array = np.asarray(y_probabilities)
    confidences = np.max(probability_array, axis=1)
    correct_predictions = (y_true_array == y_pred_array).astype(np.float32)
    calibration_error = 0.0
    calibration_bins = np.linspace(0.0, 1.0, 11)
    for bin_index in range(len(calibration_bins) - 1):
        in_bin = (confidences >= calibration_bins[bin_index]) & (
            confidences < calibration_bins[bin_index + 1]
            if bin_index < len(calibration_bins) - 2
            else confidences <= calibration_bins[bin_index + 1]
        )
        if np.any(in_bin):
            calibration_error += float(np.mean(in_bin)) * abs(
                float(np.mean(correct_predictions[in_bin]))
                - float(np.mean(confidences[in_bin]))
            )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    per_class = {}
    for idx, class_name in enumerate(class_names):
        class_y_true = y_true_array
        class_y_pred = y_pred_array
        mask = (class_y_true == idx)
        if not np.any(mask):
            per_class[class_name] = {"support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue
        class_precision = precision_score(class_y_true, class_y_pred, labels=[idx], average="macro", zero_division=0)
        class_recall = recall_score(class_y_true, class_y_pred, labels=[idx], average="macro", zero_division=0)
        class_f1 = f1_score(class_y_true, class_y_pred, labels=[idx], average="macro", zero_division=0)
        per_class[class_name] = {
            "support": int(mask.sum()),
            "precision": float(class_precision),
            "recall": float(class_recall),
            "f1": float(class_f1),
        }

    latency_ms = float(np.mean(inference_latencies))

    metrics = {
        "model_path": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "mean_confidence": float(np.mean(confidences)),
        "mean_raw_confidence": float(np.mean(raw_confidences)),
        "expected_calibration_error": float(calibration_error),
        "low_confidence_fraction": float(
            np.mean(confidences < CONFIDENCE_WARNING_THRESHOLD)
        ),
        "confidence_warning_threshold": CONFIDENCE_WARNING_THRESHOLD,
        "calibration_temperature": calibration_temperature,
        "sample_predictions": {
            class_name: [
                {
                    "image": Path(manifest["test_files"][index]).name,
                    "predicted_class": class_names[int(y_pred_array[index])],
                    "confidence": float(confidences[index]),
                    "correct": bool(y_true_array[index] == y_pred_array[index]),
                }
                for index in [
                    i
                    for i, file_path in enumerate(manifest["test_files"])
                    if Path(file_path).parent.name == class_name
                ][:5]
            ]
            for class_name in class_names
        },
        "mean_inference_latency_ms": round(latency_ms, 4),
        "model_size_bytes": os.path.getsize(model_path),
        "num_test_samples": len(y_true),
    }

    output_dir = model_path.parent
    with (output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_title("Confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained EuroSAT model on the untouched test set.")
    parser.add_argument("--model-path", default="models/best_model.keras", help="Path to the saved Keras model")
    parser.add_argument("--manifest", default="data/processed/dataset_manifest.json", help="Prepared dataset manifest")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override model input size; by default the saved model's input size is used.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    model_path = (base_dir / args.model_path).resolve()
    manifest_path = (base_dir / args.manifest).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    metrics = evaluate_model(model_path, manifest_path, image_size=args.image_size, batch_size=args.batch_size)
    print(json.dumps({
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "mean_confidence": metrics["mean_confidence"],
        "expected_calibration_error": metrics["expected_calibration_error"],
        "low_confidence_fraction": metrics["low_confidence_fraction"],
        "sample_predictions": metrics["sample_predictions"],
        "mean_inference_latency_ms": metrics["mean_inference_latency_ms"],
        "model_size_bytes": metrics["model_size_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
