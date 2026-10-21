from __future__ import annotations

import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from backend.app.config import (
    CONFIDENCE_WARNING_THRESHOLD,
    DEFAULT_EVAL_NAME,
    DEFAULT_MODEL_NAME,
    EUROSAT_CLASSES,
    MODELS_DIR,
    label_for_class,
)
from backend.app.services.image_processing import preprocess_image
from backend.app.services.probabilities import apply_temperature

try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or MODELS_DIR
        self.model = None
        self.model_path: Optional[Path] = None
        self.model_name: Optional[str] = None
        self.architecture_name = "Not loaded"
        self.image_size = [224, 224, 3]
        self.classes = list(EUROSAT_CLASSES)
        self.metadata = {}
        self.evaluation_metrics = {}

    def is_ready(self) -> bool:
        return self.model is not None

    def load_model(self, model_name: str = DEFAULT_MODEL_NAME) -> Any:
        if tf is None:
            raise RuntimeError("TensorFlow is not installed. Install backend requirements first.")

        model_file = self.model_dir / model_name
        if not model_file.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_file}")

        model = tf.keras.models.load_model(model_file, compile=False)
        metadata_file = model_file.with_name(f"{model_file.stem}_metadata.json")
        metadata = {}
        if metadata_file.exists():
            with metadata_file.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)

        classes = metadata.get("classes") or list(EUROSAT_CLASSES)
        if classes != EUROSAT_CLASSES:
            raise ValueError(
                "Model class order does not match the supported EuroSAT class order."
            )
        output_count = model.output_shape[-1]
        if output_count != len(classes):
            raise ValueError(
                f"Model outputs {output_count} classes, but its metadata defines {len(classes)}."
            )
        preprocessing = metadata.get("preprocessing")
        if preprocessing not in (None, "mobilenet_v2", "rescale_0_1"):
            raise ValueError(f"Unsupported model preprocessing metadata: {preprocessing}")

        evaluation_metrics = {}
        evaluation_file = self.model_dir / DEFAULT_EVAL_NAME
        if evaluation_file.exists():
            with evaluation_file.open("r", encoding="utf-8") as handle:
                candidate_metrics = json.load(handle)
            model_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
            if candidate_metrics.get("model_sha256") == model_hash:
                evaluation_metrics = candidate_metrics
            else:
                logger.warning("Ignoring stale evaluation metrics for model %s.", model_file.name)

        self.model_path = model_file
        self.model_name = model_file.name
        self.model = model
        self.architecture_name = model.name or model.__class__.__name__
        self.image_size = list(model.input_shape[1:])
        self.classes = classes
        self.metadata = metadata
        self.evaluation_metrics = evaluation_metrics

        return self.model

    def prepare_input(self, image_rgb: np.ndarray) -> np.ndarray:
        image_height, image_width = self.image_size[:2]
        target_size = (
            image_height if image_height is not None else 224,
            image_width if image_width is not None else 224,
        )
        prepared = preprocess_image(image_rgb, target_size=target_size)
        if self.metadata.get("preprocessing") == "mobilenet_v2":
            prepared = prepared * 2.0 - 1.0
        return prepared

    def predict(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please train or load a Keras model first.")

        if tf is None:
            raise RuntimeError("TensorFlow is required for inference.")

        start = time.perf_counter()
        prepared = self.prepare_input(image_rgb)
        raw_scores = self.model.predict(prepared, verbose=0)
        probabilities = apply_temperature(
            raw_scores,
            float(self.metadata.get("calibration_temperature", 1.0)),
        )[0]
        if not np.isfinite(probabilities).all() or not np.isclose(
            np.sum(probabilities), 1.0, atol=1e-3
        ):
            raise ValueError("The model did not return a valid probability distribution.")
        class_scores = list(probabilities)
        top_indices = np.argsort(class_scores)[::-1][:3]

        probabilities_map = {
            class_name: float(class_scores[idx])
            for idx, class_name in enumerate(self.classes)
        }

        top_predictions = []
        for rank, idx in enumerate(top_indices, start=1):
            class_name = self.classes[int(idx)]
            top_predictions.append(
                {
                    "class_name": class_name,
                    "label": label_for_class(class_name),
                    "probability": float(class_scores[int(idx)]),
                    "rank": rank,
                }
            )

        predicted_idx = int(np.argmax(class_scores))
        predicted_class = self.classes[predicted_idx]
        confidence = float(class_scores[predicted_idx])
        inference_time_ms = (time.perf_counter() - start) * 1000

        warning = None
        if confidence < CONFIDENCE_WARNING_THRESHOLD:
            warning = (
                "Low model confidence. The top class probability is "
                f"{confidence:.1%}; verify this prediction, especially for imagery outside EuroSAT."
            )
            accuracy = self.evaluation_metrics.get("accuracy")
            if accuracy is not None:
                warning += f" Held-out test accuracy is {accuracy:.1%}."

        return {
            "predicted_class": predicted_class,
            "label": label_for_class(predicted_class),
            "confidence": confidence,
            "top_3_predictions": top_predictions,
            "probabilities": probabilities_map,
            "inference_time_ms": round(inference_time_ms, 2),
            "warning": warning,
            "probability_method": (
                "validation_temperature_scaled"
                if self.metadata.get("calibration_temperature") not in (None, 1, 1.0)
                else "model_softmax"
            ),
        }

    def model_info(self) -> Dict[str, Any]:
        return {
            "architecture_name": self.architecture_name,
            "image_size": self.image_size,
            "supported_classes": self.classes,
            "model_file": self.model_name,
            "evaluation_available": bool(self.evaluation_metrics),
            "evaluation_accuracy": self.evaluation_metrics.get("accuracy"),
            "evaluation_macro_f1": self.evaluation_metrics.get("macro_f1"),
            "evaluation_test_samples": self.evaluation_metrics.get("num_test_samples"),
            "evaluation_low_confidence_fraction": self.evaluation_metrics.get(
                "low_confidence_fraction"
            ),
            "confidence_warning_threshold": CONFIDENCE_WARNING_THRESHOLD,
            "training_samples": self.metadata.get("train_samples"),
            "quick_training": self.metadata.get("quick_training", False),
            "calibration_temperature": self.metadata.get("calibration_temperature", 1.0),
            "training_metadata_available": bool(self.metadata),
            "notes": (
                "No trained model is loaded. Train on EuroSAT before predicting; "
                "the smoke-test model is intentionally not served."
                if not self.is_ready()
                else "Confidence is the model's softmax score, not a guarantee of accuracy. "
                "Predictions are intended for RGB satellite tiles similar to the training data."
            ),
        }


model_manager = ModelManager()
