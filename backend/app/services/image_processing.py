from __future__ import annotations

import base64
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image
from fastapi import UploadFile

from backend.app.config import ALLOWED_EXTENSIONS, DEFAULT_IMAGE_SIZE, MAX_UPLOAD_SIZE_BYTES


def validate_upload(file: UploadFile, max_size_bytes: int = MAX_UPLOAD_SIZE_BYTES) -> None:
    if not file.filename:
        raise ValueError("Upload is missing a filename.")

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file.filename}. Use JPG, JPEG, PNG, BMP, TIFF, or TIF.")

    contents = file.file.read()
    if len(contents) == 0:
        raise ValueError("Uploaded file is empty.")
    if len(contents) > max_size_bytes:
        raise ValueError(f"Uploaded image exceeds {max_size_bytes / (1024 * 1024):.1f} MB.")
    file.file.seek(0)


def read_image_from_upload(file: UploadFile) -> Image.Image:
    data = file.file.read()
    file.file.seek(0)
    try:
        image = Image.open(BytesIO(data))
    except Exception as exc:  # pragma: no cover - validated by PIL errors
        raise ValueError("Could not read the uploaded image file.") from exc

    image = image.convert("RGB")
    if image is None:
        raise ValueError("The uploaded file does not contain a valid image.")
    return image


def preprocess_image(image: Image.Image | np.ndarray, target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    if isinstance(image, np.ndarray):
        image_array = image.astype(np.uint8)
    else:
        image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)

    if image_array.ndim != 3 or image_array.shape[-1] != 3:
        raise ValueError("Expected an RGB image with exactly three color channels.")

    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - dependencies are installed by requirements
        raise RuntimeError("TensorFlow is required to resize images for model inference.") from exc

    resized = tf.image.resize(
        image_array,
        target_size,
        method=tf.image.ResizeMethod.BILINEAR,
    )
    array = resized.numpy().astype(np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def compute_gradcam(model, image_array: np.ndarray, class_index: int, target_layer_name: str | None = None) -> np.ndarray:
    """Approximate Grad-CAM heatmap for a standard Keras CNN model."""
    if model is None:
        raise ValueError("A trained CNN model is required to compute Grad-CAM.")

    try:
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for Grad-CAM support.") from exc

    target_layer = None
    nested_model = None
    if target_layer_name:
        target_layer = model.get_layer(target_layer_name)
    else:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer = layer
                break
            if isinstance(layer, tf.keras.Model):
                nested_layer = next(
                    (
                        child
                        for child in reversed(layer.layers)
                        if isinstance(child, tf.keras.layers.Conv2D)
                    ),
                    None,
                )
                if nested_layer is not None:
                    target_layer = nested_layer
                    nested_model = layer
                    break

    if target_layer is None:
        raise ValueError("Grad-CAM is not supported for this model architecture.")

    if nested_model is None:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )
    else:
        feature_model = tf.keras.models.Model(
            inputs=nested_model.input,
            outputs=[target_layer.output, nested_model.output],
        )
        activations, features = feature_model(model.inputs[0])
        nested_index = model.layers.index(nested_model)
        predictions = features
        for layer in model.layers[nested_index + 1 :]:
            predictions = layer(predictions)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[activations, predictions],
        )

    image_batch = tf.convert_to_tensor(np.expand_dims(image_array.astype(np.float32), axis=0), dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        score = predictions[:, class_index]

    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise ValueError("Grad-CAM could not trace gradients to the selected convolution layer.")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs[0], axis=-1)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap.numpy()
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    return heatmap


def array_to_base64(image_array: np.ndarray) -> str:
    image = image_array
    if image_array.ndim == 2:
        image = np.stack([image_array] * 3, axis=-1)
    image = (image * 255).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image).convert("RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


try:
    import tensorflow as tf

    def tf_gradients(loss, tensor):
        return tf.gradients(loss, tensor)

except Exception:  # pragma: no cover - only used when TensorFlow is absent
    def tf_gradients(loss, tensor):
        raise RuntimeError("TensorFlow is required for Grad-CAM support.")
