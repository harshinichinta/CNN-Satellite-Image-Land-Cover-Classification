from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import Headers
from fastapi import UploadFile

from backend.app.config import CONFIDENCE_WARNING_THRESHOLD
from backend.app.main import app
from backend.app.services.image_processing import array_to_base64, compute_gradcam, validate_upload
from backend.app.services.probabilities import apply_temperature, fit_temperature
from backend.app.services.predictor import ModelManager, model_manager

client = TestClient(app)


def build_dummy_model(model_path: Path) -> None:
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax'),
        ]
    )
    model.save(model_path)


def test_health_endpoint() -> None:
    response = client.get('/health')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'
    assert 'backend' in payload


def test_classes_endpoint() -> None:
    response = client.get('/classes')
    assert response.status_code == 200
    payload = response.json()
    assert 'classes' in payload
    assert len(payload['classes']) > 0


def test_missing_model_is_reported_gracefully() -> None:
    response = client.post('/predict', files={'file': ('valid.png', b'not-a-valid-image', 'image/png')})
    assert response.status_code == 400 or response.status_code == 503


def test_invalid_upload_is_rejected() -> None:
    response = client.post('/predict', files={'file': ('test.txt', b'hello world', 'text/plain')})
    assert response.status_code == 400


def test_webp_upload_is_accepted_by_file_validation() -> None:
    upload = UploadFile(
        filename='satellite.webp',
        file=io.BytesIO(b'webp-image-data'),
        headers=Headers({'content-type': 'image/webp'}),
    )

    validate_upload(upload)


def test_transfer_preprocessing_matches_mobilenet_v2_range() -> None:
    manager = ModelManager()
    manager.metadata = {'preprocessing': 'mobilenet_v2'}
    image = np.full((8, 8, 3), (12, 128, 240), dtype=np.uint8)

    prepared = manager.prepare_input(image)

    expected = np.array((12, 128, 240), dtype=np.float32) / 127.5 - 1.0
    assert prepared.shape == (1, 224, 224, 3)
    assert np.allclose(prepared[0, 0, 0], expected, atol=1e-6)


def test_temperature_scaling_preserves_class_order_and_calibrates_scores() -> None:
    scores = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]], dtype=np.float32)
    labels = np.array([0, 1, 1])

    temperature = fit_temperature(scores, labels)
    calibrated = apply_temperature(scores, temperature)

    assert temperature > 0
    assert np.array_equal(np.argmax(scores, axis=1), np.argmax(calibrated, axis=1))
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert not np.allclose(calibrated, scores)


def test_gradcam_supports_nested_convolutional_backbone() -> None:
    import tensorflow as tf

    backbone_input = tf.keras.layers.Input(shape=(32, 32, 3))
    convolution_output = tf.keras.layers.Conv2D(
        4,
        (3, 3),
        padding="same",
        activation="relu",
        name="feature_conv",
    )(backbone_input)
    backbone = tf.keras.Model(backbone_input, convolution_output, name="backbone")

    image_input = tf.keras.layers.Input(shape=(32, 32, 3))
    features = backbone(image_input)
    pooled_features = tf.keras.layers.GlobalAveragePooling2D()(features)
    predictions = tf.keras.layers.Dense(2, activation="softmax")(pooled_features)
    model = tf.keras.Model(image_input, predictions)

    heatmap = compute_gradcam(
        model,
        np.ones((32, 32, 3), dtype=np.float32),
        class_index=0,
    )

    assert heatmap.shape == (32, 32)
    assert np.isfinite(heatmap).all()


def test_heatmap_encoding_accepts_grayscale_gradcam_output() -> None:
    heatmap = np.linspace(0, 1, 49, dtype=np.float32).reshape((7, 7))

    encoded = array_to_base64(heatmap)

    with Image.open(io.BytesIO(base64.b64decode(encoded))) as result:
        assert result.size == (7, 7)
        assert result.mode == "RGB"


def test_valid_prediction_smoke() -> None:
    model_path = Path(__file__).resolve().parents[2] / 'models' / 'test_smoke_model.keras'
    model_path.parent.mkdir(exist_ok=True, parents=True)
    build_dummy_model(model_path)
    model_manager.load_model('test_smoke_model.keras')

    image = Image.new('RGB', (224, 224), color=(12, 34, 56))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    response = client.post('/predict', files={'file': ('smoke.png', buffer.getvalue(), 'image/png')})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert 'predicted_class' in payload
    assert 'confidence' in payload
    assert 'top_3_predictions' in payload
    assert len(payload['top_3_predictions']) == 3
    assert np.isclose(sum(payload['probabilities'].values()), 1.0, atol=1e-3)
    assert payload['confidence'] == payload['top_3_predictions'][0]['probability']
    assert (payload['warning'] is not None) == (
        payload['confidence'] < CONFIDENCE_WARNING_THRESHOLD
    )
