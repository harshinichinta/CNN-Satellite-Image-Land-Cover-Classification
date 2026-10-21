from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from backend.app.config import BASE_DIR, DEFAULT_MODEL_NAME
from backend.app.services.predictor import ModelManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a single satellite image using the trained model.")
    parser.add_argument("--image", required=True, help="Path to a satellite image")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Saved model archive")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_manager = ModelManager(model_dir=BASE_DIR / "models")
    model_manager.load_model(args.model_name)

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    result = model_manager.predict(np.asarray(image))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
