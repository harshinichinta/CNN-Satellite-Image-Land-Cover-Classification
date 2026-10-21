from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Keep the class list aligned with the EuroSAT RGB dataset used by default.
# These are the actual dataset class names, not invented labels.
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

CLASS_DISPLAY_NAMES = {
    "AnnualCrop": "Annual crop",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous vegetation",
    "Highway": "Highway",
    "Industrial": "Industrial buildings",
    "Pasture": "Pasture",
    "PermanentCrop": "Permanent crop",
    "Residential": "Residential buildings",
    "River": "River",
    "SeaLake": "Sea or lake",
}

DEFAULT_MODEL_NAME = "best_model.keras"
DEFAULT_METADATA_NAME = "training_metadata.json"
DEFAULT_EVAL_NAME = "evaluation_metrics.json"
DEFAULT_IMAGE_SIZE = (224, 224)
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CONFIDENCE_WARNING_THRESHOLD = 0.60


def model_path(model_name: str = DEFAULT_MODEL_NAME) -> Path:
    return MODELS_DIR / model_name


def metadata_path(model_name: str = DEFAULT_MODEL_NAME) -> Path:
    return MODELS_DIR / f"{Path(model_name).stem}_metadata.json"


def label_for_class(class_name: str) -> str:
    return CLASS_DISPLAY_NAMES.get(class_name, class_name)
