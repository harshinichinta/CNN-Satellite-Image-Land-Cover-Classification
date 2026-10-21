from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from backend.app.config import EUROSAT_CLASSES, label_for_class
from backend.app.schemas import ExplainResponse, ModelInfo, PredictionResponse, TopPrediction
from backend.app.services.image_processing import compute_gradcam, read_image_from_upload, validate_upload
from backend.app.services.predictor import model_manager

router = APIRouter(tags=["prediction"])
logger = logging.getLogger(__name__)


@router.get("/classes")
async def list_classes() -> Dict[str, Any]:
    payload = []
    for class_name in model_manager.classes or EUROSAT_CLASSES:
        payload.append({"class_name": class_name, "label": label_for_class(class_name)})
    return {"classes": payload}


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    try:
        validate_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not model_manager.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please make sure the trained model exists in the models directory.",
        )

    try:
        image = read_image_from_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        result = model_manager.predict(np.asarray(image))
    except Exception as exc:  # pragma: no cover - depends on model shape issues
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Image preprocessing failed: {exc}") from exc

    top_predictions = [
        TopPrediction(
            class_name=item["class_name"],
            label=item["label"],
            probability=item["probability"],
            rank=item["rank"],
        )
        for item in result["top_3_predictions"]
    ]

    return PredictionResponse(
        predicted_class=result["predicted_class"],
        label=result["label"],
        confidence=result["confidence"],
        top_3_predictions=top_predictions,
        probabilities={key: float(value) for key, value in result["probabilities"].items()},
        inference_time_ms=result["inference_time_ms"],
        probability_method=result["probability_method"],
        warning=result.get("warning"),
    )


@router.post("/explain", response_model=ExplainResponse)
async def explain(file: UploadFile = File(...)) -> ExplainResponse:
    prediction = await predict(file)

    try:
        image = read_image_from_upload(file)
        image_array = model_manager.prepare_input(np.asarray(image))[0]
        class_index = model_manager.classes.index(prediction.predicted_class)
        heatmap = compute_gradcam(model_manager.model, image_array, class_index)
        heatmap_base64 = None
        if heatmap is not None:
            from backend.app.services.image_processing import array_to_base64

            heatmap_base64 = array_to_base64(heatmap)
        explain_supported = True
    except Exception:
        logger.exception("Could not generate Grad-CAM explanation.")
        heatmap_base64 = None
        explain_supported = False

    return ExplainResponse(
        predicted_class=prediction.predicted_class,
        label=prediction.label,
        confidence=prediction.confidence,
        top_3_predictions=prediction.top_3_predictions,
        probabilities=prediction.probabilities,
        inference_time_ms=prediction.inference_time_ms,
        probability_method=prediction.probability_method,
        explainability_supported=explain_supported,
        heatmap_base64=heatmap_base64,
        warning=prediction.warning,
    )


@router.get("/model-info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    info = model_manager.model_info()
    return ModelInfo(
        architecture_name=info["architecture_name"],
        image_size=info["image_size"],
        supported_classes=info["supported_classes"],
        model_file=info["model_file"],
        evaluation_available=info["evaluation_available"],
        evaluation_accuracy=info["evaluation_accuracy"],
        evaluation_macro_f1=info["evaluation_macro_f1"],
        evaluation_test_samples=info["evaluation_test_samples"],
        evaluation_low_confidence_fraction=info["evaluation_low_confidence_fraction"],
        confidence_warning_threshold=info["confidence_warning_threshold"],
        training_samples=info["training_samples"],
        quick_training=info["quick_training"],
        calibration_temperature=info["calibration_temperature"],
        training_metadata_available=info["training_metadata_available"],
        notes=info["notes"],
    )
