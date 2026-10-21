from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    backend: str


class TopPrediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    class_name: str
    label: str
    probability: float
    rank: int


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predicted_class: str
    label: str
    confidence: float
    top_3_predictions: List[TopPrediction]
    probabilities: Dict[str, float]
    inference_time_ms: float
    probability_method: str = "model_softmax"
    warning: Optional[str] = None


class ExplainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predicted_class: str
    label: str
    confidence: float
    top_3_predictions: List[TopPrediction]
    probabilities: Dict[str, float]
    inference_time_ms: float
    probability_method: str = "model_softmax"
    explainability_supported: bool
    heatmap_base64: Optional[str] = None
    warning: Optional[str] = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    architecture_name: str
    image_size: List[int]
    supported_classes: List[str]
    model_file: Optional[str] = None
    evaluation_available: bool = False
    evaluation_accuracy: Optional[float] = None
    evaluation_macro_f1: Optional[float] = None
    evaluation_test_samples: Optional[int] = None
    evaluation_low_confidence_fraction: Optional[float] = None
    confidence_warning_threshold: float = 0.6
    training_samples: Optional[int] = None
    quick_training: bool = False
    calibration_temperature: float = 1.0
    training_metadata_available: bool = False
    notes: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    detail: str
