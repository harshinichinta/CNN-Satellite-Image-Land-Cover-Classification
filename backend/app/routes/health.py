from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas import HealthResponse
from backend.app.services.predictor import model_manager

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_manager.is_ready(),
        model_name=model_manager.model_name,
        backend="Satellite land-cover classification backend",
    )
