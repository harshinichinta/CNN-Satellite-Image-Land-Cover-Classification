from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import DEFAULT_MODEL_NAME, MODELS_DIR
from backend.app.routes.health import router as health_router
from backend.app.routes.prediction import router as prediction_router
from backend.app.services.predictor import model_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    model_file = MODELS_DIR / DEFAULT_MODEL_NAME
    if not model_file.is_file():
        logger.warning(
            "No trained model found at %s; prediction endpoints will return 503.",
            model_file,
        )
    else:
        logger.info("Loading prediction model: %s", model_file.name)
        model_manager.load_model(model_file.name)
    yield

app = FastAPI(
    title="CNN Satellite Image Land-Cover Classification",
    description="Deep learning-based satellite image land-cover prediction API.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(prediction_router)


@app.get("/")
async def root() -> dict:
    return {"message": "Satellite land-cover classification API is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
