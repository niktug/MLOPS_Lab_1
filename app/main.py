from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.model import ToxicityModel
from app.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

model = ToxicityModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load()
    yield


app = FastAPI(
    title="Toxicity Classifier API",
    description="Классификация токсичности русского текста",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Сервис жив? Загружена ли модель?"""
    return HealthResponse(status="ok", model_loaded=model.is_loaded)


@app.get("/model_info", response_model=ModelInfoResponse)
def model_info():
    """Метаданные модели."""
    return ModelInfoResponse(**model.get_info())


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Классифицирует один текст."""
    result = model.predict(request.text)
    return PredictResponse(**result)


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    """Классифицирует список текстов."""
    results = model.predict_batch(request.texts)
    return BatchResponse(results=results)
