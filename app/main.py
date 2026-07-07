from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import logging

from app.schemas import NetworkFlowInput, PredictionResponse, HealthResponse, BatchInput, BatchResponse
from app.model_handler import ModelHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_handler = ModelHandler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(" Chargement du modèle...")
    model_handler.load()
    logger.info(" Modèle chargé avec succès")
    yield
    logger.info(" Arrêt de l'application")


app = FastAPI(
    title="Network Anomaly Detection API",
    description="API de détection d'anomalies réseau en temps réel",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Vérifie l'état de l'API et du modèle."""
    return HealthResponse(
        status="ok",
        model_loaded=model_handler.is_loaded(),
        model_version=model_handler.version,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(flow: NetworkFlowInput):
    """
    Analyse un flux réseau et retourne une prédiction d'anomalie.

    - **label 0** → trafic normal
    - **label 1** → anomalie détectée
    """
    if not model_handler.is_loaded():
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    start = time.perf_counter()
    result = model_handler.predict(flow)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        f"Prédiction: label={result['label']}, score={result['anomaly_score']:.4f}, latency={latency_ms}ms"
    )

    return PredictionResponse(
        label=result["label"],
        label_text="anomalie" if result["label"] == 1 else "normal",
        anomaly_score=result["anomaly_score"],
        confidence=result["confidence"],
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(batch: BatchInput):
    """Analyse un lot de flux réseau (max 100 par requête)."""
    if not model_handler.is_loaded():
        raise HTTPException(status_code=503, detail="Modèle non disponible")

    if len(batch.flows) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 flux par batch")

    start = time.perf_counter()
    predictions = [model_handler.predict(flow) for flow in batch.flows]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    results = [
        PredictionResponse(
            label=p["label"],
            label_text="anomalie" if p["label"] == 1 else "normal",
            anomaly_score=p["anomaly_score"],
            confidence=p["confidence"],
            latency_ms=0,
        )
        for p in predictions
    ]

    anomaly_count = sum(1 for p in predictions if p["label"] == 1)

    return BatchResponse(
        predictions=results,
        total=len(results),
        anomaly_count=anomaly_count,
        normal_count=len(results) - anomaly_count,
        total_latency_ms=latency_ms,
    )
