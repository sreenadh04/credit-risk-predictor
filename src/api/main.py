"""
Credit Risk Predictor — FastAPI Backend
========================================
Endpoints:
  GET  /health          — health check
  GET  /metrics         — model performance metrics
  POST /predict         — risk class prediction
  POST /predict/explain — prediction + SHAP explanation
"""

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import METRICS_PATH, MODEL_PATH, RISK_DESCRIPTIONS
from src.inference.predict import explain_prediction, predict
from src.api.schemas import (
    ExplainResponse,
    HealthResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
    RiskInfo,
    ShapFeature,
)
from src.logger import get_logger

logger = get_logger(__name__)

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Risk Predictor API",
    description=(
        "Multi-class credit risk classification API. "
        "Predicts customer risk as P1 (Very Low) → P4 (Very High)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Startup ─────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup — warming up model...")
    try:
        from src.inference.predict import load_model

        load_model()
        logger.info("Model loaded and cached successfully.")
    except FileNotFoundError:
        logger.warning("Model not found at startup. Train the model first.")


# ─── Exception Handlers ──────────────────────────────────────────────────────


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc), "hint": "Run train_pipeline.py first."},
    )


@app.exception_handler(Exception)
async def generic_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Check logs for details."},
    )


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health():
    """Returns API health status and whether the model is loaded."""
    model_loaded = MODEL_PATH.exists()
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Model performance metrics",
    tags=["System"],
)
def get_metrics():
    """Returns test-set metrics from the last training run."""
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run train_pipeline.py first.",
        )
    with open(METRICS_PATH) as f:
        data = json.load(f)
    return MetricsResponse(**data)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict credit risk class",
    tags=["Prediction"],
)
def predict_endpoint(request: PredictionRequest):
    """
    Accepts customer features and returns:
    - `predicted_class` : P1 | P2 | P3 | P4
    - `confidence`      : probability of the predicted class
    - `all_probs`       : full probability distribution
    - `risk_info`       : business label, recommended action, and color code
    """
    raw = request.model_dump()
    logger.info("Prediction request received")

    label, confidence, all_probs = predict(raw)
    info = RISK_DESCRIPTIONS[label]

    return PredictionResponse(
        predicted_class=label,
        confidence=round(confidence, 4),
        all_probs=all_probs,
        risk_info=RiskInfo(
            label=info["label"],
            action=info["action"],
            color=info["color"],
            description=info["description"],
        ),
    )


@app.post(
    "/predict/explain",
    response_model=ExplainResponse,
    summary="Predict + SHAP explanation",
    tags=["Prediction"],
)
def explain_endpoint(request: PredictionRequest):
    """
    Same as /predict but also returns the top SHAP feature contributions
    driving the individual prediction.
    """
    raw = request.model_dump()
    logger.info("Explain request received")

    result = explain_prediction(raw, top_n=10)

    info = result["risk_info"]
    return ExplainResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        all_probs=result["all_probs"],
        top_features=[ShapFeature(**f) for f in result["top_features"]],
        risk_info=RiskInfo(
            label=info["label"],
            action=info["action"],
            color=info["color"],
            description=info["description"],
        ),
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from config.settings import API_HOST, API_PORT, API_RELOAD

    uvicorn.run(
        "src.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
    )
