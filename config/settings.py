"""
Centralized configuration — all settings in one place.
Reads from environment variables with sensible defaults.
"""

import os
from pathlib import Path

# ─── Project Root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
LOG_DIR = ROOT_DIR / "logs"

# Ensure directories exist
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Model Artifacts ──────────────────────────────────────────────────────────
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
FEATURE_NAMES_PATH = ARTIFACT_DIR / "feature_names.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
SHAP_PLOT_PATH = ARTIFACT_DIR / "shap_summary_P3.png"

# ─── Data Files ───────────────────────────────────────────────────────────────
DATA_FILE_1 = DATA_DIR / "case_study1.xlsx"
DATA_FILE_2 = DATA_DIR / "case_study2.xlsx"

# ─── API Settings ─────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))
API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"

# ─── Streamlit Settings ───────────────────────────────────────────────────────
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", f"http://localhost:{API_PORT}")

# ─── MLflow Settings ──────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(ROOT_DIR / "mlruns"))
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit-risk-prediction")
MLFLOW_REGISTERED_MODEL = "credit-risk-xgboost"

# ─── Training Settings ────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TUNE_N_ITER = 20
TUNE_CV_FOLDS = 3

# Class weights — P3 (high risk) and P4 boosted
CLASS_WEIGHTS = {0: 1, 1: 1, 2: 5, 3: 3}

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE = LOG_DIR / "app.log"

# ─── Target Mapping ───────────────────────────────────────────────────────────
TARGET_COL = "Approved_Flag"
TARGET_MAP = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
TARGET_INV = {v: k for k, v in TARGET_MAP.items()}

RISK_DESCRIPTIONS = {
    "P1": {
        "label": "Very Low Risk",
        "action": "Auto-Approve",
        "color": "#2ecc71",
        "bg": "#0d2b1e",
        "description": "Excellent credit profile. Recommend immediate approval with standard terms.",
    },
    "P2": {
        "label": "Low Risk",
        "action": "Approve with Review",
        "color": "#f39c12",
        "bg": "#2b220d",
        "description": "Good credit profile. Standard approval with minor documentation check.",
    },
    "P3": {
        "label": "High Risk",
        "action": "Manual Review Required",
        "color": "#e74c3c",
        "bg": "#2b0d0d",
        "description": "Elevated delinquency signals. Send for manual underwriting review.",
    },
    "P4": {
        "label": "Very High Risk",
        "action": "Reject / Offer Secured Only",
        "color": "#8e44ad",
        "bg": "#1e0d2b",
        "description": "High delinquency and payment failure history. Reject or offer secured product.",
    },
}
