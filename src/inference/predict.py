"""
Credit Risk Model — Inference Pipeline
=======================================
Load trained pipeline, validate input, predict risk class + confidence.
Includes SHAP individual explanations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    RISK_DESCRIPTIONS,
    TARGET_INV,
)
from src.logger import get_logger

logger = get_logger(__name__)

# Module-level cache so we don't reload on every request
_pipeline  = None
_feat_names: Optional[List[str]] = None


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train_pipeline.py first."
            )
        logger.info("Loading model from %s", MODEL_PATH)
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def load_feature_names() -> Optional[List[str]]:
    global _feat_names
    if _feat_names is None and FEATURE_NAMES_PATH.exists():
        with open(FEATURE_NAMES_PATH) as f:
            _feat_names = json.load(f)
    return _feat_names


# ─── Preprocessing ───────────────────────────────────────────────────────────

ENQ_COUNT_COLS = [
    "PL_enq", "PL_enq_L6m", "PL_enq_L12m",
    "CC_enq", "CC_enq_L6m", "CC_enq_L12m",
    "enq_L3m", "enq_L6m", "enq_L12m", "tot_enq",
]
DROP_COLS = ["PROSPECTID", "Credit_Score", "CC_utilization", "PL_utilization"]


def preprocess_input(raw_input: Dict) -> pd.DataFrame:
    """Convert a single user-input dict into a DataFrame for the sklearn Pipeline."""
    # Avoid circular import — import here
    from src.training.train_pipeline import engineer_features

    df = pd.DataFrame([raw_input])
    df.replace(-99999, np.nan, inplace=True)
    df = engineer_features(df)
    df["GENDER"]        = df["GENDER"].map({"M": 0, "F": 1})
    df["MARITALSTATUS"] = df["MARITALSTATUS"].map({"Married": 0, "Single": 1})
    df[ENQ_COUNT_COLS]  = df[ENQ_COUNT_COLS].fillna(0)

    drop = [c for c in DROP_COLS + ["PROSPECTID", "Approved_Flag"] if c in df.columns]
    df.drop(columns=drop, inplace=True, errors="ignore")

    return df


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict(raw_input: Dict) -> Tuple[str, float, Dict[str, float]]:
    """
    Main inference function.

    Returns
    -------
    label      : str    — "P1" | "P2" | "P3" | "P4"
    confidence : float  — probability of predicted class
    all_probs  : dict   — {P1: float, P2: float, P3: float, P4: float}
    """
    pipeline = load_model()
    df = preprocess_input(raw_input)

    pred_class = int(pipeline.predict(df)[0])
    pred_probs = pipeline.predict_proba(df)[0]

    label      = TARGET_INV[pred_class]
    confidence = float(pred_probs[pred_class])
    all_probs  = {TARGET_INV[i]: round(float(p), 4) for i, p in enumerate(pred_probs)}

    logger.info("Prediction: %s (confidence=%.3f)", label, confidence)
    return label, confidence, all_probs


# ─── SHAP Explanation ─────────────────────────────────────────────────────────

def explain_prediction(raw_input: Dict, top_n: int = 10) -> Dict:
    """
    Generate SHAP-based explanation for a single prediction.

    Returns a dict with:
      - predicted_class
      - confidence
      - all_probs
      - top_features : list of {feature, shap_value, direction}
    """
    pipeline = load_model()
    df = preprocess_input(raw_input)

    preprocessor  = pipeline.named_steps["preprocessor"]
    model         = pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(df)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    pred_class = int(pipeline.predict(df)[0])
    pred_probs = pipeline.predict_proba(df)[0]
    label      = TARGET_INV[pred_class]
    all_probs  = {TARGET_INV[i]: round(float(p), 4) for i, p in enumerate(pred_probs)}

    # Extract SHAP for predicted class
    if isinstance(shap_values, list):
        sv = shap_values[pred_class][0]
    else:
        # New SHAP format: (samples, features, classes)
        if len(shap_values.shape) == 3:
            sv = shap_values[0, :, pred_class]
        else:
            sv = shap_values[0]
    indices = np.argsort(np.abs(sv))[::-1][:top_n]
    top_features = [
        {
            "feature":    feature_names[i],
            "shap_value": round(float(sv[i]), 4),
            "direction":  "increases_risk" if sv[i] > 0 else "decreases_risk",
        }
        for i in indices
    ]

    return {
        "predicted_class": label,
        "confidence":      round(float(pred_probs[pred_class]), 4),
        "all_probs":       all_probs,
        "top_features":    top_features,
        "risk_info":       RISK_DESCRIPTIONS[label],
    }


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = {
        "Total_TL": 12, "Tot_Closed_TL": 4, "Tot_Active_TL": 8,
        "Total_TL_opened_L6M": 1, "Tot_TL_closed_L6M": 0,
        "pct_tl_open_L6M": 8.3, "pct_tl_closed_L6M": 0.0,
        "pct_active_tl": 66.7, "pct_closed_tl": 33.3,
        "Total_TL_opened_L12M": 2, "Tot_TL_closed_L12M": 1,
        "pct_tl_open_L12M": 16.7, "pct_tl_closed_L12M": 8.3,
        "Tot_Missed_Pmnt": 0,
        "Auto_TL": 1, "CC_TL": 2, "Consumer_TL": 1,
        "Gold_TL": 0, "Home_TL": 0, "PL_TL": 1,
        "Secured_TL": 2, "Unsecured_TL": 3, "Other_TL": 1,
        "Age_Oldest_TL": 72, "Age_Newest_TL": 6,
        "time_since_recent_payment": 1,
        "time_since_first_deliquency": np.nan,
        "time_since_recent_deliquency": np.nan,
        "num_times_delinquent": 0, "max_delinquency_level": 0,
        "max_recent_level_of_deliq": 0,
        "num_deliq_6mts": 0, "num_deliq_12mts": 0, "num_deliq_6_12mts": 0,
        "max_deliq_6mts": 0, "max_deliq_12mts": 0,
        "num_times_30p_dpd": 0, "num_times_60p_dpd": 0,
        "num_std": 10, "num_std_6mts": 1, "num_std_12mts": 2,
        "num_sub": 0, "num_sub_6mts": 0, "num_sub_12mts": 0,
        "num_dbt": 0, "num_dbt_6mts": 0, "num_dbt_12mts": 0,
        "num_lss": 0, "num_lss_6mts": 0, "num_lss_12mts": 0,
        "recent_level_of_deliq": 0,
        "tot_enq": 5, "CC_enq": 2, "CC_enq_L6m": 1, "CC_enq_L12m": 2,
        "PL_enq": 2, "PL_enq_L6m": 0, "PL_enq_L12m": 1,
        "time_since_recent_enq": 2,
        "enq_L12m": 3, "enq_L6m": 1, "enq_L3m": 0,
        "MARITALSTATUS": "Married", "EDUCATION": "GRADUATE",
        "AGE": 34, "GENDER": "M", "NETMONTHLYINCOME": 45000,
        "Time_With_Curr_Empr": 36,
        "pct_of_active_TLs_ever": 66.7, "pct_opened_TLs_L6m_of_L12m": 50.0,
        "pct_currentBal_all_TL": 45.0,
        "CC_Flag": 1, "PL_Flag": 1, "HL_Flag": 0, "GL_Flag": 0,
        "pct_PL_enq_L6m_of_L12m": 0.0, "pct_CC_enq_L6m_of_L12m": 50.0,
        "pct_PL_enq_L6m_of_ever": 0.0, "pct_CC_enq_L6m_of_ever": 50.0,
        "max_unsec_exposure_inPct": 60.0,
        "last_prod_enq2": "PL", "first_prod_enq2": "CC",
    }

    label, confidence, all_probs = predict(sample)
    info = RISK_DESCRIPTIONS[label]
    print(f"\nPredicted Risk Class : {label} — {info['label']}")
    print(f"Confidence           : {confidence:.1%}")
    print(f"All probabilities    : {all_probs}")
