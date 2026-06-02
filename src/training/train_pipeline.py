"""
Credit Risk Model — Training Pipeline
======================================
Production-grade pipeline:
  - Sklearn Pipeline (no leakage)
  - Domain feature engineering
  - Model comparison (LR, RF, XGBoost)
  - RandomizedSearchCV for tuning
  - SHAP explainability
  - MLflow experiment tracking + model registry
  - Structured logging
"""

import json
import os
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
np.random.seed(42)

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config.settings import (
    ARTIFACT_DIR,
    CLASS_WEIGHTS,
    CV_FOLDS,
    DATA_FILE_1,
    DATA_FILE_2,
    FEATURE_NAMES_PATH,
    METRICS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    RANDOM_STATE,
    SHAP_PLOT_PATH,
    TARGET_COL,
    TARGET_INV,
    TARGET_MAP,
    TEST_SIZE,
    TUNE_CV_FOLDS,
    TUNE_N_ITER,
)
from src.logger import get_logger

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import seaborn as sns

logger = get_logger(__name__)

# ─── Column Definitions ───────────────────────────────────────────────────────

DROP_COLS = ["PROSPECTID", "Credit_Score", "CC_utilization", "PL_utilization"]

CONTINUOUS_COLS = [
    "NETMONTHLYINCOME", "AGE", "Time_With_Curr_Empr",
    "Age_Oldest_TL", "Age_Newest_TL",
    "time_since_recent_payment", "time_since_recent_enq",
    "time_since_first_deliquency", "time_since_recent_deliquency",
    "credit_history_span", "delinquency_rate",
    "recent_activity_ratio", "dpd_severity_score", "enquiry_acceleration",
]

ORDINAL_COLS = ["EDUCATION"]
EDUCATION_ORDER = [["OTHERS", "SSC", "12TH", "UNDER GRADUATE",
                    "GRADUATE", "POST-GRADUATE", "PROFESSIONAL"]]

NOMINAL_COLS = ["last_prod_enq2", "first_prod_enq2"]
BINARY_COLS  = ["GENDER", "MARITALSTATUS"]

ENQ_COUNT_COLS = [
    "PL_enq", "PL_enq_L6m", "PL_enq_L12m",
    "CC_enq", "CC_enq_L6m", "CC_enq_L12m",
    "enq_L3m", "enq_L6m", "enq_L12m", "tot_enq",
]


# ─── 1. Data Loading ──────────────────────────────────────────────────────────

def load_data(path1: Path, path2: Path) -> pd.DataFrame:
    logger.info("Loading data from %s and %s", path1.name, path2.name)
    df1 = pd.read_excel(path1)
    df2 = pd.read_excel(path2)
    df  = pd.merge(df1, df2, on="PROSPECTID", how="inner")
    df.replace(-99999, np.nan, inplace=True)
    logger.info("Merged shape: %s", df.shape)
    return df


# ─── 2. Feature Engineering ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-derived features encoding business intuition."""
    df = df.copy()
    df["delinquency_rate"]       = df["num_times_delinquent"] / (df["Total_TL"] + 1)
    df["recent_activity_ratio"]  = df["Total_TL_opened_L6M"] / (df["Total_TL"] + 1)
    df["credit_history_span"]    = df["Age_Oldest_TL"] - df["Age_Newest_TL"]
    df["dpd_severity_score"]     = df["num_times_30p_dpd"] + 2 * df["num_times_60p_dpd"]
    df["enquiry_acceleration"]   = df["enq_L3m"] / (df["enq_L12m"] + 1)
    df["has_delinquency_history"] = df["time_since_recent_deliquency"].notna().astype(int)
    df["has_payment_history"]    = df["time_since_recent_payment"].notna().astype(int)
    df["has_enquiry_history"]    = df["tot_enq"].notna().astype(int)
    df["has_credit_history"]     = df["Age_Oldest_TL"].notna().astype(int)
    return df


# ─── 3. Prepare X, y ─────────────────────────────────────────────────────────

def prepare_xy(df: pd.DataFrame):
    df = df.copy()
    df["GENDER"]        = df["GENDER"].map({"M": 0, "F": 1})
    df["MARITALSTATUS"] = df["MARITALSTATUS"].map({"Married": 0, "Single": 1})
    df[ENQ_COUNT_COLS]  = df[ENQ_COUNT_COLS].fillna(0)
    y = df[TARGET_COL].map(TARGET_MAP)
    X = df.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")
    return X, y


# ─── 4. Build Pipeline ────────────────────────────────────────────────────────

def build_pipeline(model) -> Pipeline:
    continuous_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
    ])
    ordinal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=EDUCATION_ORDER,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    nominal_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            drop="first",
        )),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("continuous", continuous_transformer, CONTINUOUS_COLS),
            ("ordinal",    ordinal_transformer,    ORDINAL_COLS),
            ("nominal",    nominal_transformer,    NOMINAL_COLS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("final_imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("model", model),
    ])


# ─── 5. Evaluate ─────────────────────────────────────────────────────────────

def evaluate(name: str, pipeline: Pipeline,
             X_train, y_train, X_test, y_test,
             sample_weight=None) -> dict:
    fit_params = {}
    if sample_weight is not None:
        fit_params["model__sample_weight"] = sample_weight

    pipeline.fit(X_train, y_train, **fit_params)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test  = pipeline.predict(X_test)

    train_acc   = accuracy_score(y_train, y_pred_train)
    test_acc    = accuracy_score(y_test,  y_pred_test)
    macro_f1    = f1_score(y_test, y_pred_test, average="macro")
    weighted_f1 = f1_score(y_test, y_pred_test, average="weighted")

    report = classification_report(
        y_test, y_pred_test,
        target_names=["P1", "P2", "P3", "P4"],
        output_dict=True,
    )

    logger.info(
        "%s | train_acc=%.4f test_acc=%.4f macro_f1=%.4f weighted_f1=%.4f gap=%.4f",
        name, train_acc, test_acc, macro_f1, weighted_f1, train_acc - test_acc,
    )
    logger.info("\n%s", classification_report(y_test, y_pred_test,
                                               target_names=["P1", "P2", "P3", "P4"]))

    return {
        "name":        name,
        "train_acc":   round(train_acc,   4),
        "test_acc":    round(test_acc,    4),
        "macro_f1":    round(macro_f1,    4),
        "weighted_f1": round(weighted_f1, 4),
        "p3_recall":   round(report["P3"]["recall"], 4),
        "pipeline":    pipeline,
        "y_test_pred": y_pred_test,
        "report":      report,
    }


# ─── 6. Cross-Validation ─────────────────────────────────────────────────────

def cross_validate_model(pipeline, X_train, y_train, cv: int = CV_FOLDS) -> dict:
    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=skf, scoring="f1_macro", n_jobs=-1)
    logger.info("CV Macro F1 (%d-fold): %.4f ± %.4f", cv, scores.mean(), scores.std())
    return {
        "cv_mean": round(float(scores.mean()), 4),
        "cv_std":  round(float(scores.std()),  4),
    }


# ─── 7. Hyperparameter Tuning ────────────────────────────────────────────────

def tune_xgboost(X_train, y_train, sample_weight=None) -> Pipeline:
    logger.info("Running RandomizedSearchCV for XGBoost (n_iter=%d, cv=%d)",
                TUNE_N_ITER, TUNE_CV_FOLDS)

    base_xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    param_dist = {
        "model__n_estimators":     [200, 300, 400],
        "model__max_depth":        [3, 4, 5, 6],
        "model__learning_rate":    [0.03, 0.05, 0.1],
        "model__subsample":        [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.6, 0.7, 0.8],
        "model__min_child_weight": [3, 4, 5, 6],
    }

    pipeline = build_pipeline(base_xgb)
    fit_params = {}
    if sample_weight is not None:
        fit_params["model__sample_weight"] = sample_weight

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=TUNE_N_ITER,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=TUNE_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    search.fit(X_train, y_train, **fit_params)

    logger.info("Best CV Macro F1: %.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)
    return search.best_estimator_


# ─── 8. SHAP Explainability ──────────────────────────────────────────────────

def explain_with_shap(pipeline: Pipeline, X_test: pd.DataFrame) -> str:
    logger.info("Computing SHAP values...")

    model        = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_test)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]

    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    plt.figure(figsize=(10, 7))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[2], X_df, show=False,
                          plot_type="beeswarm", max_display=20)
    else:
        shap.summary_plot(shap_values, X_df, show=False,
                          plot_type="beeswarm", max_display=20)

    plt.title("SHAP Feature Importance — P3 (High Risk Class)", fontsize=14)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP plot saved to %s", SHAP_PLOT_PATH)
    return str(SHAP_PLOT_PATH)


# ─── 9. Confusion Matrix Plot ────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["P1", "P2", "P3", "P4"],
                yticklabels=["P1", "P2", "P3", "P4"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    path = ARTIFACT_DIR / f"cm_{model_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", path)
    return str(path)


# ─── 10. Main Entry Point ────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("  CREDIT RISK MODEL — TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ── Load data
    df = load_data(DATA_FILE_1, DATA_FILE_2)
    df = engineer_features(df)
    X, y = prepare_xy(df)

    # ── Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    logger.info("Train: %s  Test: %s", X_train.shape, X_test.shape)

    # ── Sample weights (class imbalance)
    sample_weights = compute_sample_weight(class_weight=CLASS_WEIGHTS, y=y_train)

    all_results = []

    # ── 1. Logistic Regression (baseline)
    with mlflow.start_run(run_name="LogisticRegression", nested=False):
        lr = build_pipeline(LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1,
            class_weight={0: 1, 1: 1, 2: 5, 3: 3},
        ))
        r_lr = evaluate("Logistic Regression", lr, X_train, y_train, X_test, y_test)
        cv_lr = cross_validate_model(lr, X_train, y_train)
        mlflow.log_metrics({
            "test_accuracy": r_lr["test_acc"],
            "macro_f1":      r_lr["macro_f1"],
            "cv_mean_f1":    cv_lr["cv_mean"],
        })
        mlflow.log_param("model_type", "LogisticRegression")
        all_results.append({**r_lr, **cv_lr})

    # ── 2. Random Forest
    with mlflow.start_run(run_name="RandomForest", nested=False):
        rf = build_pipeline(RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_split=50,
            min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE,
            class_weight={0: 1, 1: 1, 2: 5, 3: 3},
        ))
        r_rf = evaluate("Random Forest", rf, X_train, y_train, X_test, y_test)
        cv_rf = cross_validate_model(rf, X_train, y_train)
        mlflow.log_metrics({
            "test_accuracy": r_rf["test_acc"],
            "macro_f1":      r_rf["macro_f1"],
            "cv_mean_f1":    cv_rf["cv_mean"],
        })
        mlflow.log_param("model_type", "RandomForest")
        all_results.append({**r_rf, **cv_rf})

    # ── 3. XGBoost (tuned + weighted) — main MLflow run
    with mlflow.start_run(run_name="XGBoost_Tuned_Weighted") as run:
        best_pipeline = tune_xgboost(X_train, y_train, sample_weights)
        r_xgb = evaluate(
            "XGBoost (Tuned + Weighted)", best_pipeline,
            X_train, y_train, X_test, y_test, sample_weights,
        )
        cv_xgb = cross_validate_model(best_pipeline, X_train, y_train)
        cm_path = plot_confusion_matrix(y_test, r_xgb["y_test_pred"], "XGBoost_Weighted")
        shap_path = explain_with_shap(best_pipeline, X_test)

        # Log metrics
        mlflow.log_metrics({
            "train_accuracy": r_xgb["train_acc"],
            "test_accuracy":  r_xgb["test_acc"],
            "macro_f1":       r_xgb["macro_f1"],
            "weighted_f1":    r_xgb["weighted_f1"],
            "p3_recall":      r_xgb["p3_recall"],
            "cv_mean_f1":     cv_xgb["cv_mean"],
            "cv_std_f1":      cv_xgb["cv_std"],
        })

        # Log params
        xgb_model = best_pipeline.named_steps["model"]
        mlflow.log_params({
            "model_type":        "XGBClassifier",
            "n_estimators":      xgb_model.n_estimators,
            "max_depth":         xgb_model.max_depth,
            "learning_rate":     xgb_model.learning_rate,
            "subsample":         xgb_model.subsample,
            "colsample_bytree":  xgb_model.colsample_bytree,
            "min_child_weight":  xgb_model.min_child_weight,
            "test_size":         TEST_SIZE,
            "random_state":      RANDOM_STATE,
        })

        # Log artifacts
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(shap_path)

        # Register model in MLflow Model Registry
        mlflow.sklearn.log_model(
            best_pipeline,
            artifact_path="model",
            registered_model_name="credit-risk-xgboost",
        )

        run_id = run.info.run_id
        logger.info("MLflow run_id: %s", run_id)
        all_results.append({**r_xgb, **cv_xgb})

    # ── Model comparison table
    comparison_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ["pipeline", "y_test_pred", "report"]}
        for r in all_results
    ])
    logger.info("\n%s", comparison_df.to_string(index=False))

    # ── Save artifacts
    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(list(X_train.columns), f, indent=2)
    logger.info("Feature names saved to %s", FEATURE_NAMES_PATH)

    metrics = {
        "test_accuracy": r_xgb["test_acc"],
        "macro_f1":      r_xgb["macro_f1"],
        "weighted_f1":   r_xgb["weighted_f1"],
        "p3_recall":     r_xgb["p3_recall"],
        "cv_mean_f1":    cv_xgb["cv_mean"],
        "cv_std_f1":     cv_xgb["cv_std"],
        "mlflow_run_id": run_id,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", METRICS_PATH)

    logger.info("Training complete.")
    return best_pipeline, comparison_df


if __name__ == "__main__":
    main()
