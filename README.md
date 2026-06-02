# 🏦 Credit Risk Predictor

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-blue)](https://mlflow.org)
[![CI](https://github.com/YOUR_USERNAME/credit-risk-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/credit-risk-predictor/actions)

A **production-grade multi-class credit risk classification system** that classifies loan applicants into four risk tiers using 60+ credit bureau behavioral signals, backed by a FastAPI REST API and an interactive Streamlit dashboard with SHAP-powered explanations.

---

## Problem Statement

Banks and NBFCs need to consistently classify loan applicants into risk tiers to make data-driven lending decisions. This system classifies customers into:

| Class | Risk Level | Business Action | Approx. Share |
|-------|-----------|-----------------|---------------|
| P1 | Very Low Risk | ✅ Auto-Approve | ~30% |
| P2 | Low Risk | ✅ Approve with Review | ~35% |
| P3 | High Risk | ⚠️ Manual Underwriting | ~20% |
| P4 | Very High Risk | ❌ Reject / Secured Only | ~15% |

---

## Key Results

| Model | Test Accuracy | Macro F1 | P3 Recall |
|-------|--------------|----------|-----------|
| Logistic Regression (baseline) | ~72% | 0.64 | 0.51 |
| Random Forest | ~81% | 0.73 | 0.68 |
| **XGBoost (Tuned + Weighted)** | **~80%** | **0.77** | **0.76** |

> **Note:** The 80% accuracy at 0.77 macro F1 is the *correct* result after removing `Credit_Score` — a proxy leakage variable that inflated accuracy to 99.5% by letting the model memorize label assignment rules rather than learning customer behavior.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User / Browser                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│              Streamlit Frontend  :8501                       │
│   Input form → call FastAPI → render results + SHAP chart   │
└───────────────────────┬─────────────────────────────────────┘
                        │  REST API
┌───────────────────────▼─────────────────────────────────────┐
│           FastAPI Backend  :8000                             │
│   POST /predict        — risk class + probabilities          │
│   POST /predict/explain — prediction + SHAP features         │
│   GET  /health          — service health                     │
│   GET  /metrics         — model performance metrics          │
└───────────────────────┬─────────────────────────────────────┘
                        │  joblib.load
┌───────────────────────▼─────────────────────────────────────┐
│           Sklearn Pipeline  (artifacts/model.joblib)        │
│   ColumnTransformer → RobustScaler / OrdinalEncoder /        │
│   OneHotEncoder / SimpleImputer → XGBClassifier             │
└───────────────────────┬─────────────────────────────────────┘
                        │  logged by
┌───────────────────────▼─────────────────────────────────────┐
│                MLflow Experiment Tracking                    │
│   Metrics / Parameters / Artifacts / Model Registry         │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Streamlit | Input form, result visualization, SHAP waterfall charts |
| Backend | FastAPI + Pydantic | REST API, input validation, error handling |
| ML Model | XGBoost + scikit-learn Pipeline | Multi-class classification with preprocessing |
| Explainability | SHAP TreeExplainer | Global (training) + local (per-prediction) feature importance |
| MLOps | MLflow | Experiment tracking, model registry, metric comparison |
| Testing | Pytest | Unit tests, API tests, schema contract tests |
| CI/CD | GitHub Actions | Lint + test + Docker build on every push |
| Deployment | Render | Free-tier cloud deployment |

---

## Project Structure

```
credit-risk-predictor/
├── app/
│   └── streamlit_app.py          # Streamlit frontend
├── config/
│   └── settings.py               # Centralized config (env-var aware)
├── src/
│   ├── logger.py                 # Structured logging setup
│   ├── training/
│   │   └── train_pipeline.py     # Full training pipeline + MLflow
│   ├── inference/
│   │   └── predict.py            # Inference + SHAP explanation
│   └── api/
│       ├── main.py               # FastAPI app + endpoints
│       └── schemas.py            # Pydantic request/response models
├── tests/
│   ├── unit/
│   │   └── test_features.py      # Feature engineering unit tests
│   ├── api/
│   │   └── test_endpoints.py     # FastAPI endpoint tests
│   └── model/
│       └── test_model_contract.py # Schema + preprocessing tests
├── data/
│   └── raw/                      # Source Excel files (not committed)
├── artifacts/                    # Model artifacts (not committed)
├── notebooks/                    # EDA and exploration notebooks
├── .github/workflows/ci.yml      # GitHub Actions CI
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Local multi-service setup
├── render.yaml                   # Render deployment config
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Local Development Setup

### Prerequisites

- Python 3.12
- Git
- Docker (optional, for containerized run)

### 1. Clone & Set Up

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-predictor.git
cd credit-risk-predictor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Add Data Files

Place the two source Excel files in `data/raw/`:

```
data/raw/case_study1.xlsx
data/raw/case_study2.xlsx
```

### 3. Train the Model

```bash
python -m src.training.train_pipeline
```

This will:
- Load and merge the two datasets
- Engineer domain features
- Train Logistic Regression, Random Forest, and XGBoost
- Run hyperparameter tuning (RandomizedSearchCV)
- Log all experiments to MLflow
- Save `artifacts/model.joblib`, `feature_names.json`, `metrics.json`
- Generate `artifacts/shap_summary_P3.png`

### 4. Start the FastAPI Backend

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### 5. Start the Streamlit Frontend

In a second terminal:

```bash
streamlit run app/streamlit_app.py
```

Frontend: http://localhost:8501

### 6. View MLflow Dashboard

```bash
mlflow ui --backend-store-uri ./mlruns
```

MLflow UI: http://localhost:5000

---

## Docker

### Run Everything with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Services:
#   FastAPI:   http://localhost:8000
#   Streamlit: http://localhost:8501
#   MLflow UI: http://localhost:5000

# Stop
docker-compose down
```

---

## API Documentation

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### `GET /metrics`

Returns test-set performance from the last training run.

### `POST /predict`

**Request body:** Full customer feature object (see `/docs` for schema).

**Response:**
```json
{
  "predicted_class": "P1",
  "confidence": 0.8521,
  "all_probs": { "P1": 0.8521, "P2": 0.0981, "P3": 0.0312, "P4": 0.0186 },
  "risk_info": {
    "label": "Very Low Risk",
    "action": "Auto-Approve",
    "color": "#2ecc71",
    "description": "Excellent credit profile..."
  }
}
```

### `POST /predict/explain`

Same as `/predict` but adds SHAP feature contributions:

```json
{
  "predicted_class": "P1",
  "confidence": 0.8521,
  "all_probs": {...},
  "top_features": [
    { "feature": "delinquency_rate", "shap_value": -0.412, "direction": "decreases_risk" },
    { "feature": "dpd_severity_score", "shap_value": -0.287, "direction": "decreases_risk" }
  ],
  "risk_info": {...}
}
```

---

## Testing

```bash
# Run all tests (no model required)
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only API tests
pytest tests/api/

# Skip tests that require a trained model
pytest -m "not requires_model"
```

---

## Deployment on Render

### Step-by-Step

1. Push your code to GitHub (include `artifacts/` with trained model)
2. Go to [render.com](https://render.com) → New → Blueprint
3. Connect your GitHub repo
4. Render reads `render.yaml` and creates both services automatically
5. Add environment variables in the Render dashboard if needed

### Manual Deploy (single service)

1. **New Web Service** → connect repo
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
4. **Environment Variables:** set `LOG_LEVEL=INFO`

> **Important:** Render's free tier spins down after 15 minutes of inactivity. First request after spin-down takes ~30 seconds.

---

## Key ML Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| Removed `Credit_Score` | Proxy leakage — it's derived from the same features and directly encodes the label assignment rule |
| `RobustScaler` over `StandardScaler` | Income and age have heavy-tailed distributions; robust scaler is median/IQR-based and ignores outliers |
| `OrdinalEncoder` with `handle_unknown` | Pipeline-safe encoding; `handle_unknown='use_encoded_value'` prevents silent NaN bugs on new education values |
| Class weights `{P3: 5, P4: 3}` | P3 (high risk) has highest business cost if missed; boosting recall from 0.29 → 0.76 |
| `RandomizedSearchCV` over `GridSearchCV` | Same coverage at 1/10th the compute for large hyperparameter spaces |
| Sklearn Pipeline (not manual preprocessing) | Zero preprocessing leakage; `fit_transform` on train, `transform` on test |
| SHAP `TreeExplainer` | Model-specific (XGBoost), exact SHAP values, fast — unlike `KernelExplainer` |

---

## Troubleshooting

**"Model not found" error**
```bash
python -m src.training.train_pipeline
```

**`ImportError` or `ModuleNotFoundError`**

Make sure you're running from the project root and your venv is activated:
```bash
cd credit-risk-predictor
source .venv/bin/activate
```

**Streamlit can't connect to API**

The Streamlit app falls back to direct inference automatically. Check the sidebar for "● API: OFFLINE (direct mode)".

**Port already in use**
```bash
lsof -i :8000 | awk 'NR>1 {print $2}' | xargs kill -9
```

---

## License

MIT
