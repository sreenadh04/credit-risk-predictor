"""
API endpoint tests using FastAPI TestClient.
Run after training the model: pytest tests/api/
"""

import sys
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.api.main import app  # noqa: E402

client = TestClient(app, raise_server_exceptions=False)


def _sample_payload() -> dict:
    return {
        "AGE": 34,
        "GENDER": "M",
        "MARITALSTATUS": "Married",
        "EDUCATION": "GRADUATE",
        "NETMONTHLYINCOME": 45000,
        "Time_With_Curr_Empr": 36,
        "Total_TL": 12,
        "Tot_Closed_TL": 4,
        "Tot_Active_TL": 8,
        "Total_TL_opened_L6M": 1,
        "Tot_TL_closed_L6M": 0,
        "pct_tl_open_L6M": 8.3,
        "pct_tl_closed_L6M": 0.0,
        "pct_active_tl": 66.7,
        "pct_closed_tl": 33.3,
        "Total_TL_opened_L12M": 2,
        "Tot_TL_closed_L12M": 1,
        "pct_tl_open_L12M": 16.7,
        "pct_tl_closed_L12M": 8.3,
        "Tot_Missed_Pmnt": 0,
        "Age_Oldest_TL": 72,
        "Age_Newest_TL": 6,
        "Auto_TL": 1,
        "CC_TL": 2,
        "Consumer_TL": 1,
        "Gold_TL": 0,
        "Home_TL": 0,
        "PL_TL": 1,
        "Secured_TL": 2,
        "Unsecured_TL": 3,
        "Other_TL": 1,
        "time_since_recent_payment": 1.0,
        "time_since_first_deliquency": None,
        "time_since_recent_deliquency": None,
        "num_times_delinquent": 0,
        "max_delinquency_level": 0,
        "max_recent_level_of_deliq": 0,
        "num_deliq_6mts": 0,
        "num_deliq_12mts": 0,
        "num_deliq_6_12mts": 0,
        "max_deliq_6mts": 0,
        "max_deliq_12mts": 0,
        "num_times_30p_dpd": 0,
        "num_times_60p_dpd": 0,
        "recent_level_of_deliq": 0,
        "num_std": 10,
        "num_std_6mts": 1,
        "num_std_12mts": 2,
        "num_sub": 0,
        "num_sub_6mts": 0,
        "num_sub_12mts": 0,
        "num_dbt": 0,
        "num_dbt_6mts": 0,
        "num_dbt_12mts": 0,
        "num_lss": 0,
        "num_lss_6mts": 0,
        "num_lss_12mts": 0,
        "tot_enq": 5,
        "CC_enq": 2,
        "CC_enq_L6m": 1,
        "CC_enq_L12m": 2,
        "PL_enq": 2,
        "PL_enq_L6m": 0,
        "PL_enq_L12m": 1,
        "time_since_recent_enq": 2.0,
        "enq_L12m": 3,
        "enq_L6m": 1,
        "enq_L3m": 0,
        "CC_Flag": 1,
        "PL_Flag": 1,
        "HL_Flag": 0,
        "GL_Flag": 0,
        "pct_of_active_TLs_ever": 66.7,
        "pct_opened_TLs_L6m_of_L12m": 50.0,
        "pct_currentBal_all_TL": 45.0,
        "pct_PL_enq_L6m_of_L12m": 0.0,
        "pct_CC_enq_L6m_of_L12m": 50.0,
        "pct_PL_enq_L6m_of_ever": 0.0,
        "pct_CC_enq_L6m_of_ever": 50.0,
        "max_unsec_exposure_inPct": 60.0,
        "last_prod_enq2": "PL",
        "first_prod_enq2": "CC",
    }


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_field(self):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")

    def test_health_has_model_loaded_field(self):
        response = client.get("/health")
        data = response.json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_has_version(self):
        response = client.get("/health")
        data = response.json()
        assert "version" in data


class TestPredictEndpoint:
    def test_predict_returns_200_with_valid_payload(self):
        mock_result = ("P1", 0.85, {"P1": 0.85, "P2": 0.10, "P3": 0.03, "P4": 0.02})
        with patch("src.api.main.predict", return_value=mock_result):
            response = client.post("/predict", json=_sample_payload())
        assert response.status_code == 200

    def test_predict_response_has_required_fields(self):
        mock_result = ("P1", 0.85, {"P1": 0.85, "P2": 0.10, "P3": 0.03, "P4": 0.02})
        with patch("src.api.main.predict", return_value=mock_result):
            response = client.post("/predict", json=_sample_payload())
        data = response.json()
        for field in ["predicted_class", "confidence", "all_probs", "risk_info"]:
            assert field in data, f"Missing field: {field}"

    def test_predict_class_is_valid(self):
        mock_result = ("P3", 0.72, {"P1": 0.05, "P2": 0.10, "P3": 0.72, "P4": 0.13})
        with patch("src.api.main.predict", return_value=mock_result):
            response = client.post("/predict", json=_sample_payload())
        data = response.json()
        assert data["predicted_class"] in ("P1", "P2", "P3", "P4")

    def test_predict_confidence_between_0_and_1(self):
        mock_result = ("P1", 0.85, {"P1": 0.85, "P2": 0.10, "P3": 0.03, "P4": 0.02})
        with patch("src.api.main.predict", return_value=mock_result):
            response = client.post("/predict", json=_sample_payload())
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_all_probs_sum_to_one(self):
        mock_result = ("P1", 0.85, {"P1": 0.85, "P2": 0.10, "P3": 0.03, "P4": 0.02})
        with patch("src.api.main.predict", return_value=mock_result):
            response = client.post("/predict", json=_sample_payload())
        data = response.json()
        total = sum(data["all_probs"].values())
        assert abs(total - 1.0) < 0.01

    def test_predict_missing_required_field_returns_422(self):
        payload = _sample_payload()
        del payload["AGE"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_invalid_gender_returns_422(self):
        payload = _sample_payload()
        payload["GENDER"] = "X"
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_age_returns_422(self):
        payload = _sample_payload()
        payload["AGE"] = -5
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_risk_info_has_label_and_action(self):
        mock_result = ("P2", 0.70, {"P1": 0.10, "P2": 0.70, "P3": 0.15, "P4": 0.05})
        with patch("src.api.main.predict", return_value=mock_result):
            response = client.post("/predict", json=_sample_payload())
        data = response.json()
        assert "label" in data["risk_info"]
        assert "action" in data["risk_info"]


class TestExplainEndpoint:
    def test_explain_returns_200(self):
        mock_result = {
            "predicted_class": "P1",
            "confidence": 0.85,
            "all_probs": {"P1": 0.85, "P2": 0.10, "P3": 0.03, "P4": 0.02},
            "top_features": [
                {
                    "feature": "delinquency_rate",
                    "shap_value": -0.5,
                    "direction": "decreases_risk",
                },
                {"feature": "AGE", "shap_value": 0.2, "direction": "increases_risk"},
            ],
            "risk_info": {
                "label": "Very Low Risk",
                "action": "Auto-Approve",
                "color": "#2ecc71",
                "bg": "#0d2b1e",
                "description": "Excellent credit profile.",
            },
        }
        with patch("src.api.main.explain_prediction", return_value=mock_result):
            response = client.post("/predict/explain", json=_sample_payload())
        assert response.status_code == 200

    def test_explain_has_top_features(self):
        mock_result = {
            "predicted_class": "P1",
            "confidence": 0.85,
            "all_probs": {"P1": 0.85, "P2": 0.10, "P3": 0.03, "P4": 0.02},
            "top_features": [
                {
                    "feature": "delinquency_rate",
                    "shap_value": -0.5,
                    "direction": "decreases_risk",
                },
            ],
            "risk_info": {
                "label": "Very Low Risk",
                "action": "Auto-Approve",
                "color": "#2ecc71",
                "bg": "#0d2b1e",
                "description": "Excellent credit profile.",
            },
        }
        with patch("src.api.main.explain_prediction", return_value=mock_result):
            response = client.post("/predict/explain", json=_sample_payload())
        data = response.json()
        assert "top_features" in data
        assert len(data["top_features"]) > 0


class TestDocsEndpoints:
    def test_openapi_docs_accessible(self):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_accessible(self):
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_json_accessible(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
