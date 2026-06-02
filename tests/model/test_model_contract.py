"""
Model contract tests — validate schema, prediction types, and probability properties.
These tests run WITHOUT a trained model (schema/logic only).
Tests that require a trained model are marked with @pytest.mark.requires_model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestPredictionRequestSchema:
    """Validate that our Pydantic schemas enforce the right rules."""

    def _base_payload(self) -> dict:
        return {
            "AGE": 34, "GENDER": "M", "MARITALSTATUS": "Married",
            "EDUCATION": "GRADUATE", "NETMONTHLYINCOME": 45000,
            "Time_With_Curr_Empr": 36,
            "Total_TL": 12, "Tot_Closed_TL": 4, "Tot_Active_TL": 8,
            "Total_TL_opened_L6M": 1, "Tot_TL_closed_L6M": 0,
            "pct_tl_open_L6M": 8.3, "pct_tl_closed_L6M": 0.0,
            "pct_active_tl": 66.7, "pct_closed_tl": 33.3,
            "Total_TL_opened_L12M": 2, "Tot_TL_closed_L12M": 1,
            "pct_tl_open_L12M": 16.7, "pct_tl_closed_L12M": 8.3,
            "Tot_Missed_Pmnt": 0, "Age_Oldest_TL": 72, "Age_Newest_TL": 6,
            "Auto_TL": 1, "CC_TL": 2, "Consumer_TL": 1,
            "Gold_TL": 0, "Home_TL": 0, "PL_TL": 1,
            "Secured_TL": 2, "Unsecured_TL": 3, "Other_TL": 1,
            "time_since_recent_payment": 1.0,
            "time_since_first_deliquency": None,
            "time_since_recent_deliquency": None,
            "num_times_delinquent": 0, "max_delinquency_level": 0,
            "max_recent_level_of_deliq": 0,
            "num_deliq_6mts": 0, "num_deliq_12mts": 0, "num_deliq_6_12mts": 0,
            "max_deliq_6mts": 0, "max_deliq_12mts": 0,
            "num_times_30p_dpd": 0, "num_times_60p_dpd": 0, "recent_level_of_deliq": 0,
            "num_std": 10, "num_std_6mts": 1, "num_std_12mts": 2,
            "num_sub": 0, "num_sub_6mts": 0, "num_sub_12mts": 0,
            "num_dbt": 0, "num_dbt_6mts": 0, "num_dbt_12mts": 0,
            "num_lss": 0, "num_lss_6mts": 0, "num_lss_12mts": 0,
            "tot_enq": 5, "CC_enq": 2, "CC_enq_L6m": 1, "CC_enq_L12m": 2,
            "PL_enq": 2, "PL_enq_L6m": 0, "PL_enq_L12m": 1,
            "time_since_recent_enq": 2.0,
            "enq_L12m": 3, "enq_L6m": 1, "enq_L3m": 0,
            "CC_Flag": 1, "PL_Flag": 1, "HL_Flag": 0, "GL_Flag": 0,
            "pct_of_active_TLs_ever": 66.7, "pct_opened_TLs_L6m_of_L12m": 50.0,
            "pct_currentBal_all_TL": 45.0,
            "pct_PL_enq_L6m_of_L12m": 0.0, "pct_CC_enq_L6m_of_L12m": 50.0,
            "pct_PL_enq_L6m_of_ever": 0.0, "pct_CC_enq_L6m_of_ever": 50.0,
            "max_unsec_exposure_inPct": 60.0,
            "last_prod_enq2": "PL", "first_prod_enq2": "CC",
        }

    def test_valid_payload_parses(self):
        from pydantic import ValidationError
        from src.api.schemas import PredictionRequest
        payload = self._base_payload()
        model = PredictionRequest(**payload)   # should not raise
        assert model.AGE == 34

    def test_invalid_gender_raises(self):
        from pydantic import ValidationError
        from src.api.schemas import PredictionRequest
        payload = {**self._base_payload(), "GENDER": "X"}
        with pytest.raises(ValidationError):
            PredictionRequest(**payload)

    def test_age_below_minimum_raises(self):
        from pydantic import ValidationError
        from src.api.schemas import PredictionRequest
        payload = {**self._base_payload(), "AGE": 10}
        with pytest.raises(ValidationError):
            PredictionRequest(**payload)

    def test_negative_income_raises(self):
        from pydantic import ValidationError
        from src.api.schemas import PredictionRequest
        payload = {**self._base_payload(), "NETMONTHLYINCOME": -1000}
        with pytest.raises(ValidationError):
            PredictionRequest(**payload)

    def test_invalid_education_raises(self):
        from pydantic import ValidationError
        from src.api.schemas import PredictionRequest
        payload = {**self._base_payload(), "EDUCATION": "PHD"}
        with pytest.raises(ValidationError):
            PredictionRequest(**payload)

    def test_invalid_product_type_raises(self):
        from pydantic import ValidationError
        from src.api.schemas import PredictionRequest
        payload = {**self._base_payload(), "last_prod_enq2": "MORTGAGE"}
        with pytest.raises(ValidationError):
            PredictionRequest(**payload)

    def test_none_optional_fields_accepted(self):
        from pydantic import ValidationError
        from src.api.schemas import PredictionRequest
        payload = {**self._base_payload(),
                   "time_since_recent_payment": None,
                   "time_since_first_deliquency": None,
                   "time_since_recent_deliquency": None,
                   "time_since_recent_enq": None}
        model = PredictionRequest(**payload)   # should not raise
        assert model.time_since_recent_payment is None


class TestPreprocessInput:
    def _raw_input(self) -> dict:
        from src.api.schemas import PredictionRequest
        return {
            "AGE": 34, "GENDER": "M", "MARITALSTATUS": "Married",
            "EDUCATION": "GRADUATE", "NETMONTHLYINCOME": 45000,
            "Time_With_Curr_Empr": 36,
            "Total_TL": 12, "Tot_Closed_TL": 4, "Tot_Active_TL": 8,
            "Total_TL_opened_L6M": 1, "Tot_TL_closed_L6M": 0,
            "pct_tl_open_L6M": 8.3, "pct_tl_closed_L6M": 0.0,
            "pct_active_tl": 66.7, "pct_closed_tl": 33.3,
            "Total_TL_opened_L12M": 2, "Tot_TL_closed_L12M": 1,
            "pct_tl_open_L12M": 16.7, "pct_tl_closed_L12M": 8.3,
            "Tot_Missed_Pmnt": 0, "Age_Oldest_TL": 72, "Age_Newest_TL": 6,
            "Auto_TL": 1, "CC_TL": 2, "Consumer_TL": 1,
            "Gold_TL": 0, "Home_TL": 0, "PL_TL": 1,
            "Secured_TL": 2, "Unsecured_TL": 3, "Other_TL": 1,
            "time_since_recent_payment": 1.0,
            "time_since_first_deliquency": None,
            "time_since_recent_deliquency": None,
            "num_times_delinquent": 0, "max_delinquency_level": 0,
            "max_recent_level_of_deliq": 0,
            "num_deliq_6mts": 0, "num_deliq_12mts": 0, "num_deliq_6_12mts": 0,
            "max_deliq_6mts": 0, "max_deliq_12mts": 0,
            "num_times_30p_dpd": 0, "num_times_60p_dpd": 0, "recent_level_of_deliq": 0,
            "num_std": 10, "num_std_6mts": 1, "num_std_12mts": 2,
            "num_sub": 0, "num_sub_6mts": 0, "num_sub_12mts": 0,
            "num_dbt": 0, "num_dbt_6mts": 0, "num_dbt_12mts": 0,
            "num_lss": 0, "num_lss_6mts": 0, "num_lss_12mts": 0,
            "tot_enq": 5, "CC_enq": 2, "CC_enq_L6m": 1, "CC_enq_L12m": 2,
            "PL_enq": 2, "PL_enq_L6m": 0, "PL_enq_L12m": 1,
            "time_since_recent_enq": 2.0,
            "enq_L12m": 3, "enq_L6m": 1, "enq_L3m": 0,
            "CC_Flag": 1, "PL_Flag": 1, "HL_Flag": 0, "GL_Flag": 0,
            "pct_of_active_TLs_ever": 66.7, "pct_opened_TLs_L6m_of_L12m": 50.0,
            "pct_currentBal_all_TL": 45.0,
            "pct_PL_enq_L6m_of_L12m": 0.0, "pct_CC_enq_L6m_of_L12m": 50.0,
            "pct_PL_enq_L6m_of_ever": 0.0, "pct_CC_enq_L6m_of_ever": 50.0,
            "max_unsec_exposure_inPct": 60.0,
            "last_prod_enq2": "PL", "first_prod_enq2": "CC",
        }

    def test_preprocess_returns_dataframe(self):
        from src.inference.predict import preprocess_input
        df = preprocess_input(self._raw_input())
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_preprocess_gender_encoded(self):
        from src.inference.predict import preprocess_input
        df = preprocess_input(self._raw_input())
        assert df["GENDER"].iloc[0] == 0   # "M" → 0

    def test_preprocess_female_encoded(self):
        from src.inference.predict import preprocess_input
        raw = {**self._raw_input(), "GENDER": "F"}
        df = preprocess_input(raw)
        assert df["GENDER"].iloc[0] == 1   # "F" → 1

    def test_preprocess_no_approved_flag(self):
        from src.inference.predict import preprocess_input
        df = preprocess_input(self._raw_input())
        assert "Approved_Flag" not in df.columns

    def test_preprocess_has_engineered_features(self):
        from src.inference.predict import preprocess_input
        df = preprocess_input(self._raw_input())
        for col in ["delinquency_rate", "credit_history_span", "dpd_severity_score"]:
            assert col in df.columns, f"Missing: {col}"
