"""
Unit tests for feature engineering and preprocessing functions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _sample_row() -> dict:
    return {
        "Total_TL": 12,
        "Tot_Closed_TL": 4,
        "Tot_Active_TL": 8,
        "Total_TL_opened_L6M": 2,
        "Tot_TL_closed_L6M": 0,
        "pct_tl_open_L6M": 16.7,
        "pct_tl_closed_L6M": 0.0,
        "pct_active_tl": 66.7,
        "pct_closed_tl": 33.3,
        "Total_TL_opened_L12M": 3,
        "Tot_TL_closed_L12M": 1,
        "pct_tl_open_L12M": 25.0,
        "pct_tl_closed_L12M": 8.3,
        "Tot_Missed_Pmnt": 0,
        "Auto_TL": 1,
        "CC_TL": 2,
        "Consumer_TL": 1,
        "Gold_TL": 0,
        "Home_TL": 0,
        "PL_TL": 1,
        "Secured_TL": 2,
        "Unsecured_TL": 3,
        "Other_TL": 1,
        "Age_Oldest_TL": 72,
        "Age_Newest_TL": 6,
        "time_since_recent_payment": 1.0,
        "time_since_first_deliquency": np.nan,
        "time_since_recent_deliquency": np.nan,
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
        "MARITALSTATUS": "Married",
        "EDUCATION": "GRADUATE",
        "AGE": 34,
        "GENDER": "M",
        "NETMONTHLYINCOME": 45000,
        "Time_With_Curr_Empr": 36,
        "pct_of_active_TLs_ever": 66.7,
        "pct_opened_TLs_L6m_of_L12m": 50.0,
        "pct_currentBal_all_TL": 45.0,
        "CC_Flag": 1,
        "PL_Flag": 1,
        "HL_Flag": 0,
        "GL_Flag": 0,
        "pct_PL_enq_L6m_of_L12m": 0.0,
        "pct_CC_enq_L6m_of_L12m": 50.0,
        "pct_PL_enq_L6m_of_ever": 0.0,
        "pct_CC_enq_L6m_of_ever": 50.0,
        "max_unsec_exposure_inPct": 60.0,
        "last_prod_enq2": "PL",
        "first_prod_enq2": "CC",
    }


class TestEngineerFeatures:
    def test_delinquency_rate_no_delinquency(self):
        from src.training.train_pipeline import engineer_features

        df = pd.DataFrame(
            [{**_sample_row(), "num_times_delinquent": 0, "Total_TL": 10}]
        )
        out = engineer_features(df)
        assert out["delinquency_rate"].iloc[0] == pytest.approx(0.0 / 11, abs=1e-6)

    def test_delinquency_rate_with_delinquency(self):
        from src.training.train_pipeline import engineer_features

        df = pd.DataFrame(
            [{**_sample_row(), "num_times_delinquent": 3, "Total_TL": 10}]
        )
        out = engineer_features(df)
        assert out["delinquency_rate"].iloc[0] == pytest.approx(3 / 11, abs=1e-6)

    def test_credit_history_span(self):
        from src.training.train_pipeline import engineer_features

        df = pd.DataFrame([{**_sample_row(), "Age_Oldest_TL": 72, "Age_Newest_TL": 12}])
        out = engineer_features(df)
        assert out["credit_history_span"].iloc[0] == 60

    def test_dpd_severity_score(self):
        from src.training.train_pipeline import engineer_features

        df = pd.DataFrame(
            [{**_sample_row(), "num_times_30p_dpd": 2, "num_times_60p_dpd": 1}]
        )
        out = engineer_features(df)
        assert out["dpd_severity_score"].iloc[0] == 4  # 2 + 2*1

    def test_has_delinquency_history_false_when_nan(self):
        from src.training.train_pipeline import engineer_features

        df = pd.DataFrame([{**_sample_row(), "time_since_recent_deliquency": np.nan}])
        out = engineer_features(df)
        assert out["has_delinquency_history"].iloc[0] == 0

    def test_has_delinquency_history_true_when_present(self):
        from src.training.train_pipeline import engineer_features

        df = pd.DataFrame([{**_sample_row(), "time_since_recent_deliquency": 6.0}])
        out = engineer_features(df)
        assert out["has_delinquency_history"].iloc[0] == 1

    def test_enquiry_acceleration_no_history(self):
        from src.training.train_pipeline import engineer_features

        df = pd.DataFrame([{**_sample_row(), "enq_L3m": 0, "enq_L12m": 0}])
        out = engineer_features(df)
        assert out["enquiry_acceleration"].iloc[0] == pytest.approx(0.0 / 1, abs=1e-6)

    def test_engineered_columns_present(self):
        from src.training.train_pipeline import engineer_features

        expected = [
            "delinquency_rate",
            "recent_activity_ratio",
            "credit_history_span",
            "dpd_severity_score",
            "enquiry_acceleration",
            "has_delinquency_history",
            "has_payment_history",
            "has_enquiry_history",
            "has_credit_history",
        ]
        df = pd.DataFrame([_sample_row()])
        out = engineer_features(df)
        for col in expected:
            assert col in out.columns, f"Missing column: {col}"


class TestPrepareXY:
    def test_target_encoding(self):
        from src.training.train_pipeline import engineer_features, prepare_xy

        rows = []
        for flag in ["P1", "P2", "P3", "P4"]:
            rows.append({**_sample_row(), "Approved_Flag": flag})
        df = engineer_features(pd.DataFrame(rows))
        X, y = prepare_xy(df)
        assert list(y) == [0, 1, 2, 3]

    def test_drop_columns_removed(self):
        from src.training.train_pipeline import engineer_features, prepare_xy

        row = {
            **_sample_row(),
            "Approved_Flag": "P1",
            "Credit_Score": 700,
            "PROSPECTID": 999,
        }
        df = engineer_features(pd.DataFrame([row]))
        X, _ = prepare_xy(df)
        for col in ["Approved_Flag", "Credit_Score", "PROSPECTID"]:
            assert col not in X.columns

    def test_gender_encoding(self):
        from src.training.train_pipeline import engineer_features, prepare_xy

        rows = [
            {**_sample_row(), "Approved_Flag": "P1", "GENDER": "M"},
            {**_sample_row(), "Approved_Flag": "P1", "GENDER": "F"},
        ]
        df = engineer_features(pd.DataFrame(rows))
        X, _ = prepare_xy(df)
        assert X["GENDER"].iloc[0] == 0
        assert X["GENDER"].iloc[1] == 1

    def test_no_nan_in_enquiry_cols(self):
        from src.training.train_pipeline import (
            ENQ_COUNT_COLS,
            engineer_features,
            prepare_xy,
        )

        row = {
            **_sample_row(),
            "Approved_Flag": "P1",
            "tot_enq": np.nan,
            "enq_L3m": np.nan,
        }
        df = engineer_features(pd.DataFrame([row]))
        X, _ = prepare_xy(df)
        for col in ENQ_COUNT_COLS:
            if col in X.columns:
                assert X[col].isna().sum() == 0, f"NaN in {col}"


class TestBuildPipeline:
    def test_pipeline_has_steps(self):
        from sklearn.linear_model import LogisticRegression
        from src.training.train_pipeline import build_pipeline

        pipe = build_pipeline(LogisticRegression())
        assert "preprocessor" in pipe.named_steps
        assert "model" in pipe.named_steps

    def test_pipeline_column_transformer(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.compose import ColumnTransformer
        from src.training.train_pipeline import build_pipeline

        pipe = build_pipeline(LogisticRegression())
        assert isinstance(pipe.named_steps["preprocessor"], ColumnTransformer)
