"""
Pydantic schemas for FastAPI request and response validation.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ─── Enums / Literals ─────────────────────────────────────────────────────────

GenderType = Literal["M", "F"]
MaritalType = Literal["Married", "Single"]
EducationType = Literal[
    "SSC",
    "12TH",
    "UNDER GRADUATE",
    "GRADUATE",
    "POST-GRADUATE",
    "PROFESSIONAL",
    "OTHERS",
]
ProductType = Literal["PL", "CC", "ConsumerLoan", "HL", "AL", "others"]
RiskClassType = Literal["P1", "P2", "P3", "P4"]


# ─── Prediction Request ───────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    """All customer features required for a credit risk prediction."""

    # Demographics
    AGE: int = Field(..., ge=18, le=100, description="Customer age in years")
    GENDER: GenderType = Field(..., description="M or F")
    MARITALSTATUS: MaritalType = Field(..., description="Married or Single")
    EDUCATION: EducationType = Field(..., description="Highest education level")
    NETMONTHLYINCOME: float = Field(
        ..., ge=0, le=10_000_000, description="Net monthly income in INR"
    )
    Time_With_Curr_Empr: int = Field(
        ..., ge=0, le=600, description="Months with current employer"
    )

    # Credit trade lines
    Total_TL: int = Field(..., ge=0)
    Tot_Closed_TL: int = Field(..., ge=0)
    Tot_Active_TL: int = Field(..., ge=0)
    Total_TL_opened_L6M: int = Field(..., ge=0)
    Tot_TL_closed_L6M: int = Field(0, ge=0)
    pct_tl_open_L6M: float = Field(0.0, ge=0.0, le=100.0)
    pct_tl_closed_L6M: float = Field(0.0, ge=0.0, le=100.0)
    pct_active_tl: float = Field(0.0, ge=0.0, le=100.0)
    pct_closed_tl: float = Field(0.0, ge=0.0, le=100.0)
    Total_TL_opened_L12M: int = Field(..., ge=0)
    Tot_TL_closed_L12M: int = Field(0, ge=0)
    pct_tl_open_L12M: float = Field(0.0, ge=0.0, le=100.0)
    pct_tl_closed_L12M: float = Field(0.0, ge=0.0, le=100.0)
    Tot_Missed_Pmnt: int = Field(..., ge=0)
    Age_Oldest_TL: int = Field(..., ge=0, le=600)
    Age_Newest_TL: int = Field(..., ge=0, le=600)

    # Trade line type counts
    Auto_TL: int = Field(0, ge=0)
    CC_TL: int = Field(0, ge=0)
    Consumer_TL: int = Field(0, ge=0)
    Gold_TL: int = Field(0, ge=0)
    Home_TL: int = Field(0, ge=0)
    PL_TL: int = Field(0, ge=0)
    Secured_TL: int = Field(0, ge=0)
    Unsecured_TL: int = Field(0, ge=0)
    Other_TL: int = Field(0, ge=0)

    # Payment history
    time_since_recent_payment: Optional[float] = Field(None, ge=0)
    time_since_first_deliquency: Optional[float] = Field(None, ge=0)
    time_since_recent_deliquency: Optional[float] = Field(None, ge=0)
    num_times_delinquent: int = Field(0, ge=0)
    max_delinquency_level: int = Field(0, ge=0)
    max_recent_level_of_deliq: int = Field(0, ge=0)
    num_deliq_6mts: int = Field(0, ge=0)
    num_deliq_12mts: int = Field(0, ge=0)
    num_deliq_6_12mts: int = Field(0, ge=0)
    max_deliq_6mts: int = Field(0, ge=0)
    max_deliq_12mts: int = Field(0, ge=0)
    num_times_30p_dpd: int = Field(0, ge=0)
    num_times_60p_dpd: int = Field(0, ge=0)
    recent_level_of_deliq: int = Field(0, ge=0)

    # Account status counts
    num_std: int = Field(0, ge=0)
    num_std_6mts: int = Field(0, ge=0)
    num_std_12mts: int = Field(0, ge=0)
    num_sub: int = Field(0, ge=0)
    num_sub_6mts: int = Field(0, ge=0)
    num_sub_12mts: int = Field(0, ge=0)
    num_dbt: int = Field(0, ge=0)
    num_dbt_6mts: int = Field(0, ge=0)
    num_dbt_12mts: int = Field(0, ge=0)
    num_lss: int = Field(0, ge=0)
    num_lss_6mts: int = Field(0, ge=0)
    num_lss_12mts: int = Field(0, ge=0)

    # Enquiries
    tot_enq: float = Field(0.0, ge=0)
    CC_enq: float = Field(0.0, ge=0)
    CC_enq_L6m: float = Field(0.0, ge=0)
    CC_enq_L12m: float = Field(0.0, ge=0)
    PL_enq: float = Field(0.0, ge=0)
    PL_enq_L6m: float = Field(0.0, ge=0)
    PL_enq_L12m: float = Field(0.0, ge=0)
    time_since_recent_enq: Optional[float] = Field(None, ge=0)
    enq_L12m: float = Field(0.0, ge=0)
    enq_L6m: float = Field(0.0, ge=0)
    enq_L3m: float = Field(0.0, ge=0)

    # Flags
    CC_Flag: int = Field(0, ge=0, le=1)
    PL_Flag: int = Field(0, ge=0, le=1)
    HL_Flag: int = Field(0, ge=0, le=1)
    GL_Flag: int = Field(0, ge=0, le=1)

    # Percentages
    pct_of_active_TLs_ever: float = Field(0.0, ge=0.0, le=100.0)
    pct_opened_TLs_L6m_of_L12m: float = Field(0.0, ge=0.0, le=100.0)
    pct_currentBal_all_TL: float = Field(0.0, ge=0.0, le=100.0)
    pct_PL_enq_L6m_of_L12m: float = Field(0.0, ge=0.0, le=100.0)
    pct_CC_enq_L6m_of_L12m: float = Field(0.0, ge=0.0, le=100.0)
    pct_PL_enq_L6m_of_ever: float = Field(0.0, ge=0.0, le=100.0)
    pct_CC_enq_L6m_of_ever: float = Field(0.0, ge=0.0, le=100.0)
    max_unsec_exposure_inPct: float = Field(0.0, ge=0.0, le=100.0)

    # Product enquiries
    last_prod_enq2: ProductType = Field(
        ..., description="Most recent product enquiry type"
    )
    first_prod_enq2: ProductType = Field(
        ..., description="First ever product enquiry type"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


# ─── Response Models ─────────────────────────────────────────────────────────


class RiskInfo(BaseModel):
    label: str
    action: str
    color: str
    description: str


class PredictionResponse(BaseModel):
    predicted_class: RiskClassType
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_probs: Dict[RiskClassType, float]
    risk_info: RiskInfo


class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["increases_risk", "decreases_risk"]


class ExplainResponse(BaseModel):
    predicted_class: RiskClassType
    confidence: float
    all_probs: Dict[RiskClassType, float]
    top_features: List[ShapFeature]
    risk_info: RiskInfo


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    test_accuracy: float
    macro_f1: float
    weighted_f1: float
    p3_recall: float
    cv_mean_f1: float
    cv_std_f1: float
    mlflow_run_id: Optional[str] = None
