"""
Credit Risk Predictor — Premium Fintech Dashboard
===================================================
Redesigned UI: modern SaaS fintech aesthetic.
Backend logic, API calls, payload generation unchanged.
"""

import sys
from pathlib import Path

import requests
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="CreditIQ — Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config.settings import FASTAPI_BASE_URL, RISK_DESCRIPTIONS
RISK_CONFIG = RISK_DESCRIPTIONS

# ─── DESIGN TOKENS ────────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#0F172A",
    "card":     "#1E293B",
    "card2":    "#162032",
    "border":   "#334155",
    "blue":     "#3B82F6",
    "blue_dim": "#1D4ED8",
    "green":    "#22C55E",
    "yellow":   "#EAB308",
    "orange":   "#F97316",
    "red":      "#EF4444",
    "text":     "#F8FAFC",
    "muted":    "#94A3B8",
    "subtle":   "#475569",
}

RISK_COLORS = {
    "P1": {"text": "#4ade80", "bg": "#052e16", "border": "#16a34a", "bar": "#22c55e", "glow": "rgba(34,197,94,0.15)"},
    "P2": {"text": "#fde047", "bg": "#1c1a05", "border": "#ca8a04", "bar": "#eab308", "glow": "rgba(234,179,8,0.15)"},
    "P3": {"text": "#fb923c", "bg": "#1a0a00", "border": "#c2410c", "bar": "#f97316", "glow": "rgba(249,115,22,0.15)"},
    "P4": {"text": "#f87171", "bg": "#1c0505", "border": "#b91c1c", "bar": "#ef4444", "glow": "rgba(239,68,68,0.15)"},
}

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [data-testid="stAppViewContainer"], .stApp {{
    background-color: {PALETTE['bg']} !important;
    color: {PALETTE['text']} !important;
    font-family: 'DM Sans', sans-serif !important;
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
[data-testid="stDecoration"] {{ display: none; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0a1628 0%, #0f1e35 100%) !important;
    border-right: 1px solid {PALETTE['border']} !important;
}}
[data-testid="stSidebar"] * {{ color: {PALETTE['text']} !important; }}

/* ── Main content padding ── */
.block-container {{
    padding: 0 2rem 4rem 2rem !important;
    max-width: 1400px !important;
}}

/* ── Hero section ── */
.hero-section {{
    background: linear-gradient(135deg, #0F172A 0%, #162032 40%, #1a1f35 100%);
    border: 1px solid {PALETTE['border']};
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}}
.hero-section::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
}}
.hero-section::after {{
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(34,197,94,0.07) 0%, transparent 70%);
    border-radius: 50%;
}}
.hero-title {{
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0 0 0.4rem 0;
    background: linear-gradient(135deg, {PALETTE['text']} 0%, #93C5FD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.hero-sub {{
    font-size: 1rem;
    color: {PALETTE['muted']};
    margin: 0 0 1.5rem 0;
    line-height: 1.6;
    max-width: 560px;
}}
.badge-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}}
.badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.28rem 0.75rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.3px;
    border: 1px solid;
}}
.badge-blue  {{ color: #93C5FD; background: rgba(59,130,246,0.12); border-color: rgba(59,130,246,0.3); }}
.badge-green {{ color: #86EFAC; background: rgba(34,197,94,0.10); border-color: rgba(34,197,94,0.3); }}
.badge-amber {{ color: #FDE68A; background: rgba(234,179,8,0.10); border-color: rgba(234,179,8,0.3); }}
.badge-purple{{ color: #C4B5FD; background: rgba(139,92,246,0.10); border-color: rgba(139,92,246,0.3); }}

/* ── KPI Cards ── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}}
.kpi-card {{
    background: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 14px;
    padding: 1.3rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    border-color: {PALETTE['blue']};
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 14px 14px 0 0;
}}
.kpi-blue::before   {{ background: linear-gradient(90deg, {PALETTE['blue']}, transparent); }}
.kpi-green::before  {{ background: linear-gradient(90deg, {PALETTE['green']}, transparent); }}
.kpi-yellow::before {{ background: linear-gradient(90deg, {PALETTE['yellow']}, transparent); }}
.kpi-orange::before {{ background: linear-gradient(90deg, {PALETTE['orange']}, transparent); }}
.kpi-icon  {{ font-size: 1.4rem; margin-bottom: 0.6rem; }}
.kpi-value {{
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.25rem;
}}
.kpi-label {{
    font-size: 0.72rem;
    color: {PALETTE['muted']};
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
.kpi-blue   .kpi-value {{ color: #93C5FD; }}
.kpi-green  .kpi-value {{ color: #86EFAC; }}
.kpi-yellow .kpi-value {{ color: #FDE68A; }}
.kpi-orange .kpi-value {{ color: #FED7AA; }}

/* ── Section titles ── */
.sec-title {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: {PALETTE['subtle']};
    border-left: 3px solid {PALETTE['blue']};
    padding-left: 0.6rem;
    margin: 0 0 1rem 0;
}}

/* ── Tab styling ── */
[data-testid="stTabs"] [data-testid="stTab"] {{
    background: {PALETTE['card2']} !important;
    border: 1px solid {PALETTE['border']} !important;
    border-radius: 8px !important;
    color: {PALETTE['muted']} !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    margin-right: 0.3rem !important;
    transition: all 0.2s ease !important;
}}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(59,130,246,0.05)) !important;
    border-color: {PALETTE['blue']} !important;
    color: {PALETTE['text']} !important;
}}
[data-testid="stTabsContent"] {{
    background: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 0 12px 12px 12px;
    padding: 1.5rem;
}}

/* ── Inputs ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div,
[data-testid="stSlider"] {{
    background: {PALETTE['card2']} !important;
    border-color: {PALETTE['border']} !important;
    color: {PALETTE['text']} !important;
    border-radius: 8px !important;
}}
.stSlider [data-testid="stThumbValue"] {{
    background: {PALETTE['blue']} !important;
    color: white !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
}}
.stSlider [role="slider"] {{
    background: {PALETTE['blue']} !important;
    border: 2px solid #93C5FD !important;
}}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label {{
    font-size: 0.82rem !important;
    color: {PALETTE['muted']} !important;
}}
[data-testid="stCheckbox"] input:checked + div {{
    background: {PALETTE['blue']} !important;
    border-color: {PALETTE['blue']} !important;
}}

/* ── Toggle ── */
[data-testid="stToggle"] {{
    background: {PALETTE['card2']} !important;
}}

/* ── Predict button ── */
.stButton > button {{
    background: linear-gradient(135deg, {PALETTE['blue']} 0%, {PALETTE['blue_dim']} 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.3) !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    background: linear-gradient(135deg, #60A5FA 0%, {PALETTE['blue']} 100%) !important;
    box-shadow: 0 6px 28px rgba(59,130,246,0.45) !important;
    transform: translateY(-1px) !important;
}}

/* ── Result card ── */
.result-wrapper {{
    background: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
}}

/* ── Signal chips ── */
.signal-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-top: 1.5rem;
}}
.signal-chip {{
    background: {PALETTE['card2']};
    border: 1px solid {PALETTE['border']};
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}}
.signal-chip .s-val {{
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.2rem;
}}
.signal-chip .s-lbl {{
    font-size: 0.68rem;
    color: {PALETTE['muted']};
    text-transform: uppercase;
    letter-spacing: 1px;
}}
.signal-safe .s-val {{ color: #86EFAC; }}
.signal-warn .s-val {{ color: #FCA5A5; }}

/* ── SHAP cards ── */
.shap-row {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.65rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
    background: {PALETTE['card2']};
    border: 1px solid {PALETTE['border']};
}}
.shap-rank {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: {PALETTE['subtle']};
    min-width: 20px;
}}
.shap-name {{
    flex: 1;
    font-size: 0.8rem;
    color: {PALETTE['text']};
    font-weight: 500;
}}
.shap-bar-outer {{
    width: 120px;
    height: 5px;
    background: {PALETTE['border']};
    border-radius: 99px;
    overflow: hidden;
}}
.shap-bar-inner {{
    height: 100%;
    border-radius: 99px;
}}
.shap-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    min-width: 55px;
    text-align: right;
}}

/* ── Divider ── */
hr {{
    border: none !important;
    border-top: 1px solid {PALETTE['border']} !important;
    margin: 1.5rem 0 !important;
}}

/* ── Spinner ── */
[data-testid="stSpinner"] {{ color: {PALETTE['blue']} !important; }}

/* ── Sidebar components ── */
.sb-status {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 0.75rem;
    background: rgba(255,255,255,0.04);
    border-radius: 8px;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
    border: 1px solid {PALETTE['border']};
}}
.sb-dot-green {{ color: #4ade80; font-size: 0.7rem; }}
.sb-dot-yellow {{ color: #fde047; font-size: 0.7rem; }}
.sb-dot-red    {{ color: #f87171; font-size: 0.7rem; }}

/* ── Responsive ── */
@media (max-width: 768px) {{
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .signal-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .hero-title {{ font-size: 1.8rem; }}
    .block-container {{ padding: 0 0.75rem 4rem 0.75rem !important; }}
}}
</style>
""", unsafe_allow_html=True)


# ─── API HELPERS (unchanged) ──────────────────────────────────────────────────
def call_api(endpoint: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{FASTAPI_BASE_URL}{endpoint}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def call_api_directly(raw_input: dict, explain: bool = False):
    if explain:
        from src.inference.predict import explain_prediction
        return explain_prediction(raw_input, top_n=10)
    from src.inference.predict import predict
    label, confidence, all_probs = predict(raw_input)
    from config.settings import RISK_DESCRIPTIONS
    return {
        "predicted_class": label,
        "confidence": confidence,
        "all_probs": all_probs,
        "risk_info": RISK_DESCRIPTIONS[label],
    }


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.25rem;">
  <span style="font-size:1.4rem">🏦</span>
  <span style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.05rem;
               background:linear-gradient(135deg,#F8FAFC,#93C5FD);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    CreditIQ
  </span>
</div>
<div style="font-size:0.7rem;color:#64748b;letter-spacing:1px;text-transform:uppercase;
            margin-bottom:1.2rem;padding-left:2rem;">
  Risk Intelligence Platform
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#475569;margin-bottom:0.6rem;">System Status</p>', unsafe_allow_html=True)

    api_ok = model_ok = False
    metrics = {}
    try:
        health = requests.get(f"{FASTAPI_BASE_URL}/health", timeout=3).json()
        api_ok = health.get("status") == "ok"
        model_ok = health.get("model_loaded", False)
        m = requests.get(f"{FASTAPI_BASE_URL}/metrics", timeout=3).json()
        metrics = m
    except Exception:
        pass

    st.markdown(f"""
<div class="sb-status">
  <span class="{'sb-dot-green' if api_ok else 'sb-dot-red'}">●</span>
  <span style="color:#94a3b8;font-size:0.8rem;">API</span>
  <span style="margin-left:auto;font-size:0.75rem;font-family:'JetBrains Mono',monospace;
               color:{'#4ade80' if api_ok else '#f87171'}">
    {'ONLINE' if api_ok else 'OFFLINE'}
  </span>
</div>
<div class="sb-status">
  <span class="{'sb-dot-green' if model_ok else 'sb-dot-yellow'}">●</span>
  <span style="color:#94a3b8;font-size:0.8rem;">XGBoost Model</span>
  <span style="margin-left:auto;font-size:0.75rem;font-family:'JetBrains Mono',monospace;
               color:{'#4ade80' if model_ok else '#fde047'}">
    {'LOADED' if model_ok else 'STANDBY'}
  </span>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#475569;margin-bottom:0.8rem;">Quick Metrics</p>', unsafe_allow_html=True)

    acc  = f"{metrics.get('test_accuracy', 0.77):.1%}" if metrics else "77.0%"
    f1   = f"{metrics.get('macro_f1', 0.0):.3f}"       if metrics else "—"
    rec  = f"{metrics.get('p3_recall', 0.0):.3f}"      if metrics else "—"

    for label_txt, val, clr in [("Accuracy", acc, "#93C5FD"), ("Macro F1", f1, "#86EFAC"), ("P3 Recall", rec, "#FCA5A5")]:
        st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:0.45rem 0;border-bottom:1px solid #1e293b;">
  <span style="font-size:0.78rem;color:#64748b;">{label_txt}</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;
               font-weight:600;color:{clr};">{val}</span>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#475569;margin-bottom:0.8rem;">Risk Classes</p>', unsafe_allow_html=True)
    for cls, lbl, clr in [("P1","Very Low","#4ade80"),("P2","Low","#fde047"),("P3","High","#fb923c"),("P4","Very High","#f87171")]:
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.5rem;padding:0.3rem 0;">
  <span style="color:{clr};font-size:0.65rem;">■</span>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
               font-weight:600;color:{clr};">{cls}</span>
  <span style="font-size:0.75rem;color:#64748b;">— {lbl} Risk</span>
</div>
""", unsafe_allow_html=True)

    st.divider()
    show_explanation = st.toggle("🔬 SHAP Explanation", value=False)
    st.markdown('<p style="font-size:0.68rem;color:#475569;margin-top:0.3rem;">Enable feature-level AI explainability</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown("""
<div style="font-size:0.7rem;color:#334155;line-height:1.7;">
  <div>⚡ XGBoost · Class-weighted</div>
  <div>📊 60+ credit bureau features</div>
  <div>🔍 SHAP explainability layer</div>
  <div>🚀 FastAPI · Streamlit stack</div>
</div>
""", unsafe_allow_html=True)


# ─── HERO SECTION ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
    <div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#3B82F6;
                  letter-spacing:3px;text-transform:uppercase;margin-bottom:0.5rem;">
        AI · Credit Intelligence
      </div>
      <h1 class="hero-title">Credit Risk Predictor</h1>
      <p class="hero-sub">
        Real-time credit risk classification powered by XGBoost with SHAP explainability.
        Classifies applicants into four risk bands with calibrated confidence scores.
      </p>
      <div class="badge-row">
        <span class="badge badge-blue">⚡ XGBoost</span>
        <span class="badge badge-green">🔍 SHAP Explainer</span>
        <span class="badge badge-purple">🚀 FastAPI Backend</span>
        <span class="badge badge-amber">🎯 77% Accuracy</span>
        <span class="badge badge-blue">📊 60+ Features</span>
        <span class="badge badge-green">4 Risk Classes</span>
      </div>
    </div>
    <div style="text-align:right;flex-shrink:0;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#334155;margin-bottom:0.4rem;">MODEL STATUS</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#4ade80;">● LIVE</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI CARDS ────────────────────────────────────────────────────────────────
acc_val = f"{metrics.get('test_accuracy', 0.77):.1%}" if metrics else "77.0%"
f1_val  = f"{metrics.get('macro_f1', 0.0):.3f}"       if metrics else "—"
rec_val = f"{metrics.get('p3_recall', 0.0):.3f}"      if metrics else "—"

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card kpi-blue">
    <div class="kpi-icon">🎯</div>
    <div class="kpi-value">{acc_val}</div>
    <div class="kpi-label">Test Accuracy</div>
  </div>
  <div class="kpi-card kpi-green">
    <div class="kpi-icon">📐</div>
    <div class="kpi-value">{f1_val}</div>
    <div class="kpi-label">Macro F1 Score</div>
  </div>
  <div class="kpi-card kpi-yellow">
    <div class="kpi-icon">⚠️</div>
    <div class="kpi-value">{rec_val}</div>
    <div class="kpi-label">P3 Recall</div>
  </div>
  <div class="kpi-card kpi-orange">
    <div class="kpi-icon">🧩</div>
    <div class="kpi-value">60+</div>
    <div class="kpi-label">Input Features</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── INPUT SECTION ────────────────────────────────────────────────────────────
st.markdown('<p class="sec-title">Customer Profile & Financial Data</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤  Demographics",
    "💳  Credit History",
    "⚠️  Delinquencies",
    "🔍  Enquiries",
    "🏷️  Products",
])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        age      = st.number_input("Age (years)", 21, 70, 34)
        income   = st.number_input("Net Monthly Income (₹)", 5000, 500000, 45000, step=1000,
                                   help="Take-home salary after tax deductions")
    with c2:
        gender   = st.selectbox("Gender", ["M", "F"])
        marital  = st.selectbox("Marital Status", ["Married", "Single"])
    with c3:
        education = st.selectbox("Education Level", [
            "SSC", "12TH", "UNDER GRADUATE", "GRADUATE",
            "POST-GRADUATE", "PROFESSIONAL", "OTHERS"
        ])
        employer  = st.number_input("Months with Current Employer", 0, 120, 36,
                                    help="Employment stability indicator")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p style="font-size:0.75rem;color:#64748b;font-weight:600;margin-bottom:0.5rem;">Account Counts</p>', unsafe_allow_html=True)
        total_tl       = st.slider("Total Credit Accounts (ever)", 0, 50, 12)
        tot_active     = st.slider("Active Accounts", 0, 30, 8)
        tot_closed     = st.slider("Closed Accounts", 0, 30, 4)
    with c2:
        st.markdown('<p style="font-size:0.75rem;color:#64748b;font-weight:600;margin-bottom:0.5rem;">Account Age & Activity</p>', unsafe_allow_html=True)
        age_oldest     = st.number_input("Age of Oldest Account (months)", 0, 300, 72)
        age_newest     = st.number_input("Age of Newest Account (months)", 0, 120, 6)
        tl_opened_l6m  = st.number_input("Accounts Opened — Last 6M", 0, 10, 1)
        tl_opened_l12m = st.number_input("Accounts Opened — Last 12M", 0, 20, 2)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p style="font-size:0.75rem;color:#64748b;font-weight:600;margin-bottom:0.5rem;">Missed Payments</p>', unsafe_allow_html=True)
        tot_missed      = st.number_input("Total Missed Payments", 0, 30, 0)
        time_since_pay  = st.number_input("Months Since Last Payment", 0, 48, 1)
        num_deliq       = st.number_input("Total Times Delinquent", 0, 20, 0)
    with c2:
        st.markdown('<p style="font-size:0.75rem;color:#64748b;font-weight:600;margin-bottom:0.5rem;">Days Past Due</p>', unsafe_allow_html=True)
        num_30dpd       = st.number_input("Times 30+ Days Past Due", 0, 15, 0)
        num_60dpd       = st.number_input("Times 60+ Days Past Due", 0, 10, 0)
        max_deliq_level = st.selectbox("Max Delinquency Level", [0, 1, 2, 3, 4, 5],
                                       help="0 = none, 5 = loss/write-off")

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p style="font-size:0.75rem;color:#64748b;font-weight:600;margin-bottom:0.5rem;">Total Enquiries</p>', unsafe_allow_html=True)
        tot_enq  = st.number_input("Total Enquiries (ever)", 0, 30, 5)
        enq_l12m = st.number_input("Enquiries — Last 12M", 0, 20, 3)
    with c2:
        st.markdown('<p style="font-size:0.75rem;color:#64748b;font-weight:600;margin-bottom:0.5rem;">Recent Enquiries</p>', unsafe_allow_html=True)
        enq_l6m  = st.number_input("Enquiries — Last 6M", 0, 15, 1)
        enq_l3m  = st.number_input("Enquiries — Last 3M", 0, 10, 0)
        enq_since = st.number_input("Months Since Last Enquiry", 0, 36, 2)

with tab5:
    st.markdown('<p style="font-size:0.75rem;color:#64748b;font-weight:600;margin-bottom:0.8rem;">Active Credit Products</p>', unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns(4)
    cc_flag = pc1.checkbox("💳 Credit Card",   value=True)
    pl_flag = pc2.checkbox("💰 Personal Loan", value=True)
    hl_flag = pc3.checkbox("🏠 Home Loan",     value=False)
    gl_flag = pc4.checkbox("🥇 Gold Loan",     value=False)

    st.markdown('<p style="font-size:0.75rem;color:#64748b;font-weight:600;margin:1rem 0 0.5rem 0;">Enquiry History</p>', unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)
    with ec1:
        last_prod  = st.selectbox("Most Recent Product Enquired",
            ["PL", "CC", "ConsumerLoan", "HL", "AL", "others"])
    with ec2:
        first_prod = st.selectbox("First Product Enquired",
            ["CC", "PL", "ConsumerLoan", "HL", "AL", "others"])


# ─── PAYLOAD BUILDER (unchanged logic) ────────────────────────────────────────
def build_payload() -> dict:
    return {
        "Total_TL": total_tl, "Tot_Closed_TL": tot_closed, "Tot_Active_TL": tot_active,
        "Total_TL_opened_L6M": tl_opened_l6m, "Tot_TL_closed_L6M": 0,
        "pct_tl_open_L6M":   (tl_opened_l6m  / (total_tl + 1)) * 100,
        "pct_tl_closed_L6M": 0.0,
        "pct_active_tl":     (tot_active / (total_tl + 1)) * 100,
        "pct_closed_tl":     (tot_closed / (total_tl + 1)) * 100,
        "Total_TL_opened_L12M": tl_opened_l12m, "Tot_TL_closed_L12M": 0,
        "pct_tl_open_L12M":  (tl_opened_l12m / (total_tl + 1)) * 100,
        "pct_tl_closed_L12M": 0.0,
        "Tot_Missed_Pmnt": tot_missed,
        "Auto_TL": 1, "CC_TL": int(cc_flag), "Consumer_TL": 0,
        "Gold_TL": int(gl_flag), "Home_TL": int(hl_flag), "PL_TL": int(pl_flag),
        "Secured_TL": int(hl_flag) + int(gl_flag),
        "Unsecured_TL": int(cc_flag) + int(pl_flag), "Other_TL": 1,
        "Age_Oldest_TL": age_oldest, "Age_Newest_TL": age_newest,
        "time_since_recent_payment": float(time_since_pay),
        "time_since_first_deliquency":  None if num_deliq == 0 else 24.0,
        "time_since_recent_deliquency": None if num_deliq == 0 else 6.0,
        "num_times_delinquent": num_deliq, "max_delinquency_level": max_deliq_level,
        "max_recent_level_of_deliq": max_deliq_level,
        "num_deliq_6mts": 0, "num_deliq_12mts": num_deliq, "num_deliq_6_12mts": 0,
        "max_deliq_6mts": 0, "max_deliq_12mts": max_deliq_level,
        "num_times_30p_dpd": num_30dpd, "num_times_60p_dpd": num_60dpd,
        "num_std": max(0, total_tl - num_deliq), "num_std_6mts": 1, "num_std_12mts": 2,
        "num_sub": 0, "num_sub_6mts": 0, "num_sub_12mts": 0,
        "num_dbt": 0, "num_dbt_6mts": 0, "num_dbt_12mts": 0,
        "num_lss": 0, "num_lss_6mts": 0, "num_lss_12mts": 0,
        "recent_level_of_deliq": max_deliq_level,
        "tot_enq": float(tot_enq), "CC_enq": 2.0, "CC_enq_L6m": 1.0, "CC_enq_L12m": 2.0,
        "PL_enq": 2.0, "PL_enq_L6m": 0.0, "PL_enq_L12m": 1.0,
        "time_since_recent_enq": float(enq_since),
        "enq_L12m": float(enq_l12m), "enq_L6m": float(enq_l6m), "enq_L3m": float(enq_l3m),
        "MARITALSTATUS": marital, "EDUCATION": education,
        "AGE": age, "GENDER": gender, "NETMONTHLYINCOME": float(income),
        "Time_With_Curr_Empr": employer,
        "pct_of_active_TLs_ever":     (tot_active / (total_tl + 1)) * 100,
        "pct_opened_TLs_L6m_of_L12m": (tl_opened_l6m / (tl_opened_l12m + 1)) * 100,
        "pct_currentBal_all_TL": 45.0,
        "CC_Flag": int(cc_flag), "PL_Flag": int(pl_flag),
        "HL_Flag": int(hl_flag), "GL_Flag": int(gl_flag),
        "pct_PL_enq_L6m_of_L12m": 0.0, "pct_CC_enq_L6m_of_L12m": 50.0,
        "pct_PL_enq_L6m_of_ever": 0.0, "pct_CC_enq_L6m_of_ever": 50.0,
        "max_unsec_exposure_inPct": 60.0,
        "last_prod_enq2": last_prod, "first_prod_enq2": first_prod,
    }


# ─── PREDICT BUTTON ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button("🔍  Run Credit Risk Assessment")

# ─── RESULTS ──────────────────────────────────────────────────────────────────
if predict_btn:
    payload  = build_payload()
    endpoint = "/predict/explain" if show_explanation else "/predict"

    with st.spinner("Analysing credit profile…"):
        result = call_api(endpoint, payload)
        if result is None:
            try:
                result = call_api_directly(payload, explain=show_explanation)
            except FileNotFoundError:
                st.error("Model not found. Please run: `python -m src.training.train_pipeline`")
                st.stop()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.exception(e)
                st.stop()

    if result:
        label      = result["predicted_class"]
        confidence = result["confidence"]
        all_probs  = result["all_probs"]
        cfg        = RISK_CONFIG[label]
        clr        = RISK_COLORS.get(label, RISK_COLORS["P1"])
        conf_pct   = int(confidence * 100)

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<p class="sec-title">Assessment Result</p>', unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 1.4], gap="large")

        # ── Risk Classification Card ──────────────────────────────────────
        with res_col1:
            st.markdown(f"""
<div style="
    background: linear-gradient(145deg, {clr['bg']}, #0F172A);
    border: 1.5px solid {clr['border']};
    border-radius: 18px;
    padding: 2.5rem 2rem;
    text-align: center;
    box-shadow: 0 8px 40px {clr['glow']};
    position: relative;
    overflow: hidden;
">
  <!-- Glow orb -->
  <div style="
    position:absolute;top:-40px;right:-40px;width:150px;height:150px;
    background:radial-gradient(circle,{clr['glow']} 0%,transparent 70%);
    border-radius:50%;
  "></div>

  <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
              letter-spacing:3px;color:{clr['text']};opacity:0.6;
              text-transform:uppercase;margin-bottom:0.8rem;">
    Risk Classification
  </div>

  <div style="font-family:'Syne',sans-serif;font-size:5.5rem;font-weight:800;
              letter-spacing:6px;line-height:1;color:{clr['text']};
              margin-bottom:0.5rem;">
    {label}
  </div>

  <div style="font-size:1rem;font-weight:600;color:{clr['text']};
              margin-bottom:0.5rem;">
    {cfg['label']}
  </div>

  <div style="font-size:0.8rem;color:#94a3b8;line-height:1.6;
              margin-bottom:1.8rem;max-width:240px;margin-left:auto;margin-right:auto;">
    {cfg['description']}
  </div>

  <!-- Confidence gauge -->
  <div style="margin-bottom:1.2rem;">
    <div style="display:flex;justify-content:space-between;align-items:center;
                margin-bottom:0.4rem;">
      <span style="font-size:0.65rem;font-family:'JetBrains Mono',monospace;
                   color:#475569;letter-spacing:2px;text-transform:uppercase;">
        Confidence
      </span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                   font-weight:600;color:{clr['text']};">
        {conf_pct}%
      </span>
    </div>
    <div style="background:#1e293b;border-radius:99px;height:6px;overflow:hidden;">
      <div style="width:{conf_pct}%;height:100%;
                  background:linear-gradient(90deg,{clr['text']},{clr['bar']});
                  border-radius:99px;
                  box-shadow:0 0 8px {clr['glow']};"></div>
    </div>
  </div>

  <!-- Action tag -->
  <div style="display:inline-block;border:1px solid {clr['border']};border-radius:99px;
              padding:0.5rem 1.2rem;font-size:0.73rem;font-family:'JetBrains Mono',monospace;
              letter-spacing:0.5px;color:{clr['text']};
              background:rgba(255,255,255,0.03);">
    {cfg['action']}
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Probability Chart ─────────────────────────────────────────────
        with res_col2:
            classes   = ["P1", "P2", "P3", "P4"]
            xlabels   = ["P1 · Very Low", "P2 · Low", "P3 · High", "P4 · Very High"]
            probs     = [all_probs.get(k, 0) for k in classes]
            bar_clrs  = [RISK_COLORS[k]["bar"] for k in classes]
            opacities = [1.0 if k == label else 0.3 for k in classes]
            line_clrs = [RISK_COLORS[k]["border"] for k in classes]

            fig = go.Figure()
            for i, (x, y, bc, op, lc) in enumerate(zip(xlabels, probs, bar_clrs, opacities, line_clrs)):
                fig.add_trace(go.Bar(
                    x=[x], y=[y],
                    marker=dict(color=bc, opacity=op, line=dict(color=lc, width=1.5)),
                    text=[f"{y:.1%}"],
                    textposition="outside",
                    textfont=dict(
                        color="#f1f5f9" if classes[i] == label else "#475569",
                        size=12,
                        family="JetBrains Mono"
                    ),
                    hovertemplate=f"<b>{x}</b><br>Prob: {y:.2%}<extra></extra>",
                    showlegend=False,
                ))

            # Highlight selected
            sel_idx = classes.index(label)
            fig.add_shape(
                type="rect",
                x0=sel_idx - 0.4, x1=sel_idx + 0.4,
                y0=0, y1=probs[sel_idx] * 1.05,
                line=dict(color=RISK_COLORS[label]["border"], width=2, dash="dot"),
                fillcolor="rgba(0,0,0,0)",
                layer="above",
            )

            fig.update_layout(
                title=dict(
                    text="Probability Distribution Across Risk Classes",
                    font=dict(color="#64748b", size=12, family="DM Sans"),
                    x=0,
                ),
                yaxis=dict(
                    range=[0, 1.25],
                    tickformat=".0%",
                    gridcolor="#1e293b",
                    gridwidth=1,
                    color="#475569",
                    tickfont=dict(size=10, color="#475569", family="JetBrains Mono"),
                    zeroline=False,
                ),
                xaxis=dict(
                    tickfont=dict(size=11, color="#94a3b8", family="DM Sans"),
                    tickangle=0,
                ),
                plot_bgcolor="#161b27",
                paper_bgcolor="#1E293B",
                font=dict(color="#f1f5f9", family="DM Sans"),
                margin=dict(t=50, b=30, l=55, r=20),
                showlegend=False,
                bargap=0.45,
                height=370,
                bargroupgap=0.1,
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Key Risk Signal Chips ─────────────────────────────────────────
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<p class="sec-title">Key Risk Signals</p>', unsafe_allow_html=True)

        def risk_class(val, threshold=0):
            return "signal-warn" if val > threshold else "signal-safe"

        st.markdown(f"""
<div class="signal-grid">
  <div class="signal-chip {risk_class(num_deliq)}">
    <div class="s-val">{num_deliq}</div>
    <div class="s-lbl">Delinquencies</div>
  </div>
  <div class="signal-chip {risk_class(tot_missed)}">
    <div class="s-val">{tot_missed}</div>
    <div class="s-lbl">Missed Payments</div>
  </div>
  <div class="signal-chip {risk_class(enq_l6m, 2)}">
    <div class="s-val">{enq_l6m}</div>
    <div class="s-lbl">Enquiries (L6M)</div>
  </div>
  <div class="signal-chip {risk_class(num_60dpd)}">
    <div class="s-val">{num_60dpd}</div>
    <div class="s-lbl">60+ DPD Events</div>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── SHAP Explanation ──────────────────────────────────────────────
        if show_explanation and "top_features" in result:
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<p class="sec-title">SHAP Feature Attribution</p>', unsafe_allow_html=True)

            features   = result["top_features"]
            feat_names = [f["feature"]    for f in features]
            shap_vals  = [f["shap_value"] for f in features]
            directions = [f["direction"]  for f in features]

            # Interactive Plotly chart
            bar_colors = ["#ef4444" if d == "increases_risk" else "#22c55e" for d in directions]
            max_abs = max(abs(v) for v in shap_vals) if shap_vals else 1

            # Ranked cards list
            shap_html = ""
            for i, (fn, sv, d) in enumerate(zip(feat_names, shap_vals, directions)):
                clr2   = "#ef4444" if d == "increases_risk" else "#22c55e"
                bg2    = "rgba(239,68,68,0.08)" if d == "increases_risk" else "rgba(34,197,94,0.08)"
                pct    = min(abs(sv) / max_abs * 100, 100)
                icon   = "↑" if d == "increases_risk" else "↓"
                shap_html += f"""
<div class="shap-row" style="border-left:3px solid {clr2};">
  <span class="shap-rank">#{i+1:02d}</span>
  <span class="shap-name">{fn}</span>
  <div class="shap-bar-outer">
    <div class="shap-bar-inner" style="width:{pct}%;background:{clr2};"></div>
  </div>
  <span class="shap-val" style="color:{clr2};">{icon} {sv:+.3f}</span>
</div>
"""
            st.markdown(shap_html, unsafe_allow_html=True)

            # Interactive chart
            st.markdown('<br>', unsafe_allow_html=True)
            fig2 = go.Figure(go.Bar(
                x=shap_vals[::-1],
                y=feat_names[::-1],
                orientation="h",
                marker=dict(
                    color=bar_colors[::-1],
                    line=dict(color=bar_colors[::-1], width=0.5),
                    opacity=0.85,
                ),
                text=[f"{v:+.3f}" for v in shap_vals[::-1]],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=10, family="JetBrains Mono"),
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
            ))
            fig2.update_layout(
                title=dict(
                    text=f"Feature Impact on Predicted Class: {label}",
                    font=dict(color="#64748b", size=12, family="DM Sans"),
                    x=0,
                ),
                xaxis=dict(
                    title="← Decreases Risk  |  SHAP Value  |  Increases Risk →",
                    title_font=dict(size=10, color="#475569", family="DM Sans"),
                    gridcolor="#1e293b", color="#64748b",
                    zeroline=True, zerolinecolor="#475569", zerolinewidth=1.5,
                    tickfont=dict(size=9, color="#64748b", family="JetBrains Mono"),
                ),
                yaxis=dict(
                    tickfont=dict(size=10, color="#94a3b8", family="DM Sans"),
                    automargin=True,
                ),
                plot_bgcolor="#161b27",
                paper_bgcolor="#1E293B",
                font=dict(color="#f1f5f9"),
                margin=dict(t=50, b=40, l=20, r=80),
                height=420,
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Legend
            st.markdown("""
<div style="display:flex;gap:1.5rem;margin-top:0.5rem;">
  <div style="display:flex;align-items:center;gap:0.4rem;">
    <div style="width:12px;height:12px;background:#ef4444;border-radius:2px;"></div>
    <span style="font-size:0.75rem;color:#94a3b8;">Increases credit risk</span>
  </div>
  <div style="display:flex;align-items:center;gap:0.4rem;">
    <div style="width:12px;height:12px;background:#22c55e;border-radius:2px;"></div>
    <span style="font-size:0.75rem;color:#94a3b8;">Reduces credit risk</span>
  </div>
</div>
""", unsafe_allow_html=True)