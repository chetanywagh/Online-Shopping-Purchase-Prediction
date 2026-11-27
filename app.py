
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ShopSense AI | Commerce Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. DESIGN SYSTEM
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080d18;
    --panel: #101827;
    --panel2: #141f31;
    --border: #26354b;
    --text: #f1f5f9;
    --muted: #91a0b7;
    --accent: #5b8cff;
    --cyan: #42d6cf;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(8,13,24,0.9);
}

[data-testid="stSidebar"] {
    background: #0d1523;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div {
    padding-top: 25px;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 24px;
    font-weight: 800;
    color: white;
    letter-spacing: -1px;
}

.brand span {
    color: #6ea2ff;
}

.sidebar-label {
    color: #70819b;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 28px;
    margin-bottom: 10px;
}

.hero {
    padding: 36px;
    border-radius: 22px;
    background:
      radial-gradient(circle at 85% 15%, rgba(70,123,255,.22), transparent 35%),
      linear-gradient(120deg, #14223a, #101b2d 65%, #14233d);
    border: 1px solid #2a4164;
    margin-bottom: 26px;
}

.eyebrow {
    color: #80aaff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: 36px;
    line-height: 1.2;
    font-weight: 800;
    color: white;
    margin: 12px 0;
}

.hero p {
    color: #aabbd2;
    font-size: 15px;
    max-width: 700px;
    line-height: 1.7;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #f5f7fb;
    margin: 22px 0 14px 0;
}

.section-caption {
    color: var(--muted);
    font-size: 13px;
    margin-top: -10px;
    margin-bottom: 18px;
}

.kpi {
    background: linear-gradient(145deg, #152238, #101827);
    border: 1px solid #293c58;
    border-radius: 16px;
    padding: 19px;
    min-height: 125px;
}

.kpi-label {
    color: #9aabc3;
    font-size: 12px;
    font-weight: 600;
}

.kpi-value {
    color: #ffffff;
    font-family: 'Manrope', sans-serif;
    font-size: 27px;
    font-weight: 800;
    margin-top: 12px;
}

.kpi-note {
    color: #7d91ac;
    font-size: 11px;
    margin-top: 4px;
}

.info-panel {
    background: #111d2e;
    border: 1px solid #293b54;
    border-radius: 16px;
    padding: 22px;
}

.result-panel {
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 18px;
}

.result-buy {
    background: linear-gradient(120deg, #103b35, #142c32);
    border: 1px solid #238b73;
}

.result-no {
    background: linear-gradient(120deg, #362536, #251d2e);
    border: 1px solid #85506d;
}

.result-panel h2 {
    color: white;
    font-family: 'Manrope', sans-serif;
    font-size: 27px;
    font-weight: 800;
}

.result-panel p {
    color: #c0ccdc;
    font-size: 13px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(100deg, #3976ef, #536df2);
    color: white;
    border: 0;
    border-radius: 10px;
    min-height: 46px;
    font-weight: 700;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: linear-gradient(100deg, #285fd1, #415bd7);
    color: white;
    border: 0;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] {
    background-color: #121f32;
}

[data-testid="stMetric"] {
    background: #111d2e;
    border: 1px solid #293b54;
    border-radius: 13px;
    padding: 15px;
}

[data-testid="stMetricLabel"] {
    color: #9aabc3;
}

[data-testid="stMetricValue"] {
    color: white;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #293b54;
    border-radius: 12px;
}

hr {
    border-color: #26354b;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


#
