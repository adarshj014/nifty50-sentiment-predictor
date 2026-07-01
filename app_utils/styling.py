import streamlit as st

def apply_theme():
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
    }

    .metric-card {
        background: linear-gradient(135deg, #1A1F2B 0%, #161B26 100%);
        border: 1px solid #2A2F3A;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 10px;
    }

    .bullish {
        color: #00D26A;
        font-weight: 700;
    }

    .bearish {
        color: #FF4D4D;
        font-weight: 700;
    }

    .disclaimer-box {
        background-color: #1A1410;
        border-left: 4px solid #FFA500;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
    }

    .stMetric {
        background-color: #1A1F2B;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)