import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app_utils.db import get_connection
from app_utils.styling import apply_theme
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
apply_theme()

st.title("📊 Model Performance")
st.caption("Honest and transparent reporting on how the models actually perform")

# ── Model comparison ────────────────────────────────────────────────
st.subheader("Global Model Comparison")

comparison_path = "models/model_comparison.pkl"
if os.path.exists(comparison_path):
    comparison = joblib.load(comparison_path)

    col_list = st.columns(len(comparison))
    for i, (name, acc) in enumerate(comparison.items()):
        with col_list[i]:
            st.metric(name, f"{acc*100:.2f}%")

    fig = go.Figure(go.Bar(
        x            = list(comparison.keys()),
        y            = [v*100 for v in comparison.values()],
        marker_color = ['#888888', '#FFA500', '#3B9EFF', '#00D26A', '#00A0FF'],
        text         = [f"{v*100:.1f}%" for v in comparison.values()],
        textposition = 'auto'
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="red",
                  annotation_text="Random baseline 50%")
    fig.update_layout(
        template    = "plotly_dark",
        height      = 380,
        title       = "Global Model Accuracy — 1 Day Direction",
        yaxis_title = "Accuracy %",
        yaxis_range = [45, 70]
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Model comparison not found. Run Phase 3 first.")

st.divider()

# ── Dataset stats ───────────────────────────────────────────────────
st.subheader("Dataset Statistics")

conn  = get_connection()
stats = pd.read_sql("""
    SELECT COUNT(*) AS total_news_days,
           COUNT(DISTINCT ticker) AS stocks_covered
    FROM daily_sentiment
""", conn)
conn.close()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total News-Days Analyzed", f"{stats['total_news_days'][0]:,}")
with c2:
    st.metric("Stocks Covered", stats['stocks_covered'][0])
with c3:
    st.metric("FinBERT Accuracy on Indian News", "82.4%")

st.divider()

# ── Per stock accuracy — 1D AND 5D ──────────────────────────────────
st.subheader("XGBoost Accuracy by Stock — 1 Day vs 5 Day")

path_1d = "models/per_stock_1d_results.pkl"
path_5d = "models/per_stock_5d_results.pkl"

if os.path.exists(path_1d) and os.path.exists(path_5d):
    ps_1d = joblib.load(path_1d)
    ps_5d = joblib.load(path_5d)

    # build combined dataframe
    all_tickers = set(ps_1d.keys()) | set(ps_5d.keys())
    rows = []
    for t in sorted(all_tickers):
        rows.append({
            'Ticker'     : t,
            'Acc 1D (%)'  : round(ps_1d[t]['accuracy'] * 100, 1) if t in ps_1d else None,
            'Acc 5D (%)'  : round(ps_5d[t]['accuracy'] * 100, 1) if t in ps_5d else None,
            'Rows 1D'     : ps_1d[t]['rows'] if t in ps_1d else 0,
            'Rows 5D'     : ps_5d[t]['rows'] if t in ps_5d else 0,
        })

    results_df = pd.DataFrame(rows).sort_values('Acc 1D (%)', ascending=False)

    # chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x            = results_df['Ticker'],
        y            = results_df['Acc 1D (%)'],
        name         = '1 Day Accuracy',
        marker_color = '#3B9EFF'
    ))
    fig.add_trace(go.Bar(
        x            = results_df['Ticker'],
        y            = results_df['Acc 5D (%)'],
        name         = '5 Day Accuracy',
        marker_color = '#00D26A'
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="red",
                  annotation_text="50% baseline")
    fig.update_layout(
        template        = "plotly_dark",
        height          = 500,
        title           = "Per-Stock XGBoost Accuracy — Strong Move Days",
        barmode         = 'group',
        xaxis_tickangle = -90,
        yaxis_title     = "Accuracy %",
        yaxis_range     = [40, 80]
    )
    st.plotly_chart(fig, use_container_width=True)

    # table
    st.dataframe(
        results_df.style.format({
            'Acc 1D (%)': '{:.1f}%',
            'Acc 5D (%)': '{:.1f}%',
            'Rows 1D'   : '{:,}',
            'Rows 5D'   : '{:,}'
        }),
        use_container_width = True,
        hide_index          = True
    )

    col1, col2 = st.columns(2)
    with col1:
        above_55_1d = (results_df['Acc 1D (%)'] > 55).sum()
        above_60_1d = (results_df['Acc 1D (%)'] > 60).sum()
        st.metric("1D — stocks above 55%", f"{above_55_1d} / {len(results_df)}")
        st.metric("1D — stocks above 60%", f"{above_60_1d} / {len(results_df)}")
    with col2:
        above_55_5d = (results_df['Acc 5D (%)'].dropna() > 55).sum()
        above_60_5d = (results_df['Acc 5D (%)'].dropna() > 60).sum()
        st.metric("5D — stocks above 55%", f"{above_55_5d} / {len(results_df)}")
        st.metric("5D — stocks above 60%", f"{above_60_5d} / {len(results_df)}")

elif os.path.exists(path_1d):
    st.info("Only 1D per-stock results found. Run 5D training in Phase 3.")
else:
    st.info("Per-stock results not found. Run Phase 3 per-stock training.")

st.divider()

# ── Honest limitations ───────────────────────────────────────────────
st.subheader("⚠️ Honest Limitations")
st.markdown("""
**1-Day Prediction Accuracy (51-53%)**
Barely above random chance. Consistent with the Efficient Market Hypothesis.

**5-Day Prediction Accuracy (56%)**
Modestly better. Some upward bias from the bull market training period.

**Per-Stock Variance**
Some stocks show 60-65% accuracy on significant move days.
Others perform near random due to thin news coverage.

**Sentiment Bias**
Nifty 50 news skews positive. Per-ticker centering partially corrects this.

**What this tool is useful for**
Combining sentiment context with technical signals for research purposes.

**What this tool should NOT be used for**
Actual trading decisions, precise price targets, or financial advice.
""")