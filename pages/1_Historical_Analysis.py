import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app_utils.db import (get_all_tickers, get_stock_prices,
                           get_daily_sentiment, get_articles_with_returns)
from app_utils.styling import apply_theme
import plotly.graph_objects as go
import pandas as pd
import joblib

st.set_page_config(page_title="Historical Analysis", page_icon="📈", layout="wide")
apply_theme()

st.title("📈 Historical Analysis")
st.caption("Past news sentiment vs price movement for any Nifty 50 stock")

# ── Stock selector ──────────────────────────────────────────────────
tickers = get_all_tickers()
ticker  = st.selectbox("Select stock", tickers,
                        index=tickers.index("RELIANCE.NS") if "RELIANCE.NS" in tickers else 0)

days = st.slider("Days of history", 30, 730, 180, step=30)

prices    = get_stock_prices(ticker, days=days)
sentiment = get_daily_sentiment(ticker, days=days)

if len(prices) == 0:
    st.warning("No data available for this stock.")
    st.stop()

# ── Price chart ─────────────────────────────────────────────────────
st.subheader("Price Movement")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=prices['date'], y=prices['close_price'],
    mode='lines', name='Close Price',
    line=dict(color='#3B9EFF', width=2)
))
if 'dma_20' in prices.columns:
    fig.add_trace(go.Scatter(
        x=prices['date'], y=prices['dma_20'],
        mode='lines', name='DMA 20',
        line=dict(color='#FFA500', width=1, dash='dot')
    ))
if 'dma_50' in prices.columns:
    fig.add_trace(go.Scatter(
        x=prices['date'], y=prices['dma_50'],
        mode='lines', name='DMA 50',
        line=dict(color='#FF4D4D', width=1, dash='dot')
    ))
fig.update_layout(template="plotly_dark", height=350,
                  margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# ── Sentiment LINE chart ────────────────────────────────────────────
st.subheader("Daily Sentiment Trend")

if len(sentiment) > 0:
    fig2 = go.Figure()

    # fill area under the line — green above 0, red below 0
    fig2.add_trace(go.Scatter(
        x    = sentiment['date'],
        y    = sentiment['avg_compound'],
        mode = 'lines',
        name = 'Sentiment',
        line = dict(color='#00D26A', width=2),
        fill = 'tozeroy',
        fillcolor = 'rgba(0,210,106,0.15)'
    ))

    fig2.add_hline(y=0, line_color='white', line_width=0.8,
                   line_dash='dash')

    fig2.update_layout(
        template = "plotly_dark",
        height   = 220,
        margin   = dict(l=20, r=20, t=20, b=20),
        yaxis    = dict(title="Compound Score (-1 to +1)"),
        hovermode = "x unified"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Above 0 = positive sentiment period   Below 0 = negative sentiment period")
else:
    st.info("No sentiment data for this period.")

st.divider()

# ── XGBoost accuracy for this stock — 1D AND 5D ─────────────────────
st.subheader(f"XGBoost Model Accuracy — {ticker}")

path_1d = "models/per_stock_1d_results.pkl"
path_5d = "models/per_stock_5d_results.pkl"

col1, col2 = st.columns(2)

with col1:
    st.markdown("**1 Day Direction Accuracy**")
    if os.path.exists(path_1d):
        ps_1d = joblib.load(path_1d)
        if ticker in ps_1d:
            acc_1d  = ps_1d[ticker]['accuracy'] * 100
            rows_1d = ps_1d[ticker]['rows']
            st.metric("Strong-Move 1D Accuracy", f"{acc_1d:.1f}%",
                      delta=f"{acc_1d-50:.1f}% vs baseline")
            st.caption(f"Trained on {rows_1d} significant price move days (>1%)")
            if acc_1d > 60:
                st.success("Strong signal for this stock.")
            elif acc_1d > 55:
                st.warning("Moderate signal above baseline.")
            else:
                st.error("Weak signal — near random.")
        else:
            st.info("No per-stock 1D model. Global model used.")
    else:
        st.info("Per-stock 1D results not found.")

with col2:
    st.markdown("**5 Day Direction Accuracy**")
    if os.path.exists(path_5d):
        ps_5d = joblib.load(path_5d)
        if ticker in ps_5d:
            acc_5d  = ps_5d[ticker]['accuracy'] * 100
            rows_5d = ps_5d[ticker]['rows']
            st.metric("Strong-Move 5D Accuracy", f"{acc_5d:.1f}%",
                      delta=f"{acc_5d-50:.1f}% vs baseline")
            st.caption(f"Trained on {rows_5d} significant price move days (>1.5%)")
            if acc_5d > 60:
                st.success("Strong signal for this stock.")
            elif acc_5d > 55:
                st.warning("Moderate signal above baseline.")
            else:
                st.error("Weak signal — near random.")
        else:
            st.info("No per-stock 5D model. Global model used.")
    else:
        st.info("Per-stock 5D results not found.")

st.divider()

# ── Historical articles with returns ────────────────────────────────
st.subheader(f"Historical Articles — {ticker}")
st.caption("Each article shown with its sentiment score and how the stock moved after")

articles = get_articles_with_returns(ticker)

if len(articles) > 0:
    col1, col2 = st.columns(2)
    with col1:
        filter_label = st.selectbox(
            "Filter by sentiment",
            ["All", "positive", "negative", "neutral"]
        )
    with col2:
        search_term = st.text_input("Search", placeholder="e.g. profit, NPA")

    filtered = articles.copy()
    if filter_label != "All":
        filtered = filtered[filtered['sentiment_label'] == filter_label]
    if search_term:
        filtered = filtered[
            filtered['content'].str.lower().str.contains(
                search_term.lower(), na=False
            )
        ]

    st.caption(f"Showing {min(50, len(filtered)):,} of {len(filtered):,} articles")

    for _, row in filtered.head(50).iterrows():
        date_str = row['published_at'].strftime('%d %b %Y') \
                   if pd.notna(row['published_at']) else "Unknown"

        label    = row['sentiment_label'] or 'neutral'
        icon     = "🟢" if label == 'positive' else "🔴" if label == 'negative' else "⚪"
        compound = row['sentiment_compound'] or 0

        ret_1d = f"{row['return_1d']:+.2f}%" if pd.notna(row.get('return_1d')) else "N/A"
        ret_5d = f"{row['return_5d']:+.2f}%" if pd.notna(row.get('return_5d')) else "N/A"

        with st.expander(f"{icon} {date_str} — {row['content'][:80]}..."):
            st.write(row['content'])

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Sentiment", label.upper())
            with c2:
                st.metric("Compound", f"{compound:+.3f}")
            with c3:
                st.metric("Next Day Return", ret_1d)
            with c4:
                st.metric("5 Day Return", ret_5d)

            url = row.get('url', '')
            if url and str(url) not in ['nan', 'None', '']:
                st.markdown(f"[Read article →]({url})")
            else:
                st.caption("URL: Not available")
else:
    st.info("No articles found for this stock.")