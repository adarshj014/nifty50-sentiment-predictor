import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(__file__))

from app_utils.db import get_all_tickers, get_stock_prices, get_daily_sentiment
from app_utils.styling import apply_theme
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Nifty 50 Sentiment Signal Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

# ---------------- TITLE ----------------
st.title("📊 Nifty 50 Sentiment Signal Generator")
st.markdown("""
A research tool that combines **fine-tuned FinBERT sentiment analysis** on Indian
financial news with **technical indicators** to analyze directional signals for
Nifty 50 stocks.
""")

st.divider()

# ---------------- STOCK SELECTOR ----------------
col1, col2 = st.columns([3, 1])

with col1:
    tickers = get_all_tickers()

    selected_ticker = st.selectbox(
        "Search any Nifty 50 stock",
        options=tickers,
        index=tickers.index("RELIANCE.NS") if "RELIANCE.NS" in tickers else 0
    )

st.divider()

# ---------------- MAIN DASHBOARD ----------------
if selected_ticker:
    # Pull latest 30 records from database
    prices = get_stock_prices(selected_ticker, days=30)
    sentiment = get_daily_sentiment(selected_ticker, days=30)

    if len(prices) > 0:
        latest = prices.iloc[-1]
        prev = prices.iloc[-2] if len(prices) > 1 else latest

        # -------- Get actual date range from database --------
        start_date = prices['date'].min()
        end_date = prices['date'].max()

        # Convert to readable format
        start_date_str = start_date.strftime("%d %b %Y")
        end_date_str = end_date.strftime("%d %b %Y")

        # -------- Disclaimer --------
        st.warning(
            f"⚠️ Disclaimer: This dashboard uses historical data stored in the database. "
            f"The latest available data is updated only till **{end_date_str}**. "
            f"This is NOT real-time market data."
        )

        st.caption(
            f"📅 Current dashboard window: **{start_date_str} → {end_date_str}** "
            f"(latest 30 trading sessions available in database)"
        )

        # -------- Metrics --------
        col1, col2, col3 = st.columns(3)

        with col1:
            change = latest['close_price'] - prev['close_price']
            pct_change = (change / prev['close_price']) * 100

            st.metric(
                "Last Recorded Close",
                f"₹{latest['close_price']:.2f}",
                f"{pct_change:+.2f}%"
            )

        with col2:
            avg_sent = sentiment['avg_compound'].mean() if len(sentiment) > 0 else 0

            if avg_sent > 0.05:
                sent_label = "Bullish"
            elif avg_sent < -0.05:
                sent_label = "Bearish"
            else:
                sent_label = "Neutral"

            st.metric(
                "Avg Sentiment(Shown Period)",
                sent_label,
                f"{avg_sent:+.3f}"
            )

        with col3:
            total_articles = sentiment['article_count'].sum() if len(sentiment) > 0 else 0
            st.metric(
                "News Coverage",
                f"{int(total_articles)} articles"
            )

        # -------- Price Chart --------
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=prices['date'],
            y=prices['close_price'],
            mode='lines',
            name='Close Price',
            line=dict(color='#00D26A', width=2)
        ))

        fig.update_layout(
            title=f"{selected_ticker} ({start_date_str} → {end_date_str})",
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("No stock price data available for selected ticker.")

st.divider()

# ---------------- TOOL INFO ----------------
st.subheader("What this tool does")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    **📰 Sentiment Analysis**
    
    Fine-tuned FinBERT scores Indian financial news with high accuracy
    """)

with c2:
    st.markdown("""
    **📈 Technical Indicators**
    
    MACD, DMA crossovers, volume spikes
    """)

with c3:
    st.markdown("""
    **🤖 ML Predictions**
    
    XGBoost models for 1-day and 5-day directional signals
    """)

with c4:
    st.markdown("""
    **🔍 Explainability**
    
    SHAP values show why a prediction was made
    """)

st.info("👈 Use the sidebar to explore Historical Analysis, Live Prediction, and Model Performance")