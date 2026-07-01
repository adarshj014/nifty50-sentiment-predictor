import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app_utils.db import get_all_tickers, get_connection
from app_utils.predictor import load_models, score_headline
from app_utils.styling import apply_theme
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="Live Prediction", page_icon="🔮", layout="wide")
apply_theme()

st.title("🔮 Live Prediction")
st.caption("Enter a headline to get a real-time sentiment-driven prediction using live stock data")

st.markdown("""
<div class="disclaimer-box">
⚠️ <b>Risk Disclaimer:</b> This tool is for educational and research purposes only.
It is not financial advice. 1-day prediction accuracy is approximately 51-53%.
5-day prediction accuracy is approximately 56-58%.
Do not use this for actual trading decisions.
</div>
""", unsafe_allow_html=True)

st.divider()


def fetch_live_features(ticker):
    """
    Fetches real-time stock data from yfinance and calculates
    all technical features needed by the XGBoost model.
    Returns a dict of feature values.
    """
    try:
        stock = yf.Ticker(ticker)

        # download last 60 days to calculate indicators properly
        df = stock.history(period="60d")

        if df.empty or len(df) < 20:
            return None

        df = df.reset_index()

        # DMA
        df['dma_20'] = df['Close'].rolling(20).mean()
        df['dma_50'] = df['Close'].rolling(50).mean()

        # RSI
        delta        = df['Close'].diff()
        gain         = delta.clip(lower=0).rolling(14).mean()
        loss         = (-delta.clip(upper=0)).rolling(14).mean()
        df['rsi']    = 100 - (100 / (1 + gain / loss))

        # MACD
        ema_12            = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26            = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd']        = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist']   = df['macd'] - df['macd_signal']

        # Bollinger Bands
        roll_mean       = df['Close'].rolling(20).mean()
        roll_std        = df['Close'].rolling(20).std()
        bb_upper        = roll_mean + (2 * roll_std)
        bb_lower        = roll_mean - (2 * roll_std)
        df['bb_position'] = ((df['Close'] - bb_lower) / (bb_upper - bb_lower)).clip(0, 1)

        # price momentum
        df['return_1d'] = df['Close'].pct_change(1) * 100
        df['return_3d'] = df['Close'].pct_change(3) * 100
        df['return_5d'] = df['Close'].pct_change(5) * 100

        # distance from 20d high and low
        df['pct_from_20d_high'] = (df['Close'] / df['Close'].rolling(20).max() - 1) * 100
        df['pct_from_20d_low']  = (df['Close'] / df['Close'].rolling(20).min() - 1) * 100

        # volume spike
        df['volume_spike'] = (df['Volume'] > df['Volume'].rolling(20).mean() * 2).astype(int)

        # get latest row
        latest = df.dropna().iloc[-1]

        return {
            'close_price'       : latest['Close'],
            'dma_20'            : latest['dma_20'],
            'dma_50'            : latest['dma_50'],
            'rsi'               : latest['rsi'],
            'macd'              : latest['macd'],
            'macd_signal'       : latest['macd_signal'],
            'macd_hist'         : latest['macd_hist'],
            'bb_position'       : latest['bb_position'],
            'return_1d'         : latest['return_1d'],
            'return_3d'         : latest['return_3d'],
            'return_5d'         : latest['return_5d'],
            'pct_from_20d_high' : latest['pct_from_20d_high'],
            'pct_from_20d_low'  : latest['pct_from_20d_low'],
            'volume_spike'      : latest['volume_spike'],
        }

    except Exception as e:
        st.error(f"Error fetching live data: {e}")
        return None


# ── Input section ───────────────────────────────────────────────────
tickers = get_all_tickers()
ticker  = st.selectbox("Select stock", tickers,
                        index=tickers.index("RELIANCE.NS") if "RELIANCE.NS" in tickers else 0)

headline = st.text_area(
    "Enter headline or paste article",
    placeholder="e.g. Reliance Q4 profit rises 20% beating analyst estimates",
    height=120
)

st.markdown("**Quick Examples:**")
ex_col1, ex_col2, ex_col3 = st.columns(3)
with ex_col1:
    if st.button("Reliance profit rises 20%"):
        headline = "Reliance Q4 profit rises 20% beating analyst estimates"
with ex_col2:
    if st.button("TCS announces layoffs"):
        headline = "TCS announces massive layoffs amid global cost cutting"
with ex_col3:
    if st.button("SBI record quarterly profit"):
        headline = "SBI reports record quarterly profit on strong loan recovery"

predict_button = st.button("🔍 Analyze and Predict", type="primary", use_container_width=True)

st.divider()

# ── Prediction logic ────────────────────────────────────────────────
if predict_button and headline:
    with st.spinner("Fetching live stock data from yfinance..."):
        live_data = fetch_live_features(ticker)

    if live_data is None:
        st.error("Could not fetch live data. Check your internet connection.")
        st.stop()

    with st.spinner("Scoring sentiment with fine-tuned FinBERT..."):
        sentiment_result = score_headline(headline)

    with st.spinner("Running XGBoost prediction..."):
        models = load_models()

        # build feature row using live yfinance data + sentiment
        feature_values = {
            # sentiment features — from FinBERT
            'avg_compound_centered' : sentiment_result['compound'],
            'sentiment_3d_centered' : sentiment_result['compound'],
            'sentiment_momentum'    : 0,
            'article_count'         : 1,
            'positive_count'        : 1 if sentiment_result['label'] == 'positive' else 0,
            'negative_count'        : 1 if sentiment_result['label'] == 'negative' else 0,
            'max_compound'          : sentiment_result['compound'],
            'min_compound'          : sentiment_result['compound'],
            'std_compound'          : 0,

            # technical features — from yfinance live data
            'rsi'                   : live_data['rsi'],
            'rsi_oversold'          : 1 if live_data['rsi'] < 30 else 0,
            'rsi_overbought'        : 1 if live_data['rsi'] > 70 else 0,
            'price_above_dma20'     : 1 if live_data['close_price'] > live_data['dma_20'] else 0,
            'price_above_dma50'     : 1 if live_data['close_price'] > live_data['dma_50'] else 0,
            'dma20_above_dma50'     : 1 if live_data['dma_20'] > live_data['dma_50'] else 0,
            'volume_spike'          : live_data['volume_spike'],
            'macd'                  : live_data['macd'],
            'macd_signal'           : live_data['macd_signal'],
            'macd_hist'             : live_data['macd_hist'],
            'macd_bullish'          : 1 if live_data['macd'] > live_data['macd_signal'] else 0,
            'macd_bearish'          : 1 if live_data['macd'] < live_data['macd_signal'] else 0,
            'bb_position'           : live_data['bb_position'],
            'bb_oversold'           : 1 if live_data['bb_position'] < 0.2 else 0,
            'bb_overbought'         : 1 if live_data['bb_position'] > 0.8 else 0,
            'return_1d'             : live_data['return_1d'],
            'return_3d'             : live_data['return_3d'],
            'return_5d'             : live_data['return_5d'],
            'pct_from_20d_high'     : live_data['pct_from_20d_high'],
            'pct_from_20d_low'      : live_data['pct_from_20d_low'],
        }

        # align with exact features model was trained on
        model_features = models['features']
        feature_row    = pd.DataFrame([feature_values])
        for col in model_features:
            if col not in feature_row.columns:
                feature_row[col] = 0
        feature_row = feature_row[model_features]

        # predictions
        dir_1d  = models['clf_1d'].predict(feature_row)[0]
        prob_1d = models['clf_1d'].predict_proba(feature_row)[0]
        pct_1d  = models['reg_1d'].predict(feature_row)[0]

        dir_5d  = models['clf_5d'].predict(feature_row)[0]
        prob_5d = models['clf_5d'].predict_proba(feature_row)[0]
        pct_5d  = models['reg_5d'].predict(feature_row)[0]

    # ── Show live stock snapshot ─────────────────────────────────────
    st.subheader(f"Live Data — {ticker}")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Current Price", f"₹{live_data['close_price']:.2f}")
    with d2:
        rsi_val = live_data['rsi']
        rsi_note = " (Overbought)" if rsi_val > 70 else " (Oversold)" if rsi_val < 30 else ""
        st.metric("RSI", f"{rsi_val:.1f}{rsi_note}")
    with d3:
        st.metric("MACD Signal", "Bullish" if live_data['macd'] > live_data['macd_signal'] else "Bearish")
    with d4:
        st.metric("1D Return", f"{live_data['return_1d']:+.2f}%")

    st.divider()

    # ── Sentiment result ─────────────────────────────────────────────
    st.subheader("Sentiment Analysis")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Sentiment", sentiment_result['label'].upper())
    with s2:
        st.metric("Confidence", f"{sentiment_result['score']*100:.1f}%")
    with s3:
        st.metric("Compound Score", f"{sentiment_result['compound']:+.3f}")

    st.divider()

    # ── Bull/Bear indicator ──────────────────────────────────────────
    st.subheader("Bull / Bear Indicator")
    p1, p2 = st.columns(2)

    with p1:
        st.markdown("**Short Term — 1 Day**")
        label_1d = "BULLISH ↑" if dir_1d == 1 else "BEARISH ↓"
        color_1d = "bullish"   if dir_1d == 1 else "bearish"
        st.markdown(f"<h2 class='{color_1d}'>{label_1d}</h2>", unsafe_allow_html=True)
        st.progress(float(max(prob_1d)))
        st.caption(f"Confidence: {max(prob_1d)*100:.1f}%")
        st.metric("Expected Move", f"{pct_1d:+.2f}%")

    with p2:
        st.markdown("**Long Term — 5 Days**")
        label_5d = "BULLISH ↑" if dir_5d == 1 else "BEARISH ↓"
        color_5d = "bullish"   if dir_5d == 1 else "bearish"
        st.markdown(f"<h2 class='{color_5d}'>{label_5d}</h2>", unsafe_allow_html=True)
        st.progress(float(max(prob_5d)))
        st.caption(f"Confidence: {max(prob_5d)*100:.1f}%")
        st.metric("Expected Move", f"{pct_5d:+.2f}%")

    st.divider()

    # ── Explain prediction ───────────────────────────────────────────
    st.subheader("Explain Prediction")
    try:
        import shap
        import plotly.express as px

        explainer   = shap.TreeExplainer(models['clf_1d'])
        shap_values = explainer.shap_values(feature_row)

        shap_df = pd.DataFrame({
            'feature' : model_features,
            'impact'  : shap_values[0]
        }).sort_values('impact', key=abs, ascending=False).head(10)

        fig = px.bar(
            shap_df, x='impact', y='feature', orientation='h',
            color='impact',
            color_continuous_scale=['#FF4D4D', '#888888', '#00D26A'],
            title="Top factors driving this prediction (green = bullish push, red = bearish push)"
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("SHAP explanation unavailable.")

    st.divider()

    # ── Similar past events ──────────────────────────────────────────
    st.subheader("Similar Past Events with Outcomes")
    conn    = get_connection()
    similar = pd.read_sql(f"""
        SELECT published_at, content, sentiment_label, sentiment_compound
        FROM news_articles
        WHERE ticker = '{ticker}'
          AND sentiment_label = '{sentiment_result['label']}'
        ORDER BY published_at DESC
        LIMIT 5
    """, conn)
    conn.close()

    if len(similar) > 0:
        for _, row in similar.iterrows():
            date_str = row['published_at'].strftime('%d %b %Y') if pd.notna(row['published_at']) else "Unknown"
            icon     = "🟢" if row['sentiment_label'] == 'positive' else "🔴" if row['sentiment_label'] == 'negative' else "⚪"
            st.caption(f"{icon} **{date_str}** — {row['content'][:160]}")
    else:
        st.caption("No similar past events found.")

elif predict_button and not headline:
    st.warning("Please enter a headline first.")