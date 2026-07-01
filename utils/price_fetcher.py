import yfinance as yf
import pandas as pd

NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "BAJFINANCE.NS", "NESTLEIND.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "POWERGRID.NS", "NTPC.NS", "TECHM.NS", "HCLTECH.NS", "BAJAJFINSV.NS",
    "ONGC.NS", "JSWSTEEL.NS", "COALINDIA.NS", "GRASIM.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "CIPLA.NS", "BPCL.NS", "HEROMOTOCO.NS",
    "BRITANNIA.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "HINDALCO.NS",
    "TATASTEEL.NS", "UPL.NS", "SBILIFE.NS", "BAJAJ-AUTO.NS", "HDFCLIFE.NS",
    "INDUSINDBK.NS", "M&M.NS", "ADANIPORTS.NS", "TMPV.NS",
    "ADANIENT.NS", "SHREECEM.NS"
]


def fetch_prices(ticker, start="2015-01-01", end="2026-06-30"):
    try:
        df = yf.Ticker(ticker).history(start=start, end=end)

        if df.empty:
            print(f"{ticker} — no data found")
            return []

        df = df.reset_index()

        # moving averages
        df["dma_20"] = df["Close"].rolling(20).mean()
        df["dma_50"] = df["Close"].rolling(50).mean()

        # volume spike
        df["volume_spike"] = df["Volume"] > df["Volume"].rolling(20).mean() * 2

        # RSI
        delta         = df["Close"].diff()
        gain          = delta.clip(lower=0).rolling(14).mean()
        loss          = (-delta.clip(upper=0)).rolling(14).mean()
        df["rsi"]     = 100 - (100 / (1 + gain / loss))

        # MACD
        # MACD = 12 day EMA minus 26 day EMA
        # Signal = 9 day EMA of MACD
        # Histogram = MACD minus Signal
        # When MACD crosses above signal = bullish
        # When MACD crosses below signal = bearish
        ema_12         = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26         = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"]     = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        # Upper band = 20 day mean + 2 standard deviations
        # Lower band = 20 day mean - 2 standard deviations
        # bb_position = where price sits between bands (0=lower, 1=upper)
        rolling_mean      = df["Close"].rolling(20).mean()
        rolling_std       = df["Close"].rolling(20).std()
        df["bb_upper"]    = rolling_mean + (2 * rolling_std)
        df["bb_lower"]    = rolling_mean - (2 * rolling_std)
        df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # price momentum
        df["return_1d"] = df["Close"].pct_change(1) * 100
        df["return_3d"] = df["Close"].pct_change(3) * 100
        df["return_5d"] = df["Close"].pct_change(5) * 100

        # distance from 20 day high and low
        df["pct_from_20d_high"] = (df["Close"] / df["Close"].rolling(20).max() - 1) * 100
        df["pct_from_20d_low"]  = (df["Close"] / df["Close"].rolling(20).min() - 1) * 100

        results = []
        for _, row in df.iterrows():

            def safe(val):
                return round(float(val), 4) if pd.notna(val) else None

            results.append({
                "ticker"          : ticker,
                "date"            : row["Date"].date(),
                "open_price"      : safe(row["Open"]),
                "close_price"     : safe(row["Close"]),
                "volume"          : int(row["Volume"]),
                "dma_20"          : safe(row["dma_20"]),
                "dma_50"          : safe(row["dma_50"]),
                "volume_spike"    : bool(row["volume_spike"]) if pd.notna(row["volume_spike"]) else False,
                "rsi"             : safe(row["rsi"]),
                "macd"            : safe(row["macd"]),
                "macd_signal"     : safe(row["macd_signal"]),
                "macd_hist"       : safe(row["macd_hist"]),
                "bb_position"     : safe(row["bb_position"]),
                "return_1d"       : safe(row["return_1d"]),
                "return_3d"       : safe(row["return_3d"]),
                "return_5d"       : safe(row["return_5d"]),
                "pct_from_20d_high": safe(row["pct_from_20d_high"]),
                "pct_from_20d_low" : safe(row["pct_from_20d_low"]),
            })

        print(f"{ticker} — {len(results)} rows")
        return results

    except Exception as e:
        print(f"{ticker} — ERROR: {e}")
        return []