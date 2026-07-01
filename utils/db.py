import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        host     = os.getenv("DB_HOST"),
        database = os.getenv("DB_NAME"),
        user     = os.getenv("DB_USER"),
        password = os.getenv("DB_PASSWORD"),
        port     = os.getenv("DB_PORT")
    )


def insert_prices(prices):
    conn = get_connection()
    cur  = conn.cursor()

    for p in prices:
        cur.execute("""
            INSERT INTO stock_prices
                (ticker, date, open_price, close_price, volume,
                 dma_20, dma_50, volume_spike, rsi,
                 macd, macd_signal, macd_hist, bb_position,
                 return_1d, return_3d, return_5d,
                 pct_from_20d_high, pct_from_20d_low)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ticker, date) DO UPDATE SET
                macd              = EXCLUDED.macd,
                macd_signal       = EXCLUDED.macd_signal,
                macd_hist         = EXCLUDED.macd_hist,
                bb_position       = EXCLUDED.bb_position,
                return_1d         = EXCLUDED.return_1d,
                return_3d         = EXCLUDED.return_3d,
                return_5d         = EXCLUDED.return_5d,
                pct_from_20d_high = EXCLUDED.pct_from_20d_high,
                pct_from_20d_low  = EXCLUDED.pct_from_20d_low
        """, (
            p["ticker"], p["date"],
            p["open_price"], p["close_price"], p["volume"],
            p["dma_20"], p["dma_50"],
            p["volume_spike"], p["rsi"],
            p["macd"], p["macd_signal"], p["macd_hist"], p["bb_position"],
            p["return_1d"], p["return_3d"], p["return_5d"],
            p["pct_from_20d_high"], p["pct_from_20d_low"]
        ))

    conn.commit()
    cur.close()
    conn.close()


def insert_news(articles):
    conn = get_connection()
    cur  = conn.cursor()

    for a in articles:
        cur.execute("""
            INSERT INTO news_articles
                (ticker, content, source, published_at, url)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            a["ticker"], a["content"],
            a["source"], a["published_at"], a["url"]
        ))

    conn.commit()
    cur.close()
    conn.close()