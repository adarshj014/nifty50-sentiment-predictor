import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

def get_connection():
    try:
        import streamlit as st
        return psycopg2.connect(
            host     = st.secrets["DB_HOST"],
            port     = st.secrets["DB_PORT"],
            database = st.secrets["DB_NAME"],
            user     = st.secrets["DB_USER"],
            password = st.secrets["DB_PASSWORD"],
            sslmode  = "require"
        )
    except Exception:
        load_dotenv()
        return psycopg2.connect(
            host     = os.getenv("DB_HOST"),
            port     = os.getenv("DB_PORT"),
            database = os.getenv("DB_NAME"),
            user     = os.getenv("DB_USER"),
            password = os.getenv("DB_PASSWORD"),
            sslmode  = "require"
        )

@st.cache_data(ttl=3600)
def get_all_tickers():
    conn = get_connection()
    df = pd.read_sql("SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker", conn)
    conn.close()
    return df['ticker'].tolist()

@st.cache_data(ttl=3600)
def get_stock_prices(ticker, days=180):
    conn = get_connection()
    df = pd.read_sql(f"""
        SELECT * FROM stock_prices
        WHERE ticker = '{ticker}'
        ORDER BY date DESC
        LIMIT {days}
    """, conn)
    conn.close()
    return df.sort_values('date').reset_index(drop=True)

@st.cache_data(ttl=3600)
def get_daily_sentiment(ticker, days=180):
    conn = get_connection()
    df = pd.read_sql(f"""
        SELECT * FROM daily_sentiment
        WHERE ticker = '{ticker}'
        ORDER BY date DESC
        LIMIT {days}
    """, conn)
    conn.close()
    return df.sort_values('date').reset_index(drop=True)

@st.cache_data(ttl=3600)
def get_recent_news(ticker, limit=10):
    conn = get_connection()
    df = pd.read_sql(f"""
        SELECT ticker, content, source, published_at,
               sentiment_label, sentiment_compound
        FROM news_articles
        WHERE ticker = '{ticker}'
        ORDER BY published_at DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    return df
@st.cache_data(ttl=3600)
def get_articles_with_returns(ticker):
    """
    Gets all news articles for a ticker and calculates:
    - return_1d: % change from news day close to next trading day close
    - return_5d: % change from news day close to 5th trading day close

    Logic:
    - p_base = price on the same day as news (or nearest trading day after)
    - p_next = price on the next available trading day after news
    - p_5d   = price on the 5th available trading day after news
    """
    conn = get_connection()
    df   = pd.read_sql(f"""
        WITH article_dates AS (
            SELECT
                n.id,
                n.published_at,
                n.content,
                n.sentiment_label,
                n.sentiment_compound,
                n.source,
                n.url,
                DATE(n.published_at) AS news_date
            FROM news_articles n
            WHERE n.ticker = '{ticker}'
              AND n.sentiment_label IS NOT NULL
        ),

        -- base price: closest trading day on or after news date
        base_prices AS (
            SELECT
                a.id,
                MIN(sp.date) AS base_date
            FROM article_dates a
            JOIN stock_prices sp
                ON sp.ticker = '{ticker}'
                AND sp.date >= a.news_date
            GROUP BY a.id
        ),

        -- next trading day price: first trading day strictly after base date
        next_prices AS (
            SELECT
                bp.id,
                MIN(sp.date) AS next_date
            FROM base_prices bp
            JOIN stock_prices sp
                ON sp.ticker = '{ticker}'
                AND sp.date > bp.base_date
            GROUP BY bp.id
        ),

        -- 5th trading day price: 5th trading day after base date
        fifth_prices AS (
            SELECT
                bp.id,
                (
                    SELECT sp2.date
                    FROM stock_prices sp2
                    WHERE sp2.ticker = '{ticker}'
                      AND sp2.date > bp.base_date
                    ORDER BY sp2.date
                    LIMIT 1 OFFSET 4
                ) AS fifth_date
            FROM base_prices bp
        )

        SELECT
            a.published_at,
            a.content,
            a.sentiment_label,
            a.sentiment_compound,
            a.source,
            a.url,
            p_base.close_price                                          AS price_on_day,
            p_next.close_price                                          AS price_next_day,
            p_5d.close_price                                            AS price_5day,
            ROUND(
                ((p_next.close_price - p_base.close_price)
                / NULLIF(p_base.close_price, 0) * 100)::numeric, 2
            )                                                           AS return_1d,
            ROUND(
                ((p_5d.close_price - p_base.close_price)
                / NULLIF(p_base.close_price, 0) * 100)::numeric, 2
            )                                                           AS return_5d
        FROM article_dates a
        JOIN base_prices  bp   ON a.id = bp.id
        JOIN next_prices  np   ON a.id = np.id
        JOIN fifth_prices fp   ON a.id = fp.id
        LEFT JOIN stock_prices p_base ON p_base.ticker = '{ticker}' AND p_base.date = bp.base_date
        LEFT JOIN stock_prices p_next ON p_next.ticker = '{ticker}' AND p_next.date = np.next_date
        LEFT JOIN stock_prices p_5d   ON p_5d.ticker   = '{ticker}' AND p_5d.date   = fp.fifth_date
        ORDER BY a.published_at DESC
    """, conn)
    conn.close()
    return df