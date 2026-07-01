-- stock prices table
DROP TABLE IF EXISTS stock_prices;
CREATE TABLE stock_prices (
    id            SERIAL PRIMARY KEY,
    ticker        VARCHAR(20)  NOT NULL,
    date          DATE         NOT NULL,
    open_price    DECIMAL(10,2),
    close_price   DECIMAL(10,2),
    volume        BIGINT,
    dma_20        DECIMAL(10,2),
    dma_50        DECIMAL(10,2),
    volume_spike  BOOLEAN,
    rsi           DECIMAL(5,2),
    UNIQUE(ticker, date)
);

-- news articles table
DROP TABLE IF EXISTS news_articles;
CREATE TABLE news_articles (
    id            SERIAL PRIMARY KEY,
    ticker        VARCHAR(20)  NOT NULL,
    content       TEXT         NOT NULL,
    source        VARCHAR(200),
    published_at  TIMESTAMP,
    url           TEXT
);

-- daily sentiment table
DROP TABLE IF EXISTS daily_sentiment;
CREATE TABLE daily_sentiment (
    id             SERIAL PRIMARY KEY,
    ticker         VARCHAR(20) NOT NULL,
    date           DATE        NOT NULL,
    article_count  INTEGER,
    avg_compound   DECIMAL(5,4),
    positive_count INTEGER,
    negative_count INTEGER,
    UNIQUE(ticker, date)
);

ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS sentiment_label   VARCHAR(20);
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS sentiment_score   DECIMAL(5,4);
ALTER TABLE news_articles ADD COLUMN IF NOT EXISTS sentiment_compound DECIMAL(5,4);