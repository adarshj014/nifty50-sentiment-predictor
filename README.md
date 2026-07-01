# Nifty 50 Sentiment Signal Generator

A machine learning research tool that combines fine-tuned FinBERT sentiment 
analysis on Indian financial news with technical indicators to generate 
directional signals for Nifty 50 stocks.

## Live Demo
[View App](https://yourusername-nifty50-sentiment-predictor.streamlit.app)

## What it does
- Fine-tuned FinBERT on Indian financial news (82% accuracy)
- Scores 51,000+ historical articles from major Indian financial sources
- XGBoost models for 1-day and 5-day price direction prediction
- SHAP explainability for every prediction
- Real-time yfinance data integration

## Tech Stack
- Python, PostgreSQL, Streamlit, Plotly
- HuggingFace Transformers (FinBERT fine-tuned)
- XGBoost, scikit-learn, SHAP
- yfinance for live data

## Model Performance
- 1-day accuracy: 51-53% (consistent with EMH)
- 5-day accuracy: 56%
- Per-stock models: 60-65% on high-signal stocks

## Project Structure
Introduction.py        — Home page
pages/
  1_Historical_Analysis.py
  2_Live_Prediction.py
  3_Model_Performance.py
app_utils/             — DB, predictor, styling utilities
models/                — Trained models (via Git LFS)
