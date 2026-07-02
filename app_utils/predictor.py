import joblib
import os
import streamlit as st

MODELS_DIR = "models"

@st.cache_resource
def load_models():
    models = {}
    models['clf_1d'] = joblib.load(f"{MODELS_DIR}/xgb_clf_1d.pkl")
    models['clf_5d'] = joblib.load(f"{MODELS_DIR}/xgb_clf_5d.pkl")
    models['reg_1d'] = joblib.load(f"{MODELS_DIR}/xgb_reg_1d.pkl")
    models['reg_5d'] = joblib.load(f"{MODELS_DIR}/xgb_reg_5d.pkl")
    models['features'] = joblib.load(f"{MODELS_DIR}/features.pkl")

    comparison_path = f"{MODELS_DIR}/model_comparison.pkl"
    if os.path.exists(comparison_path):
        models['comparison'] = joblib.load(comparison_path)

    per_stock_path = f"{MODELS_DIR}/per_stock_results.pkl"
    if os.path.exists(per_stock_path):
        models['per_stock'] = joblib.load(per_stock_path)

    return models

@st.cache_resource
def load_finbert():
    from transformers import BertTokenizer, BertForSequenceClassification
    import os

    model_path = "models/finbert_indian"

    if os.path.exists(model_path):
        print("Loading fine-tuned Indian FinBERT...")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model     = BertForSequenceClassification.from_pretrained(model_path)
    else:
        # fall back to original FinBERT from HuggingFace
        print("Loading original FinBERT...")
        tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        model     = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")

    model.eval()
    return tokenizer, model


def score_headline(text):
    import torch
    from torch.nn.functional import softmax

    tokenizer, model = load_finbert()

    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=256, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = softmax(outputs.logits, dim=1)
    labels = ["positive", "negative", "neutral"]
    pos, neg = probs[0][0].item(), probs[0][1].item()
    top = torch.argmax(probs, dim=1).item()

    return {
        'label': labels[top],
        'score': round(probs[0][top].item(), 4),
        'compound': round(pos - neg, 4)
    }