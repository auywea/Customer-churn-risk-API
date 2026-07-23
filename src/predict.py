"""Загрузка модели и инференс для одного клиента"""
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models/model_v1.joblib")
_model = None                                   # ленивая загрузка


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)

    return _model

def predict_churn(features: dict) -> dict:
    """
    features - сырые признаки клиента.
    Возвращает {"probability": float, "model_version": str}.
    """
    model = load_model()
    X = pd.DataFrame([features])
    proba = model.predict_proba(X)[0, 1]
    return {'probability': float(proba), 'model_version': MODEL_PATH.stem}