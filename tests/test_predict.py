import pytest
from src.predict import predict_churn, MODEL_PATH
from src.data_prep import load_and_clean_data
from src.features import split_data

def test_predict_returns_valid_probability():
    data = load_and_clean_data("data/raw/Telco_customer_churn.xlsx")
    X_train, X_test, y_train, y_test = split_data(data)
    features = X_test.iloc[0].to_dict()

    result = predict_churn(features)
    assert 0.0 <= result["probability"] <= 1.0
    assert "model_version" in result
    assert isinstance(result["probability"], float)