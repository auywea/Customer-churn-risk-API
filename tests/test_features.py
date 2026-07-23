import pandas as pd
from src.features import split_data, build_preprocessor

def make_fake_data(n=60):
    return pd.DataFrame({
        'gender': [0, 1] * (n // 2),
        'city': (['A', 'B', 'C'] * n)[:n],
        'monthly_charges': range(n),
        'churn_value': [0, 1] * (n // 2),
    })

def test_split_removes_target():
    X_train, X_test, y_train, y_test = split_data(make_fake_data(), test_size=0.3)
    assert 'churn_value' not in X_train.columns
    assert len(X_train) + len(X_test) == 60

def test_preprocessor_fits_only_on_train():
    X_train, X_test, y_train, y_test = split_data(make_fake_data(), test_size=0.3)
    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    X_test_t = preprocessor.transform(X_test)  # transform, не fit_transform!
    assert X_train_t.shape[0] == len(X_train)
    assert X_test_t.shape[0] == len(X_test)