"""Обучение, сравнение с baseline, сохранение лучшей модели"""
import logging                  # логи
from pathlib import Path        # удобная работа с путями папками/файлами
import joblib                   # сохранение и загрузка моделей

from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.pipeline import Pipeline

from src.data_prep import load_and_clean_data
from src.features import split_data, build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/Telco_customer_churn.xlsx"
MODEL_DIR = Path("models")
MODEL_VERSION = "v1"

def evaluate(model, X_test, y_test, name) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {'roc_auc': roc_auc_score(y_test, y_proba), 'f1': f1_score(y_test, y_pred)}
    logger.info('%s: ROC-AUC=%.3f, F1=%.3f', name, metrics['roc_auc'], metrics['f1'])
    
    return metrics

def main():
    data = load_and_clean_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(data)
    preprocessor = build_preprocessor(X_train)

    candidates = {
        'dummy': DummyClassifier(strategy='most_frequent'),
        'log_reg': LogisticRegression(max_iter=10000),
        'catboost': CatBoostClassifier(random_state=42, verbose=0)    # verbose=0 - без логов
    }
    results = {}

    for name, cls in candidates.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', cls)])
        pipeline.fit(X_train, y_train)
        results[name] = (pipeline, evaluate(pipeline, X_test, y_test, name))

    best_name = max((k for k in results if k != "dummy"), key=lambda k: results[k][1]["roc_auc"])
    best_model, best_metrics = results[best_name]
    logger.info("Лучшая модель: %s (%s)", best_name, best_metrics)

    MODEL_DIR.mkdir(exist_ok=True)
    out_path = MODEL_DIR / f"model_{MODEL_VERSION}.joblib"
    joblib.dump(best_model, out_path)
    logger.info("Модель сохранена: %s", out_path)

if __name__ == '__main__':
    main()
