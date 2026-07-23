# Customer Churn Risk API

Predicts the probability that a telecom customer will churn, and (in progress) exposes that
prediction as a REST API with prediction history stored in PostgreSQL.

**Status: 🚧 in progress.** The ML pipeline, data cleaning, and tests are done and covered by
unit tests. The FastAPI layer is not built yet — see [Roadmap](#roadmap).

## Why this project

Built to practice a realistic ML engineering workflow end-to-end: clean data → engineer
features without leakage → compare models against a baseline → serve predictions → persist
them for later analysis. Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/abdallahwagih/telco-customer-churn)
(IBM sample dataset, via Kaggle).

## Tech stack

- **ML**: pandas, scikit-learn (`Pipeline`, `ColumnTransformer`, `TargetEncoder`), CatBoost
- **Data**: PostgreSQL, Docker Compose
- **Testing**: pytest

## What's done

- [x] EDA (`notebooks/01_eda.ipynb`)
- [x] Data cleaning module (`src/data_prep.py`)
- [x] Feature engineering pipeline with no train/test leakage — one-hot + target encoding
      fitted only on train, inside a single `ColumnTransformer` (`src/features.py`)
- [x] Model comparison: `DummyClassifier` baseline vs. `LogisticRegression` vs. `CatBoostClassifier`,
      selected by ROC-AUC (`src/train.py`)
- [x] Inference module (`src/predict.py`)
- [x] Unit tests for data cleaning, feature pipeline, and inference (`tests/`)
- [x] PostgreSQL schema: `clients` + `predictions` (with indexes on `client_id` and `predicted_at`),
      provisioned via `db/init.sql` and Docker Compose

## Roadmap

- [ ] FastAPI service: `POST /predict`, `GET /predictions/{client_id}`, `GET /stats`
- [ ] Wire the API and Postgres together in `docker-compose.yml`
- [ ] Request validation, structured logging, error handling
- [ ] CI (GitHub Actions) running `pytest` on every push

## Project structure

```
.
├── src/
│   ├── data_prep.py      # load + clean raw data
│   ├── features.py       # train/test split, ColumnTransformer (no leakage)
│   ├── train.py          # trains & compares models, saves the best one
│   └── predict.py        # loads saved model, runs inference
├── tests/                 # pytest unit tests
├── notebooks/
│   └── 01_eda.ipynb       # exploratory data analysis
├── db/
│   └── init.sql           # Postgres schema (clients, predictions)
├── docker-compose.yml
├── pytest.ini
└── README.md
```

## Model results



| Model               | ROC-AUC | F1   |
|---------------------|---------|------|
| Dummy baseline       | 0.5       | 0    |
| Logistic Regression  | 0.846       | 0.609    |
| CatBoost             | 0.856       | 0.6    |

## Running locally

**ML pipeline:**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# place the dataset at data/raw/Telco_customer_churn.xlsx (from the Kaggle link above)

python -m src.train        # trains models, saves the best one to models/
pytest tests/ -v            # runs the test suite
```

**Database:**
```bash
cp .env.example .env        # fill in POSTGRES_* values
docker compose up -d        # starts Postgres, applies db/init.sql on first run
```

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — IBM sample
dataset distributed via Kaggle, ~7,000 customer records with a binary churn label.
