import pandas as pd

# путь к датасету: ../data/raw/Telco_customer_churn.xlsx
def load_and_clean_data(path: str) -> pd.DataFrame:
    """Загружает сырые данные и применяет базовую очистку."""

    data = pd.read_excel(path)
    data.columns = data.columns.str.replace(' ', '_').str.lower()
    data = data.drop(columns=['country', 'state', 'customerid', 'churn_label', 'count', 'lat_long', 'churn_reason', 'churn_score', 'latitude', 'longitude', 'total_charges'])

    data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})
    data['senior_citizen'] = data['senior_citizen'].map({'No': 0, 'Yes': 1})
    data['partner'] = data['partner'].map({'No': 0, 'Yes': 1})
    data['dependents'] = data['dependents'].map({'No': 0, 'Yes': 1})
    data['phone_service'] = data['phone_service'].map({'No': 0, 'Yes': 1})
    data['paperless_billing'] = data['paperless_billing'].map({'No': 0, 'Yes': 1})
    
    return data
