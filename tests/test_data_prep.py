import pandas as pd
from src.data_prep import load_and_clean_data

def test_load_and_clean_data(tmp_path):
    df = pd.DataFrame({'customerid': ['3318-ISQFK'], 'count': [1], 'country': ['US'], 'state': ['CA'], 'city': ['Los Angeles'],
                       'zip_code': ['90001'], 'lat_long': ['34.0522,-118.2437'], 'latitude': [34.0522], 'longitude': [-118.2437],
                       'gender': ['Female'], 'senior_citizen': ['No'], 'partner': ['Yes'], 'dependents': ['No'], 'tenure_months': [24],
                       'phone_service': ['Yes'], 'multiple_lines': ['No'], 'internet_service': ['Fiber optic'],
                       'online_security': ['No'], 'online_backup': ['Yes'], 'device_protection': ['No'], 'tech_support': ['No'],
                       'streaming_tv': ['Yes'], 'streaming_movies': ['Yes'], 'contract': ['Month-to-month'], 'paperless_billing': ['Yes'],
                       'payment_method': ['Electronic check'], 'monthly_charges': [85.5], 'total_charges': ['2052.0'], 'churn_label': ['Yes'],
                       'churn_value': [1], 'churn_score': [85], 'cltv': [3500.0], 'churn_reason': ['Competitor had better offers']
                       })
    
    path = tmp_path / "test.xlsx"
    df.to_excel(path, index=False)

    cleaned = load_and_clean_data(str(path))

    assert 'total_charges' not in cleaned.columns     # дропнута
    assert 'churn_score' not in cleaned.columns       # утёкший признак убран
    assert cleaned['gender'].isin([0, 1]).all()       # маппинг сработал
    assert cleaned.isnull().sum().sum() == 0          # нет пропусков после очистки