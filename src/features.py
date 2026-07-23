import pandas as pd

from sklearn.preprocessing import TargetEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def split_data(data: pd.DataFrame, target_col: str = 'churn_value', test_size: float = 0.2, random_state: int = 42):
    """split до любого feature engineering, чтобы избежать утечки данных в test data"""
    y = data[target_col]
    X = data.drop(columns=[target_col])
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    num_cols = X_train.select_dtypes(include='number').columns
    cat_cols = X_train.select_dtypes(exclude='number').columns.drop('city')

    num_transformer = Pipeline([('scaler', StandardScaler())])
    cat_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    city_transformer = Pipeline([('encoder', TargetEncoder(smooth=1))])

    return ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols),
        ('city', city_transformer, ['city']),
    ])