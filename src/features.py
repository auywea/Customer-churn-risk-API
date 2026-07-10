import pandas as pd

from sklearn.preprocessing import TargetEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def build_preprocessor(data) -> ColumnTransformer:
    encoder = TargetEncoder(smooth=1)
    data['city_encoded'] = encoder.fit_transform(data[['city']], data['churn_value'])
    data = data.drop(columns=['city'])

    y = data['churn_value']
    X = data.drop(columns=['churn_value'])

    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = X.select_dtypes(include=['str']).columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='No reason')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    return preprocessor