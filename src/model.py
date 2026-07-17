import pickle
from typing import Tuple

import numpy as np
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

from src.data import cat_cols, feature_cols, num_cols


def build_pipeline() -> Pipeline:
    num_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols),
        ]
    )

    clf = LogisticRegressionCV(
        Cs=[100, 10, 1, 0.1, 0.01, 0.001],
        cv=5,
        scoring='roc_auc',
        max_iter=500,
        refit=True,
        random_state=42,
    )

    return Pipeline([('preprocessor', preprocessor), ('classifier', clf)])


def train_model(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_pipeline()
    model.fit(X_train, y_train)

    y_valid_pred = model.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, y_valid_pred)

    return model, roc_auc


def train_catboost(X, y, depth=6, iterations=800, learning_rate=0.01):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        cat_features=cat_cols,
        verbose=0,
        random_seed=42,
    )
    model.fit(X_train, y_train)

    y_valid_pred = model.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, y_valid_pred)

    return model, roc_auc


def save_model(model, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)
