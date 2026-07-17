import os

import pandas as pd

from src.data import clean_data, feature_cols, load_data
from src.model import save_model, train_catboost


def main():
    train_path = os.path.join('data', 'raw', 'train.csv')
    model_path = os.path.join('output', 'best_model.pkl')

    data = load_data(train_path)
    data = clean_data(data)

    X = data[feature_cols]
    y = data['Churn']

    model, roc_auc = train_catboost(X, y)
    print(f'Validation ROC-AUC: {roc_auc:.4f}')

    save_model(model, model_path)
    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
