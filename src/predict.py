import os

import pandas as pd

from src.data import clean_data, feature_cols, load_test_data
from src.model import load_model


def main():
    test_path = os.path.join('data', 'raw', 'test.csv')
    submission_path = os.path.join('data', 'raw', 'submission.csv')
    output_path = os.path.join('output', 'my_submission.csv')
    model_path = os.path.join('output', 'best_model.pkl')

    model = load_model(model_path)

    X_test = load_test_data(test_path)
    X_test = clean_data(X_test)

    submission = pd.read_csv(submission_path)
    submission['Churn'] = model.predict_proba(X_test[feature_cols])[:, 1]
    submission.to_csv(output_path, index=False)

    print(f'Submission file saved to {output_path}')


if __name__ == '__main__':
    main()
