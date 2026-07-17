# Journey to Springfield

A Kaggle-style churn prediction project based on the `Journey to Springfield` dataset.

## Project structure

- `README.md` - project overview and instructions
- `requirements.txt` - Python dependencies
- `data/raw/` - raw input data files
- `notebooks/` - original notebook and exploratory analysis
- `src/` - source code modules for data preparation, training, and prediction
- `output/` - generated model artifacts and submission files
- `scripts/` - helper scripts for running the project

## Goals

- Build a reproducible ML pipeline for churn prediction
- Evaluate models using ROC-AUC
- Generate a submission file for Kaggle-style evaluation

## Setup

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Place your dataset files in `data/raw/`:

- `train.csv`
- `test.csv`
- `submission.csv`

3. Run training:

```bash
python src/train.py
```

4. Generate final predictions:

```bash
python src/predict.py
```

## Notes

- The notebook was reorganized into reusable Python modules.
- If you want, use Git to track changes and commit each step.
