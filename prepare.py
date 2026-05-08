"""
prepare.py — IMMUTABLE evaluator. Do not modify during a run.
SHA pinned at top. The agent must call evaluate() and never redefine the split.
"""
EVALUATOR_VERSION = "v1.0.0-fixed-split-seed-42"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "train.csv"
RANDOM_STATE = 42
VAL_SIZE = 0.2


def load_split():
    """Returns (train_df, val_df) with a fixed split. Same call always returns the same split."""
    df = pd.read_csv(DATA_PATH)
    train_df, val_df = train_test_split(
        df, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=df["Transported"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def evaluate(predict_fn):
    """
    predict_fn: callable that takes a DataFrame (validation features, including PassengerId
    but excluding 'Transported') and returns a 1-D array-like of bool predictions.
    Returns: float validation accuracy in [0, 1].
    """
    _, val_df = load_split()
    y_true = val_df["Transported"].values
    X_val = val_df.drop(columns=["Transported"])
    y_pred = predict_fn(X_val)
    return float(accuracy_score(y_true, y_pred))
