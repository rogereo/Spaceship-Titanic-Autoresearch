"""
train.py — the agent's sandbox. Rewritten each turn.
Must call prepare.evaluate(predict_fn) and print VAL_ACCURACY: 0.XXXX as the last line.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import prepare

NUMERIC = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def build_predict_fn():
    train_df, _ = prepare.load_split()
    
    # First, explore the data structure
    print("=== DATA EXPLORATION ===", file=__import__('sys').stderr)
    print(f"Shape: {train_df.shape}", file=__import__('sys').stderr)
    print(f"Columns: {list(train_df.columns)}", file=__import__('sys').stderr)
    print(f"\nMissing values:\n{train_df.isnull().sum()}", file=__import__('sys').stderr)
    print(f"\nData types:\n{train_df.dtypes}", file=__import__('sys').stderr)
    print(f"\nFirst few rows:\n{train_df.head()}", file=__import__('sys').stderr)
    
    # Check correlations with target for numeric columns
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if "Transported" in numeric_cols:
        numeric_cols.remove("Transported")
    print(f"\nNumeric columns: {numeric_cols}", file=__import__('sys').stderr)
    
    corr_with_target = train_df[numeric_cols + ["Transported"]].corr()["Transported"].drop("Transported").sort_values(ascending=False)
    print(f"\nCorrelation with Transported:\n{corr_with_target}", file=__import__('sys').stderr)
    
    # Standard model
    X = train_df[NUMERIC]
    y = train_df["Transported"].astype(int)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X, y)

    def predict(X_val):
        return pipe.predict(X_val[NUMERIC]).astype(bool)
    return predict


if __name__ == "__main__":
    predict_fn = build_predict_fn()
    score = prepare.evaluate(predict_fn)
    print(f"VAL_ACCURACY: {score:.4f}")
