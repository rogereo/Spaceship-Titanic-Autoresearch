"""
train.py — feature engineering from Cabin and categorical columns.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import prepare

NUMERIC = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL = ["HomePlanet", "Destination", "CryoSleep", "VIP"]


def parse_cabin(cabin_str):
    """Parse Cabin 'F/0/S' into deck, room_num, side."""
    if pd.isna(cabin_str):
        return np.nan, np.nan, np.nan
    parts = str(cabin_str).split("/")
    if len(parts) == 3:
        return parts[0], int(parts[1]) if parts[1].isdigit() else np.nan, parts[2]
    return np.nan, np.nan, np.nan


def build_predict_fn():
    train_df, _ = prepare.load_split()
    
    # Parse Cabin into Deck, RoomNum, Side
    cabin_data = train_df["Cabin"].apply(parse_cabin)
    train_df["Deck"] = cabin_data.apply(lambda x: x[0])
    train_df["RoomNum"] = cabin_data.apply(lambda x: x[1])
    train_df["Side"] = cabin_data.apply(lambda x: x[2])
    
    # Convert CryoSleep and VIP to numeric (they are object dtype)
    train_df["CryoSleep"] = train_df["CryoSleep"].astype(str)
    train_df["VIP"] = train_df["VIP"].astype(str)
    
    # Build feature set
    X = train_df[NUMERIC + CATEGORICAL + ["Deck", "RoomNum", "Side"]]
    y = train_df["Transported"].astype(int)
    
    # Define preprocessing for numeric and categorical columns
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC + ["RoomNum"]),
        ("cat", categorical_transformer, CATEGORICAL + ["Deck", "Side"]),
    ])
    
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    
    pipe.fit(X, y)
    
    def predict(X_val):
        # Parse Cabin in validation set
        cabin_data = X_val["Cabin"].apply(parse_cabin)
        X_val_copy = X_val.copy()
        X_val_copy["Deck"] = cabin_data.apply(lambda x: x[0])
        X_val_copy["RoomNum"] = cabin_data.apply(lambda x: x[1])
        X_val_copy["Side"] = cabin_data.apply(lambda x: x[2])
        X_val_copy["CryoSleep"] = X_val_copy["CryoSleep"].astype(str)
        X_val_copy["VIP"] = X_val_copy["VIP"].astype(str)
        return pipe.predict(X_val_copy[NUMERIC + CATEGORICAL + ["Deck", "RoomNum", "Side"]]).astype(bool)
    
    return predict


if __name__ == "__main__":
    predict_fn = build_predict_fn()
    score = prepare.evaluate(predict_fn)
    print(f"VAL_ACCURACY: {score:.4f}")
