"""
train.py — logistic regression with cabin parsing, group features, and aggregate spending.
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
SPENDING = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def parse_cabin(cabin_str):
    """Parse Cabin 'F/0/S' into deck, room_num, side."""
    if pd.isna(cabin_str):
        return np.nan, np.nan, np.nan
    parts = str(cabin_str).split("/")
    if len(parts) == 3:
        return parts[0], int(parts[1]) if parts[1].isdigit() else np.nan, parts[2]
    return np.nan, np.nan, np.nan


def extract_group_id(passenger_id):
    """Extract group ID from PassengerId (e.g., '0001_01' -> '0001')."""
    if pd.isna(passenger_id):
        return np.nan
    return str(passenger_id).split("_")[0]


def build_predict_fn():
    train_df, _ = prepare.load_split()
    
    # Parse Cabin into Deck, RoomNum, Side
    cabin_data = train_df["Cabin"].apply(parse_cabin)
    train_df["Deck"] = cabin_data.apply(lambda x: x[0])
    train_df["RoomNum"] = cabin_data.apply(lambda x: x[1])
    train_df["Side"] = cabin_data.apply(lambda x: x[2])
    
    # Extract group ID from PassengerId
    train_df["GroupId"] = train_df["PassengerId"].apply(extract_group_id)
    
    # Compute group size for each passenger
    group_sizes = train_df["GroupId"].value_counts().to_dict()
    train_df["GroupSize"] = train_df["GroupId"].map(group_sizes).fillna(1)
    
    # Fill missing values in spending columns with 0 (indicates not spent)
    for col in SPENDING:
        train_df[col] = train_df[col].fillna(0)
    
    # Compute total spending across all categories
    train_df["TotalSpending"] = train_df[SPENDING].sum(axis=1)
    
    # Convert CryoSleep and VIP to numeric (handle NaN explicitly)
    train_df["CryoSleep"] = train_df["CryoSleep"].fillna("Unknown").astype(str)
    train_df["VIP"] = train_df["VIP"].fillna("Unknown").astype(str)
    
    # Build feature set: include both individual spending and aggregate spending
    feature_cols = ["Age", "TotalSpending"] + SPENDING + CATEGORICAL + ["Deck", "RoomNum", "Side", "GroupId", "GroupSize"]
    X = train_df[feature_cols]
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
        ("num", numeric_transformer, ["Age", "TotalSpending", "RoomNum", "GroupSize"] + SPENDING),
        ("cat", categorical_transformer, CATEGORICAL + ["Deck", "Side", "GroupId"]),
    ])
    
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    
    pipe.fit(X, y)
    
    # Store group_sizes mapping for use in predict function
    pipe.group_sizes = group_sizes
    
    def predict(X_val):
        # Parse Cabin in validation set
        cabin_data = X_val["Cabin"].apply(parse_cabin)
        X_val_copy = X_val.copy()
        X_val_copy["Deck"] = cabin_data.apply(lambda x: x[0])
        X_val_copy["RoomNum"] = cabin_data.apply(lambda x: x[1])
        X_val_copy["Side"] = cabin_data.apply(lambda x: x[2])
        
        # Extract group ID
        X_val_copy["GroupId"] = X_val_copy["PassengerId"].apply(extract_group_id)
        
        # Compute group size using training set mapping (default to 1 for unseen groups)
        X_val_copy["GroupSize"] = X_val_copy["GroupId"].map(pipe.group_sizes).fillna(1)
        
        # Fill missing spending with 0
        for col in SPENDING:
            X_val_copy[col] = X_val_copy[col].fillna(0)
        
        # Compute total spending
        X_val_copy["TotalSpending"] = X_val_copy[SPENDING].sum(axis=1)
        
        # Handle CryoSleep and VIP
        X_val_copy["CryoSleep"] = X_val_copy["CryoSleep"].fillna("Unknown").astype(str)
        X_val_copy["VIP"] = X_val_copy["VIP"].fillna("Unknown").astype(str)
        
        return pipe.predict(X_val_copy[feature_cols]).astype(bool)
    
    return predict


if __name__ == "__main__":
    predict_fn = build_predict_fn()
    score = prepare.evaluate(predict_fn)
    print(f"VAL_ACCURACY: {score:.4f}")
