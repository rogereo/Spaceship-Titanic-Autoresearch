"""
train.py — XGBoost with cabin parsing, GroupId, and richer CryoSleep-spending interaction.
Iteration 36: Replace binary CryoSleep_NoSpending with CryoSleep_Spending_Ratio to test
whether the intensity of spending conditioned on cryo status improves prediction.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import xgboost as xgb
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


def engineer_features(df):
    """Engineer spending and spending-cryo interaction features."""
    df_copy = df.copy()
    
    # Fill missing spending with 0
    for col in SPENDING:
        df_copy[col] = df_copy[col].fillna(0)
    
    # Total spending
    df_copy["TotalSpending"] = df_copy[SPENDING].sum(axis=1)
    
    # Interaction: CryoSleep status and spending ratio
    # For passengers in cryo-sleep, compute spending as a fraction; for others, set to 0.
    # Use log-transformed spending to capture intensity better.
    cryo_status = df_copy["CryoSleep"].fillna(False)
    log_spending = np.log1p(df_copy["TotalSpending"])
    df_copy["CryoSleep_LogSpending"] = (
        (cryo_status == True) * log_spending
    )
    
    return df_copy


def build_predict_fn():
    train_df, _ = prepare.load_split()
    
    # Parse Cabin into Deck, RoomNum, Side
    cabin_data = train_df["Cabin"].apply(parse_cabin)
    train_df["Deck"] = cabin_data.apply(lambda x: x[0])
    train_df["RoomNum"] = cabin_data.apply(lambda x: x[1])
    train_df["Side"] = cabin_data.apply(lambda x: x[2])
    
    # Extract group ID from PassengerId
    train_df["GroupId"] = train_df["PassengerId"].apply(extract_group_id)
    
    # Engineer features
    train_df = engineer_features(train_df)
    
    # Create explicit missingness flags for CryoSleep and VIP before filling
    train_df["CryoSleep_Missing"] = train_df["CryoSleep"].isna().astype(int)
    train_df["VIP_Missing"] = train_df["VIP"].isna().astype(int)
    
    # Convert CryoSleep and VIP to string for categorical encoding
    train_df["CryoSleep"] = train_df["CryoSleep"].fillna("Unknown").astype(str)
    train_df["VIP"] = train_df["VIP"].fillna("Unknown").astype(str)
    
    # Build feature set
    feature_cols = (
        ["Age"] + SPENDING + ["TotalSpending", "CryoSleep_LogSpending"] +
        CATEGORICAL + 
        ["Deck", "RoomNum", "Side", "GroupId"] + 
        ["CryoSleep_Missing", "VIP_Missing"]
    )
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
    
    numeric_feature_cols = (
        ["Age", "RoomNum", "CryoSleep_Missing", "VIP_Missing"] + 
        SPENDING + ["TotalSpending", "CryoSleep_LogSpending"]
    )
    categorical_feature_cols = CATEGORICAL + ["Deck", "Side", "GroupId"]
    
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_feature_cols),
        ("cat", categorical_transformer, categorical_feature_cols),
    ])
    
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0)),
    ])
    
    pipe.fit(X, y)
    
    def predict(X_val):
        # Parse Cabin in validation set
        cabin_data = X_val["Cabin"].apply(parse_cabin)
        X_val_copy = X_val.copy()
        X_val_copy["Deck"] = cabin_data.apply(lambda x: x[0])
        X_val_copy["RoomNum"] = cabin_data.apply(lambda x: x[1])
        X_val_copy["Side"] = cabin_data.apply(lambda x: x[2])
        
        # Extract group ID
        X_val_copy["GroupId"] = X_val_copy["PassengerId"].apply(extract_group_id)
        
        # Engineer features
        X_val_copy = engineer_features(X_val_copy)
        
        # Create explicit missingness flags for CryoSleep and VIP before filling
        X_val_copy["CryoSleep_Missing"] = X_val_copy["CryoSleep"].isna().astype(int)
        X_val_copy["VIP_Missing"] = X_val_copy["VIP"].isna().astype(int)
        
        # Handle CryoSleep and VIP
        X_val_copy["CryoSleep"] = X_val_copy["CryoSleep"].fillna("Unknown").astype(str)
        X_val_copy["VIP"] = X_val_copy["VIP"].fillna("Unknown").astype(str)
        
        return pipe.predict(X_val_copy[feature_cols]).astype(bool)
    
    return predict


if __name__ == "__main__":
    predict_fn = build_predict_fn()
    score = prepare.evaluate(predict_fn)
    print(f"VAL_ACCURACY: {score:.4f}")
