"""
train.py — Iteration 59: Exploratory Data Analysis (EDA).
Print detailed statistics on the training set to identify structural features missed by baseline.
Run EDA, gather insights, then revert to a feature-based improvement in next iteration.
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
from sklearn.model_selection import cross_val_score

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


def extract_passenger_num(passenger_id):
    """Extract passenger number within group (e.g., '0001_01' -> 1)."""
    if pd.isna(passenger_id):
        return np.nan
    try:
        return int(str(passenger_id).split("_")[1])
    except (IndexError, ValueError):
        return np.nan


def engineer_features(df):
    """Engineer spending and spending-cryo interaction features."""
    df_copy = df.copy()
    
    # Track which rows have any spending data recorded
    spending_recorded = ~df_copy[SPENDING].isna().any(axis=1)
    df_copy["Spending_Recorded"] = spending_recorded.astype(int)
    
    # Fill missing spending with median per HomePlanet group
    for col in SPENDING:
        missing_mask = df_copy[col].isna()
        if missing_mask.any():
            group_medians = df_copy.groupby("HomePlanet")[col].median()
            df_copy.loc[missing_mask, col] = df_copy.loc[missing_mask, "HomePlanet"].map(group_medians)
            global_median = df_copy[col].median()
            df_copy[col] = df_copy[col].fillna(global_median)
    
    # Total spending
    df_copy["TotalSpending"] = df_copy[SPENDING].sum(axis=1)
    
    # Interaction: CryoSleep status and log spending
    cryo_status = df_copy["CryoSleep"].fillna(False)
    log_spending = np.log1p(df_copy["TotalSpending"])
    df_copy["CryoSleep_LogSpending"] = (
        (cryo_status == True) * log_spending
    )
    
    return df_copy


def add_group_features(df):
    """Add group-level aggregate features based on PassengerId group."""
    df_copy = df.copy()
    
    # Extract group ID
    df_copy["GroupId"] = df_copy["PassengerId"].apply(extract_group_id)
    
    # Compute group size
    group_sizes = df_copy.groupby("GroupId").size()
    df_copy["GroupSize"] = df_copy["GroupId"].map(group_sizes)
    
    # Compute group median total spending
    group_median_spending = df_copy.groupby("GroupId")["TotalSpending"].median()
    df_copy["GroupMedianSpending"] = df_copy["GroupId"].map(group_median_spending)
    
    return df_copy


def run_eda(df):
    """Print exploratory statistics to understand data structure."""
    print("\n=== EDA SUMMARY ===")
    
    # Class balance
    target_dist = df["Transported"].value_counts()
    print(f"\nClass distribution:\n{target_dist}\nRatio (False:True) = {target_dist[False]:.0f}:{target_dist[True]:.0f}")
    
    # Missing data
    print("\nMissing data (count, pct):")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    for col, count in missing.items():
        pct = 100 * count / len(df)
        print(f"  {col}: {count} ({pct:.1f}%)")
    
    # Numeric correlations with target
    print("\nNumeric feature correlations with Transported:")
    for col in NUMERIC:
        if col in df.columns:
            corr = df[col].corr(df["Transported"].astype(int))
            print(f"  {col}: {corr:.4f}")
    
    # Categorical distributions
    print("\nCategorical feature distributions (counts per category):")
    for col in CATEGORICAL:
        if col in df.columns:
            print(f"  {col}:")
            counts = df[col].value_counts(dropna=False)
            for cat, count in counts.items():
                print(f"    {cat}: {count}")
    
    # Cabin parsing
    cabin_data = df["Cabin"].apply(parse_cabin)
    decks = cabin_data.apply(lambda x: x[0])
    print(f"\nDeck distribution (counts):")
    deck_counts = decks.value_counts(dropna=False)
    for deck, count in deck_counts.items():
        print(f"  {deck}: {count}")
    
    # Group size distribution
    group_ids = df["PassengerId"].apply(extract_group_id)
    group_sizes = group_ids.value_counts()
    print(f"\nGroup size distribution:")
    print(f"  Min: {group_sizes.min()}, Max: {group_sizes.max()}, Mean: {group_sizes.mean():.2f}")
    print(f"  Count: {group_sizes.value_counts().to_dict()}")
    
    # Passenger number within group
    pax_nums = df["PassengerId"].apply(extract_passenger_num)
    print(f"\nPassenger number within group (unique values: {pax_nums.nunique()})")
    print(f"  Distribution: {pax_nums.value_counts().sort_index().to_dict()}")
    
    # Spending pattern by Transported
    print("\nSpending by Transported status:")
    for spending_col in SPENDING:
        has_spending = (df[spending_col] > 0).sum()
        transported_with_spending = df[df["Transported"] & (df[spending_col] > 0)].shape[0]
        not_transported_with_spending = df[~df["Transported"] & (df[spending_col] > 0)].shape[0]
        print(f"  {spending_col}: {has_spending} total, {transported_with_spending} T, {not_transported_with_spending} NT")
    
    # HomePlanet × Destination cross-tabulation
    print("\nHomePlanet × Destination cross-tabulation (counts):")
    cross = pd.crosstab(df["HomePlanet"], df["Destination"], margins=False)
    print(cross)
    
    print("\n=== END EDA ===\n")


def build_predict_fn():
    train_df, _ = prepare.load_split()
    
    # Parse Cabin into Deck, RoomNum, Side
    cabin_data = train_df["Cabin"].apply(parse_cabin)
    train_df["Deck"] = cabin_data.apply(lambda x: x[0])
    train_df["RoomNum"] = cabin_data.apply(lambda x: x[1])
    train_df["Side"] = cabin_data.apply(lambda x: x[2])
    
    # Engineer features
    train_df = engineer_features(train_df)
    
    # Add group-level features
    train_df = add_group_features(train_df)
    
    # Run EDA before model building
    run_eda(train_df)
    
    # Create explicit missingness flags for CryoSleep and VIP
    train_df["CryoSleep_Missing"] = train_df["CryoSleep"].isna().astype(int)
    train_df["VIP_Missing"] = train_df["VIP"].isna().astype(int)
    
    # Convert CryoSleep and VIP to string for categorical encoding
    train_df["CryoSleep"] = train_df["CryoSleep"].fillna("Unknown").astype(str)
    train_df["VIP"] = train_df["VIP"].fillna("Unknown").astype(str)
    
    # Build feature set
    feature_cols = (
        ["Age"] + SPENDING + ["TotalSpending", "CryoSleep_LogSpending", "Spending_Recorded"] +
        CATEGORICAL + 
        ["Deck", "RoomNum", "Side", "GroupId"] + 
        ["CryoSleep_Missing", "VIP_Missing"] +
        ["GroupSize", "GroupMedianSpending"]
    )
    X = train_df[feature_cols]
    y = train_df["Transported"].astype(int)
    
    # Define preprocessing
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    numeric_feature_cols = (
        ["Age", "RoomNum", "CryoSleep_Missing", "VIP_Missing", "Spending_Recorded"] + 
        SPENDING + ["TotalSpending", "CryoSleep_LogSpending"] +
        ["GroupSize", "GroupMedianSpending"]
    )
    categorical_feature_cols = CATEGORICAL + ["Deck", "Side", "GroupId"]
    
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_feature_cols),
        ("cat", categorical_transformer, categorical_feature_cols),
    ])
    
    # Use best config from iter 58
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", xgb.XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)),
    ])
    
    pipe.fit(X, y)
    
    def predict(X_val):
        cabin_data = X_val["Cabin"].apply(parse_cabin)
        X_val_copy = X_val.copy()
        X_val_copy["Deck"] = cabin_data.apply(lambda x: x[0])
        X_val_copy["RoomNum"] = cabin_data.apply(lambda x: x[1])
        X_val_copy["Side"] = cabin_data.apply(lambda x: x[2])
        
        # Engineer features
        X_val_copy = engineer_features(X_val_copy)
        
        # Add group-level features from training stats
        X_val_copy["GroupId"] = X_val_copy["PassengerId"].apply(extract_group_id)
        X_val_copy["GroupSize"] = X_val_copy["GroupId"].map(pd.Series(
            train_df.groupby("GroupId").size().to_dict()
        ))
        X_val_copy["GroupMedianSpending"] = X_val_copy["GroupId"].map(pd.Series(
            train_df.groupby("GroupId")["TotalSpending"].median().to_dict()
        ))
        
        # Missingness flags
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
