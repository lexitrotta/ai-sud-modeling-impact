# scripts/train_models.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

PROCESSED_PATH = Path("data/processed/ai_sud_clean.csv")

TARGET_COL = "citation_count"
FEATURE_COLS = [
    "impact_factor",
    "year",
    "ethics_mentioned",
    # Add encoded dataset/method features here
]

def load_processed():
    return pd.read_csv(PROCESSED_PATH)

def split_data(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def eval_regression(model, X_test, y_test, name="model"):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"{name} -> R2={r2:.2f}, MAE={mae:.2f}, RMSE={rmse:.2f}")

def main():
    df = load_processed()
    X_train, X_test, y_train, y_test = split_data(df)

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    eval_regression(rf, X_test, y_test, name="Random Forest")

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    eval_regression(xgb, X_test, y_test, name="XGBoost")

if __name__ == "__main__":
    main()
