# scripts/data_preprocessing.py

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw" / "ai_sud_papers.csv"
PROCESSED_PATH = DATA_DIR / "processed" / "ai_sud_clean.csv"

def load_data(path=RAW_PATH):
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Example cleaning steps – customize to match your project
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    # Cast numeric
    numeric_cols = ["citation_count", "impact_factor", "year"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Standardize dataset names
    if "dataset_name" in df.columns:
        df["dataset_name"] = (
            df["dataset_name"]
            .str.strip()
            .str.lower()
            .replace({
                "ehr": "electronic health records",
                "electronic health record": "electronic health records",
                "clinical records": "electronic health records",
            })
        )
    
    # Binary encoding for ethics
    if "ethics_statement" in df.columns:
        df["ethics_mentioned"] = df["ethics_statement"].astype(str).str.lower().isin(["yes", "true", "1"])
    
    return df

def save_data(df: pd.DataFrame, path=PROCESSED_PATH):
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

if __name__ == "__main__":
    df_raw = load_data()
    df_clean = clean_data(df_raw)
    save_data(df_clean)
    print(f"Saved cleaned data to {PROCESSED_PATH}")
