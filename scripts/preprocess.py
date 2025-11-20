# scripts/preprocess.py
import pandas as pd
import numpy as np

def load_data(csv_path):
    """
    Load dataset from given path and perform light cleaning.
    Returns a DataFrame.
    """
    df = pd.read_csv(csv_path)
    # Basic normalization: strip column names
    df.columns = [c.strip() for c in df.columns]

    # If there are Yes/No columns, map them
    for col in df.columns:
        if df[col].dropna().isin(["Yes","No","yes","no"]).any():
            df[col] = df[col].map(lambda x: 1 if str(x).lower() == "yes" else (0 if str(x).lower() == "no" else np.nan))

    # Convert numeric-like columns to numeric where possible
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    # If there are missing numeric values, fill with median (safe default)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df

