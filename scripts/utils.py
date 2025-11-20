# scripts/utils.py
import os
import pandas as pd
import numpy as np
import joblib

# Default feature order used during training
DEFAULT_FEATURE_ORDER = [
    # Update this list to the exact features you trained with, in order.
    # Example for standard diabetes dataset:
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

def load_scaler(scaler_path):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
    return joblib.load(scaler_path)

def prepare_input_df(values: dict, feature_order=DEFAULT_FEATURE_ORDER, scaler_path=None):
    """
    Build a 1-row DataFrame from a dict of inputs (values) and optionally scale it
    using provided scaler_path (joblib file).
    - values: dict mapping feature name -> value
    - feature_order: list of feature names in the same order used for training
    """
    # Build row with feature_order
    row = {}
    for feat in feature_order:
        if feat in values:
            row[feat] = values[feat]
        else:
            # if missing feature, set np.nan (train script should have handled this)
            row[feat] = np.nan

    df = pd.DataFrame([row], columns=feature_order)

    # If scaler provided, load and transform numeric columns
    if scaler_path:
        scaler = load_scaler(scaler_path)
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # If scaler was fit on feature_order, assume transform on all columns
            X = df.values.astype(float)
            Xs = scaler.transform(X)
            df = pd.DataFrame(Xs, columns=feature_order)
        except Exception as e:
            # fallback: don't scale if something fails
            pass

    return df

def format_percent(p):
    """Format probability as percent string."""
    return f"{p*100:0.1f}%"

def format_yesno(v):
    return "Yes" if v else "No"

