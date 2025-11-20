# scripts/train.py
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

from scripts.preprocess import load_data
from scripts.utils import DEFAULT_FEATURE_ORDER

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data", "diabetes.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def main():
    df = load_data(DATA_PATH)
    # Determine target column (common: 'Outcome' or 'target')
    if "Outcome" in df.columns:
        target_col = "Outcome"
    elif "target" in df.columns:
        target_col = "target"
    else:
        raise ValueError("No 'Outcome' or 'target' column found in dataset. Update dataset to include target column.")

    # Ensure features exist
    features = [f for f in DEFAULT_FEATURE_ORDER if f in df.columns]
    missing = set(DEFAULT_FEATURE_ORDER) - set(features)
    if missing:
        print("Warning: the following mandatory features are missing from dataset:", missing)
        # We'll continue with intersection
    X = df[features].copy()
    y = df[target_col].astype(int)

    # Impute numeric nan with median
    for c in X.columns:
        if X[c].isna().sum() > 0:
            X[c] = X[c].fillna(X[c].median())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Model
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)

    # Evaluate
    preds = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)[:, 1]
    print("Model training completed!")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, preds))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Saved model to", MODEL_PATH)
    print("Saved scaler to", SCALER_PATH)

if __name__ == "__main__":
    main()

