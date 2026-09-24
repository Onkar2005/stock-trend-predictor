"""
train_model.py
---------------
Trains an XGBoost classifier to predict the upcoming stock trend
(Down / Sideways / Up) from OHLCV + technical-indicator features, and
saves the trained model + scaler + metadata to the models/ folder so the
Streamlit app can load them.

Usage:
    python train_model.py                       # uses data/sample_stock_data.csv
    python train_model.py --csv path/to/data.csv # use your own OHLCV data

Your CSV just needs these columns: Date, Open, High, Low, Close, Volume
"""

import argparse
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from feature_engineering import FEATURE_COLUMNS, TREND_LABELS, build_feature_dataset


def train(csv_path: str, model_dir: str = "models"):
    print(f"Loading data from {csv_path} ...")
    raw = pd.read_csv(csv_path, parse_dates=["Date"])
    raw.columns = [c.strip().capitalize() if c.lower() != "date" else "Date" for c in raw.columns]

    X, y, _ = build_feature_dataset(raw)
    print(f"Feature matrix: {X.shape}, classes: {sorted(y.unique())}")

    # Time-based split (NOT random shuffle) because this is time-series data -
    # training on the past and testing on the most recent slice is the
    # honest way to evaluate a trading model. Shuffled splits leak future
    # information into training and make results look artificially good.
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )

    print("Training XGBoost model...")
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=[TREND_LABELS[i] for i in sorted(TREND_LABELS)]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    # feature importance, handy to show in the project report / viva
    importance = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1]
    )
    print("\nTop features:")
    for name, score in importance[:10]:
        print(f"  {name:<20s} {score:.4f}")

    import os
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/xgb_trend_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{model_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "trend_labels": TREND_LABELS,
                "test_accuracy": acc,
                "n_train": len(X_train),
                "n_test": len(X_test),
            },
            f,
            indent=2,
        )
    print(f"\nSaved model, scaler and metadata to '{model_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/sample_stock_data.csv")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()
    train(args.csv, args.model_dir)
