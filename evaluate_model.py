"""
evaluate_model.py
------------------
Computes accuracy, precision, recall, F1, ROC-AUC, MAE and MSE for the
trained model on the held-out test split, and prints them as percentages
you can quote directly (e.g. on a resume or in a report).

Run this AFTER train_model.py has created models/xgb_trend_model.pkl.

Usage:
    python evaluate_model.py                       # uses data/sample_stock_data.csv
    python evaluate_model.py --csv your_data.csv    # evaluate on different data
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from feature_engineering import TREND_LABELS, build_feature_dataset


def evaluate(csv_path: str, model_dir: str = "models"):
    with open(f"{model_dir}/xgb_trend_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    raw = pd.read_csv(csv_path, parse_dates=["Date"])
    X, y, _ = build_feature_dataset(raw)

    # same chronological 80/20 split used in train_model.py, so this
    # reproduces the true held-out test set rather than re-testing on
    # data the model already saw during training
    split_idx = int(len(X) * 0.8)
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    classes = sorted(TREND_LABELS)  # [0, 1, 2]
    y_test_bin = label_binarize(y_test, classes=classes)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    roc_auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    roc_auc_weighted = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")

    # MAE / MSE treating the 3 classes as ordered integers (0/1/2) -
    # measures how far off (in class-steps) wrong predictions are.
    # NOT a price-prediction error - this model doesn't predict price.
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # normalized to a 0-100% scale: divide by the max possible error (2 steps: Down<->Up)
    max_error = 2
    mae_pct = (mae / max_error) * 100
    mse_pct = (mse / (max_error ** 2)) * 100

    print("=" * 60)
    print("MODEL EVALUATION - held-out test set")
    print(f"Test set size: {len(y_test)} rows")
    print("=" * 60)
    print(f"{'Accuracy:':<28}{accuracy * 100:6.2f}%")
    print(f"{'Precision (macro avg):':<28}{precision_macro * 100:6.2f}%")
    print(f"{'Precision (weighted avg):':<28}{precision_weighted * 100:6.2f}%")
    print(f"{'Recall (macro avg):':<28}{recall_macro * 100:6.2f}%")
    print(f"{'Recall (weighted avg):':<28}{recall_weighted * 100:6.2f}%")
    print(f"{'F1-score (macro avg):':<28}{f1_macro * 100:6.2f}%")
    print(f"{'F1-score (weighted avg):':<28}{f1_weighted * 100:6.2f}%")
    print(f"{'ROC-AUC (macro, OvR):':<28}{roc_auc_macro * 100:6.2f}%")
    print(f"{'ROC-AUC (weighted, OvR):':<28}{roc_auc_weighted * 100:6.2f}%")
    print("-" * 60)
    print("Ordinal-distance metrics (classes treated as 0/1/2, NOT price error):")
    print(f"{'MAE (normalized):':<28}{mae_pct:6.2f}%   (raw MAE = {mae:.3f} class-steps)")
    print(f"{'MSE (normalized):':<28}{mse_pct:6.2f}%   (raw MSE = {mse:.3f} class-steps^2)")
    print("=" * 60)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "mae_normalized_pct": mae_pct,
        "mse_normalized_pct": mse_pct,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/sample_stock_data.csv")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()
    evaluate(args.csv, args.model_dir)
