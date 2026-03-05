#!/usr/bin/env python3
"""
Step 3: Train XGBoost v5 — Balanced Real-World Classifier
==========================================================
Trains a single XGBoost multi-class classifier on 1.18M real-world arrays.

Balance strategy (3-pronged):
  1. Undersample majority: cap timsort at ~3× the minority class count
     so the training set is roughly 50% timsort / 25% heapsort / 25% introsort
  2. sample_weight: inverse-frequency weighting on the undersampled set
  3. eval_metric: mlogloss (proper multi-class metric)

Split: 70% train / 15% val / 15% test (stratified)

Inputs:  data/training_dataset.csv  (from Step 2)
Outputs: models/xgboost_v5/        (model JSON)
         results/xgboost_v5/       (metrics, predictions, confusion matrix)

Usage:
    python3 scripts/train_xgboost_v5.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
DATA_CSV    = ROOT / "data" / "training_dataset.csv"
MODEL_DIR   = ROOT / "models" / "xgboost_v5"
RESULTS_DIR = ROOT / "results" / "xgboost_v5"

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
SEED = 42

# ── XGBoost hyperparameters ──────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
)


# ── Balance the dataset ──────────────────────────────────────────────────

def balanced_undersample(df: pd.DataFrame, label_col: str,
                         max_ratio: float = 3.0) -> pd.DataFrame:
    """
    Undersample majority classes so no class exceeds max_ratio × minority count.

    With 52K introsort, 125K heapsort, 1.01M timsort and max_ratio=3:
      - introsort: keep all 52K
      - heapsort:  cap at 3 × 52K = 156K → keep all 125K (under cap)
      - timsort:   cap at 3 × 52K = 156K → sample 156K from 1.01M

    Result: ~333K rows, roughly 16% intro / 37% heap / 47% tim
    """
    counts = df[label_col].value_counts()
    min_count = counts.min()
    cap = int(min_count * max_ratio)

    parts = []
    for cls in counts.index:
        subset = df[df[label_col] == cls]
        if len(subset) > cap:
            subset = subset.sample(n=cap, random_state=SEED)
        parts.append(subset)

    result = pd.concat(parts, ignore_index=True)
    return result.sample(frac=1, random_state=SEED).reset_index(drop=True)


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights so each class contributes equally to loss."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[yi] for yi in y], dtype=np.float64)


# ── Evaluation helper ────────────────────────────────────────────────────

def evaluate_split(model, X, y_true, le, split_name: str) -> dict:
    """Predict + compute metrics for one split."""
    y_pred_enc = model.predict(X)
    y_pred = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=ALGORITHMS)
    report = classification_report(y_true, y_pred, labels=ALGORITHMS,
                                   output_dict=True, zero_division=0)

    print(f"\n  {split_name}:")
    print(f"    Accuracy:          {acc*100:.1f}%")
    print(f"    Balanced accuracy: {bal_acc*100:.1f}%")
    print(f"    Per-class recall:")
    for algo in ALGORITHMS:
        r = report[algo]["recall"]
        s = report[algo]["support"]
        print(f"      {algo:12s}  recall={r*100:.1f}%  (n={int(s):,})")

    return dict(
        accuracy=round(acc, 4),
        balanced_accuracy=round(bal_acc, 4),
        confusion_matrix=cm.tolist(),
        confusion_labels=ALGORITHMS,
        classification_report=report,
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  STEP 3: Train XGBoost v5 — Balanced Real-World Classifier")
    print("=" * 70)
    t0 = time.time()

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading training dataset...")
    df = pd.read_csv(DATA_CSV)
    df_full = df.copy()  # keep full dataset for final eval
    print(f"  Loaded: {len(df):,} rows")
    print(f"  Class distribution (raw):")
    for algo, cnt in df["best_algorithm"].value_counts().items():
        print(f"    {algo:12s} {cnt:>10,}  ({100*cnt/len(df):.1f}%)")

    # ── 1b. Filter noisy labels ─────────────────────────────────────────
    print("\n[1b/5] Filtering noisy labels...")
    # Keep arrays where winner is ≥5% faster OR array is large (≥2000)
    time_cols = df[["time_introsort", "time_heapsort", "time_timsort"]].values
    sorted_times = np.sort(time_cols, axis=1)
    best_time = sorted_times[:, 0]
    second_time = sorted_times[:, 1]
    margin = (second_time - best_time) / (second_time + 1e-15)
    has_margin = margin >= 0.05
    is_large = df["n_elements"].values >= 2000
    keep = has_margin | is_large
    df = df[keep].reset_index(drop=True)
    print(f"  After filter (margin≥5% OR size≥2K): {len(df):,} rows")
    print(f"  Class distribution (filtered):")
    for algo, cnt in df["best_algorithm"].value_counts().items():
        print(f"    {algo:12s} {cnt:>10,}  ({100*cnt/len(df):.1f}%)")

    # ── 2. Balance ───────────────────────────────────────────────────────
    print("\n[2/5] Balancing dataset (undersample + weights)...")
    df_bal = balanced_undersample(df, "best_algorithm", max_ratio=3.0)
    print(f"  After undersampling: {len(df_bal):,} rows")
    print(f"  Class distribution (balanced):")
    for algo, cnt in df_bal["best_algorithm"].value_counts().items():
        print(f"    {algo:12s} {cnt:>10,}  ({100*cnt/len(df_bal):.1f}%)")

    # ── 3. Split ─────────────────────────────────────────────────────────
    print("\n[3/5] Splitting 70/15/15 (stratified)...")
    X = df_bal[FEATURE_NAMES].values
    y = df_bal["best_algorithm"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # Encode labels
    le = LabelEncoder().fit(ALGORITHMS)
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    # Compute sample weights for training set
    weights = compute_sample_weights(y_train_enc)
    print(f"  Sample weights: min={weights.min():.2f}  max={weights.max():.2f}")

    # ── 4. Train ─────────────────────────────────────────────────────────
    print("\n[4/5] Training XGBoost classifier...")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train_enc,
        sample_weight=weights,
        eval_set=[(X_train, y_train_enc), (X_val, y_val_enc)],
        verbose=50,
    )

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / "xgb_v5.json"))
    print(f"  Model saved: {MODEL_DIR / 'xgb_v5.json'}")

    # ── 5. Evaluate ──────────────────────────────────────────────────────
    print("\n[5/5] Evaluating...")

    results = {}

    # Evaluate on balanced splits
    results["train"] = evaluate_split(model, X_train, y_train, le, "Train")
    results["val"]   = evaluate_split(model, X_val,   y_val,   le, "Validation")
    results["test"]  = evaluate_split(model, X_test,  y_test,  le, "Test")

    # Also evaluate on the FULL unfiltered dataset (real-world performance)
    print("\n  --- Full dataset (unfiltered, real-world view) ---")
    X_full = df_full[FEATURE_NAMES].values
    y_full = df_full["best_algorithm"].values
    results["full_dataset"] = evaluate_split(model, X_full, y_full, le,
                                              "Full Dataset (1.18M)")

    # Save predictions for test split
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    y_test_pred = le.inverse_transform(model.predict(X_test))
    pd.DataFrame({"true": y_test, "pred": y_test_pred}).to_csv(
        RESULTS_DIR / "predictions_test.csv", index=False
    )

    # Feature importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(FEATURE_NAMES, importance.tolist()), key=lambda x: -x[1])

    print(f"\n  Top-8 feature importance:")
    for f, imp in feat_imp[:8]:
        bar = "#" * int(imp * 100)
        print(f"    {f:>22s}: {imp:.4f}  {bar}")

    # ── Save all results ─────────────────────────────────────────────────
    output = dict(
        timestamp=datetime.now().isoformat(),
        xgb_params={k: str(v) if not isinstance(v, (int, float, bool)) else v
                     for k, v in XGB_PARAMS.items()},
        features=FEATURE_NAMES,
        algorithms=ALGORITHMS,
        dataset=dict(
            total_raw=len(df),
            after_undersample=len(df_bal),
            train=len(X_train),
            val=len(X_val),
            test=len(X_test),
        ),
        results=results,
        feature_importance=[dict(feature=f, importance=round(i, 6))
                            for f, i in feat_imp],
    )

    results_file = RESULTS_DIR / "evaluation_results.json"
    results_file.write_text(json.dumps(output, indent=2, default=str))

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  STEP 3 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"  Model:   {MODEL_DIR / 'xgb_v5.json'}")
    print(f"  Results: {results_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
