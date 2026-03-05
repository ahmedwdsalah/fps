#!/usr/bin/env python3
"""
Test XGBoost v5 — Standalone evaluation on the held-out test split
===================================================================
Reproduces the exact same preprocessing + split used in training
(same seed, same filter, same undersample), then loads the saved
model and evaluates on the 15 % test split.

Usage:
    python3 scripts/test_xgboost_v5.py
"""

from __future__ import annotations

import json
import sys
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
MODEL_PATH  = ROOT / "models" / "xgboost_v5" / "xgb_v5.json"

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
SEED = 42


# ── Reproduce the exact same preprocessing as training ───────────────────

def balanced_undersample(df: pd.DataFrame, label_col: str,
                         max_ratio: float = 3.0) -> pd.DataFrame:
    """Same function used during training — must match exactly."""
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


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TEST XGBoost v5 — Independent evaluation on held-out test set")
    print("=" * 70)

    # ── 1. Load dataset ──────────────────────────────────────────────────
    print("\n[1/4] Loading training_dataset.csv ...")
    df = pd.read_csv(DATA_CSV)
    print(f"  Total rows: {len(df):,}")

    # ── 2. Apply same filter ─────────────────────────────────────────────
    print("\n[2/4] Applying same noise filter (margin≥5% OR size≥2K) ...")
    time_cols = df[["time_introsort", "time_heapsort", "time_timsort"]].values
    sorted_times = np.sort(time_cols, axis=1)
    best_time  = sorted_times[:, 0]
    second_time = sorted_times[:, 1]
    margin = (second_time - best_time) / (second_time + 1e-15)
    keep = (margin >= 0.05) | (df["n_elements"].values >= 2000)
    df = df[keep].reset_index(drop=True)
    print(f"  After filter: {len(df):,} rows")

    # ── 3. Apply same undersample + split ────────────────────────────────
    print("\n[3/4] Undersampling + splitting (same seed={}) ...".format(SEED))
    df_bal = balanced_undersample(df, "best_algorithm", max_ratio=3.0)
    print(f"  After undersample: {len(df_bal):,} rows")

    X = df_bal[FEATURE_NAMES].values
    y = df_bal["best_algorithm"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"  Test class distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {u:12s} {c:>8,}  ({100*c/len(y_test):.1f}%)")

    # ── 4. Load model + evaluate ─────────────────────────────────────────
    print(f"\n[4/4] Loading model from {MODEL_PATH.relative_to(ROOT)} ...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    le = LabelEncoder().fit(ALGORITHMS)
    y_pred_enc = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    acc      = accuracy_score(y_test, y_pred)
    bal_acc  = balanced_accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred, labels=ALGORITHMS)
    report   = classification_report(y_test, y_pred, labels=ALGORITHMS,
                                     zero_division=0)

    print("\n" + "─" * 70)
    print("  TEST SET RESULTS")
    print("─" * 70)
    print(f"\n  Accuracy:          {acc*100:.2f}%")
    print(f"  Balanced accuracy: {bal_acc*100:.2f}%")

    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    print(f"  {'':>12s}  {'introsort':>10s}  {'heapsort':>10s}  {'timsort':>10s}")
    for i, algo in enumerate(ALGORITHMS):
        row = "  ".join(f"{cm[i][j]:>10,}" for j in range(3))
        print(f"  {algo:>12s}  {row}")

    print(f"\n  Classification report:")
    print(report)

    # ── Also evaluate on val and full dataset for comparison ─────────────
    print("─" * 70)
    print("  COMPARISON ACROSS SPLITS")
    print("─" * 70)

    splits = [
        ("Train",    X_train, y_train),
        ("Val",      X_val,   y_val),
        ("Test",     X_test,  y_test),
    ]
    print(f"\n  {'Split':<10s}  {'N':>8s}  {'Acc':>7s}  {'BalAcc':>7s}  "
          f"{'intro_r':>7s}  {'heap_r':>7s}  {'tim_r':>7s}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")

    for name, X_s, y_s in splits:
        yp = le.inverse_transform(model.predict(X_s))
        a  = accuracy_score(y_s, yp)
        ba = balanced_accuracy_score(y_s, yp)
        rep = classification_report(y_s, yp, labels=ALGORITHMS,
                                    output_dict=True, zero_division=0)
        ir = rep["introsort"]["recall"]
        hr = rep["heapsort"]["recall"]
        tr = rep["timsort"]["recall"]
        print(f"  {name:<10s}  {len(y_s):>8,}  {a*100:>6.1f}%  {ba*100:>6.1f}%  "
              f"{ir*100:>6.1f}%  {hr*100:>6.1f}%  {tr*100:>6.1f}%")

    # ── Full unfiltered dataset ──────────────────────────────────────────
    print(f"\n  --- Full unfiltered dataset (real-world view) ---")
    df_all = pd.read_csv(DATA_CSV)
    X_full = df_all[FEATURE_NAMES].values
    y_full = df_all["best_algorithm"].values
    yp_full = le.inverse_transform(model.predict(X_full))
    a_full  = accuracy_score(y_full, yp_full)
    ba_full = balanced_accuracy_score(y_full, yp_full)
    rep_full = classification_report(y_full, yp_full, labels=ALGORITHMS,
                                     output_dict=True, zero_division=0)
    print(f"  {'Full':<10s}  {len(y_full):>8,}  {a_full*100:>6.1f}%  {ba_full*100:>6.1f}%  "
          f"{rep_full['introsort']['recall']*100:>6.1f}%  "
          f"{rep_full['heapsort']['recall']*100:>6.1f}%  "
          f"{rep_full['timsort']['recall']*100:>6.1f}%")

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
