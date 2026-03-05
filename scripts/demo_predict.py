#!/usr/bin/env python3
"""
Demo: Load model → load test-set array → extract features → predict
=====================================================================
Reproduces the exact 70/15/15 split (same seed=42), picks arrays from
the held-out 15% TEST set, loads each raw CSV from disk, extracts all
16 features from scratch, and predicts the best sorting algorithm.

Usage:
    python3 scripts/demo_predict.py                # 5 random test arrays
    python3 scripts/demo_predict.py --n 10         # 10 random test arrays
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
RAW_DIR     = ROOT / "data" / "real_world_10k" / "raw"
DATASET_CSV = ROOT / "data" / "training_dataset.csv"
MODEL_PATH  = ROOT / "models" / "xgboost_v5" / "xgb_v5.json"

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import extract_features, FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
SEED = 42
N_MAX = 100_000


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


def time_sort(arr: np.ndarray, kind: str, repeats: int = 5) -> float:
    """Return best-of-N sort time in seconds."""
    best = float("inf")
    for _ in range(repeats):
        copy = arr.copy()
        t0 = time.perf_counter()
        np.sort(copy, kind=kind)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of test arrays to demo")
    args = parser.parse_args()

    print("=" * 72)
    print("  DEMO: Load model → load array → extract features → predict")
    print("=" * 72)

    # ── 1. Load model ────────────────────────────────────────────────────
    print(f"\n[1] Loading model from {MODEL_PATH.relative_to(ROOT)} ...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    le = LabelEncoder().fit(ALGORITHMS)
    print("    ✓ Model loaded")

    # ── 2. Reproduce the exact test split ────────────────────────────────
    print("\n[2] Reproducing the 70/15/15 split (seed=42) to get test set ...")
    df = pd.read_csv(DATASET_CSV)

    # Same noise filter as training
    time_cols = df[["time_introsort", "time_heapsort", "time_timsort"]].values
    sorted_times = np.sort(time_cols, axis=1)
    margin = (sorted_times[:, 1] - sorted_times[:, 0]) / (sorted_times[:, 1] + 1e-15)
    keep = (margin >= 0.05) | (df["n_elements"].values >= 2000)
    df = df[keep].reset_index(drop=True)

    # Same undersample
    df_bal = balanced_undersample(df, "best_algorithm", max_ratio=3.0)

    # Same split
    _, temp = train_test_split(df_bal, test_size=0.30, stratify=df_bal["best_algorithm"],
                               random_state=SEED)
    _, test_df = train_test_split(temp, test_size=0.50, stratify=temp["best_algorithm"],
                                  random_state=SEED)

    print(f"    ✓ Test set: {len(test_df):,} arrays")
    print(f"    ✓ Test class distribution:")
    for algo, cnt in test_df["best_algorithm"].value_counts().items():
        print(f"        {algo:12s} {cnt:>6,}  ({100*cnt/len(test_df):.1f}%)")

    # ── 3. Pick N random arrays from the test set ────────────────────────
    sample = test_df.sample(n=min(args.n, len(test_df)), random_state=None)
    print(f"\n[3] Picked {len(sample)} random arrays from the test set\n")

    correct = 0
    total = 0

    for idx, (_, row) in enumerate(sample.iterrows(), 1):
        filename = row["file"]
        gt_label = row["best_algorithm"]

        print("─" * 72)
        print(f"  ARRAY {idx}/{len(sample)}: {filename}")
        print(f"  Ground truth (from training data): {gt_label}")
        print("─" * 72)

        # ── Load raw array from disk ─────────────────────────────────────
        fpath = RAW_DIR / filename
        if not fpath.exists():
            print(f"  ✗ File not found on disk — skipping")
            continue

        arr = np.loadtxt(fpath)
        arr = arr[np.isfinite(arr)]
        n = arr.size

        print(f"\n  [LOAD] Loaded raw array from disk")
        print(f"         File:        raw/{filename}")
        print(f"         Elements:    {n:,}")
        print(f"         dtype:       {arr.dtype}")
        print(f"         Min:         {arr.min():.4f}")
        print(f"         Max:         {arr.max():.4f}")
        print(f"         Mean:        {arr.mean():.4f}")
        if n <= 20:
            print(f"         Values:      {arr.tolist()}")
        else:
            print(f"         First 5:     {arr[:5].tolist()}")
            print(f"         Last 5:      {arr[-5:].tolist()}")

        # ── Extract features from scratch ────────────────────────────────
        print(f"\n  [FEATURES] Extracting {len(FEATURE_NAMES)} features from raw array...")
        features = extract_features(arr, N_MAX, filename)

        feature_vec = []
        for fname in FEATURE_NAMES:
            val = features[fname]
            feature_vec.append(val)
            print(f"    {fname:>24s} = {val:.6f}")

        # ── Predict ──────────────────────────────────────────────────────
        X = np.array(feature_vec).reshape(1, -1)
        pred_enc = model.predict(X)[0]
        pred_label = le.inverse_transform([pred_enc])[0]

        proba = model.predict_proba(X)[0]
        proba_labels = le.inverse_transform(range(len(proba)))

        print(f"\n  [PREDICT] Model says:")
        print(f"    ╔══════════════════════════════════════════════════╗")
        print(f"    ║  Best sorting algorithm:  {pred_label:>12s}          ║")
        print(f"    ╚══════════════════════════════════════════════════╝")
        print(f"\n    Confidence:")
        for lbl, p in sorted(zip(proba_labels, proba), key=lambda x: -x[1]):
            bar = "█" * int(p * 40)
            tag = " ◄── predicted" if lbl == pred_label else ""
            print(f"      {lbl:>12s}: {p*100:5.1f}%  {bar}{tag}")

        # ── Verify with actual timing ────────────────────────────────────
        print(f"\n  [VERIFY] Timing all 3 sorts on this array...")
        repeats = 5 if n <= 10_000 else 3
        t_intro = time_sort(arr, "quicksort", repeats)
        t_heap  = time_sort(arr, "heapsort",  repeats)
        t_tim   = time_sort(arr, "stable",    repeats)

        times = {"introsort": t_intro, "heapsort": t_heap, "timsort": t_tim}
        actual_best = min(times, key=times.get)

        print(f"    introsort:   {t_intro*1e6:>10.1f} μs")
        print(f"    heapsort:    {t_heap*1e6:>10.1f} μs")
        print(f"    timsort:     {t_tim*1e6:>10.1f} μs")
        print(f"    Fastest now: {actual_best}")

        is_correct = pred_label == actual_best
        if is_correct:
            print(f"\n    ✅ CORRECT — predicted {pred_label}, actual best = {actual_best}")
            correct += 1
        else:
            penalty = times[pred_label] - times[actual_best]
            print(f"\n    ❌ WRONG — predicted {pred_label}, actual best = {actual_best}")
            print(f"       Time penalty: {penalty*1e6:.1f} μs")
        total += 1
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 72)
    pct = 100 * correct / max(total, 1)
    print(f"  SUMMARY: {correct}/{total} correct ({pct:.0f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
