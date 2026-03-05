#!/usr/bin/env python3
"""
Stress Test: Download brand-new data and test the model
========================================================
Downloads datasets from domains the model has NEVER seen:
  - Forest Cover Type (581K rows × 54 features) — ecology/geology
  - California Housing (20K rows × 8 features) — census/real estate
  - NASA Exoplanet data (public CSV) — astronomy
  - UCI Wine Quality (public CSV) — chemistry
  - MNIST/Digits pixels — image recognition
  - Synthetic adversarial patterns — edge cases designed to break models

For each array:
  1. Extract 16 features from scratch
  2. Model predicts the best sorting algorithm
  3. Actually time all 3 algorithms to verify
  4. Compute accuracy and regret

Usage:
    python3 scripts/stress_test_new_data.py
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
MODEL_PATH = ROOT / "models" / "xgboost_v5" / "xgb_v5.json"

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import extract_features, FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
N_MAX = 600_000  # max array size in this test


# ── Helpers ───────────────────────────────────────────────────────────────

def time_sort(arr: np.ndarray, kind: str, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(repeats):
        copy = arr.copy()
        gc.disable()
        t0 = time.perf_counter()
        np.sort(copy, kind=kind)
        t1 = time.perf_counter()
        gc.enable()
        best = min(best, t1 - t0)
    return best


def time_all(arr: np.ndarray) -> dict:
    n = len(arr)
    repeats = 5 if n <= 10_000 else 3 if n <= 100_000 else 2
    t_intro = time_sort(arr, "quicksort", repeats)
    t_heap  = time_sort(arr, "heapsort",  repeats)
    t_tim   = time_sort(arr, "stable",    repeats)
    times = {"introsort": t_intro, "heapsort": t_heap, "timsort": t_tim}
    best = min(times, key=times.get)
    return {**times, "best": best}


# ── Data Collection Functions ─────────────────────────────────────────────

def fetch_sklearn_datasets() -> list[tuple[str, str, np.ndarray]]:
    """Fetch large datasets from sklearn — ecology, housing, image domains."""
    arrays = []

    # 1. Forest Cover Type — 581K rows, 54 numeric features (ecology/geology)
    print("    Downloading Forest Cover Type (581K × 54)...")
    try:
        from sklearn.datasets import fetch_covtype
        data = fetch_covtype(as_frame=False)
        X = data.data  # (581012, 54)
        for i in range(X.shape[1]):
            col = X[:, i].astype(np.float64)
            if np.unique(col).size > 10:  # skip binary columns
                arrays.append(("CoverType", f"covtype_col{i}_{len(col)}", col))
        # Also create sub-arrays of different sizes
        for sz in [1000, 5000, 25000, 100000, 500000]:
            if sz <= len(X):
                sub = X[:sz, 0].astype(np.float64)
                arrays.append(("CoverType", f"covtype_elevation_{sz}", sub))
    except Exception as e:
        print(f"    ⚠ Cover Type failed: {e}")

    # 2. California Housing — 20K rows, 8 features (census/real estate)
    print("    Downloading California Housing (20K × 8)...")
    try:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=False)
        X = data.data
        names = data.feature_names
        for i, name in enumerate(names):
            col = X[:, i].astype(np.float64)
            arrays.append(("Housing", f"housing_{name}_{len(col)}", col))
    except Exception as e:
        print(f"    ⚠ California Housing failed: {e}")

    # 3. Digits — 1797 images, 64 pixel features (image recognition)
    print("    Downloading Digits (1797 × 64)...")
    try:
        from sklearn.datasets import load_digits
        data = load_digits()
        X = data.data
        # Flatten all images into one big array
        flat = X.flatten().astype(np.float64)  # 115,008 values
        arrays.append(("Digits", f"digits_all_pixels_{len(flat)}", flat))
        # Individual rows (small arrays)
        for i in range(0, 100, 10):
            arrays.append(("Digits", f"digits_img{i}_64", X[i].astype(np.float64)))
    except Exception as e:
        print(f"    ⚠ Digits failed: {e}")

    return arrays


def fetch_web_datasets() -> list[tuple[str, str, np.ndarray]]:
    """Fetch datasets from public URLs — wine, air quality, etc."""
    arrays = []

    # 1. Wine Quality (UCI) — chemistry measurements
    print("    Downloading Wine Quality (UCI)...")
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        df = pd.read_csv(url, sep=";")
        for col in df.select_dtypes(include=[np.number]).columns:
            arr = df[col].dropna().values.astype(np.float64)
            if len(arr) > 50:
                arrays.append(("Wine", f"wine_{col.replace(' ', '_')}_{len(arr)}", arr))
    except Exception as e:
        print(f"    ⚠ Wine Quality failed: {e}")

    # 2. Iris (classic — tiny but different domain)
    print("    Downloading Iris...")
    try:
        from sklearn.datasets import load_iris
        data = load_iris()
        X = data.data
        for i, name in enumerate(data.feature_names):
            col = X[:, i].astype(np.float64)
            short = name.replace(" ", "_").replace("(cm)", "").strip("_")
            arrays.append(("Iris", f"iris_{short}_{len(col)}", col))
    except Exception as e:
        print(f"    ⚠ Iris failed: {e}")

    # 3. Breast Cancer (medical domain)
    print("    Downloading Breast Cancer (medical)...")
    try:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = data.data
        for i in [0, 1, 2, 3, 20, 21, 22, 23]:  # key features
            col = X[:, i].astype(np.float64)
            name = data.feature_names[i].replace(" ", "_")
            arrays.append(("Medical", f"cancer_{name}_{len(col)}", col))
    except Exception as e:
        print(f"    ⚠ Breast Cancer failed: {e}")

    return arrays


def generate_adversarial_arrays() -> list[tuple[str, str, np.ndarray]]:
    """
    Craft arrays specifically designed to be HARD for the model:
    - Near-sorted with a twist
    - Pipe organ patterns
    - Sawtooth waves
    - Heavy-tail distributions
    - Massive duplicates
    - Nearly identical timing territory
    """
    rng = np.random.RandomState(99)
    arrays = []

    for n in [500, 5000, 50000, 200000]:
        tag = f"{n//1000}K" if n >= 1000 else str(n)

        # 1. Pipe organ: [1,2,3,...,n/2,...,3,2,1] — confuses run detection
        half = np.arange(n // 2, dtype=np.float64)
        pipe = np.concatenate([half, half[::-1]])
        arrays.append(("Adversarial", f"pipe_organ_{tag}", pipe))

        # 2. Sawtooth: repeating sorted segments
        seg = 100
        saw = np.tile(np.arange(seg, dtype=np.float64), n // seg + 1)[:n]
        arrays.append(("Adversarial", f"sawtooth_{tag}", saw))

        # 3. 95% sorted + 5% random swaps — nearly sorted but not quite
        nearly = np.arange(n, dtype=np.float64)
        swap_idx = rng.choice(n, size=n // 20, replace=False)
        nearly[swap_idx] = rng.uniform(0, n, size=len(swap_idx))
        arrays.append(("Adversarial", f"nearly_sorted_95pct_{tag}", nearly))

        # 4. Zipf distribution — extreme heavy tail
        zipf = rng.zipf(1.5, size=n).astype(np.float64)
        arrays.append(("Adversarial", f"zipf_{tag}", zipf))

        # 5. 99% duplicates — only 1% unique values
        vals = rng.choice(max(n // 100, 2), size=n).astype(np.float64)
        arrays.append(("Adversarial", f"99pct_dupes_{tag}", vals))

        # 6. Sorted then reversed last quarter
        arr = np.arange(n, dtype=np.float64)
        arr[3 * n // 4:] = arr[3 * n // 4:][::-1]
        arrays.append(("Adversarial", f"sorted_rev_quarter_{tag}", arr))

        # 7. Exponential with outliers
        exp = rng.exponential(1.0, size=n)
        exp[rng.choice(n, 10)] = 1e6  # inject extreme outliers
        arrays.append(("Adversarial", f"exp_outliers_{tag}", exp))

        # 8. Interleaved sorted sequences: [1,100,2,101,3,102,...]
        a = np.arange(0, n, 2, dtype=np.float64)
        b = np.arange(n, 2 * n, 2, dtype=np.float64)
        interleaved = np.empty(n, dtype=np.float64)
        half_n = min(len(a), n // 2)
        interleaved[0::2] = a[:half_n] if len(a) >= half_n else np.resize(a, half_n)
        interleaved[1::2] = b[:n - half_n] if len(b) >= (n - half_n) else np.resize(b, n - half_n)
        arrays.append(("Adversarial", f"interleaved_{tag}", interleaved))

    return arrays


def generate_large_stress_arrays() -> list[tuple[str, str, np.ndarray]]:
    """Very large arrays (100K–500K) with various patterns to stress timing."""
    rng = np.random.RandomState(77)
    arrays = []

    for n in [100_000, 250_000, 500_000]:
        tag = f"{n//1000}K"

        # Pure random (uniform) — introsort territory
        arrays.append(("LargeStress", f"uniform_random_{tag}",
                       rng.uniform(0, 1e6, n)))

        # Random normal
        arrays.append(("LargeStress", f"normal_{tag}",
                       rng.normal(0, 1, n)))

        # Fully sorted — timsort paradise
        arrays.append(("LargeStress", f"fully_sorted_{tag}",
                       np.arange(n, dtype=np.float64)))

        # Reverse sorted
        arrays.append(("LargeStress", f"reverse_sorted_{tag}",
                       np.arange(n, 0, -1, dtype=np.float64)))

        # 50 unique values (lots of duplicates) — heapsort territory
        arrays.append(("LargeStress", f"50_uniques_{tag}",
                       rng.choice(50, size=n).astype(np.float64)))

    return arrays


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  STRESS TEST: Brand-New Data the Model Has Never Seen")
    print("=" * 72)

    # ── 1. Load model ────────────────────────────────────────────────────
    print("\n[1/4] Loading XGBoost v5 model...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    le = LabelEncoder().fit(ALGORITHMS)
    print("  ✓ Model loaded")

    # ── 2. Collect test arrays ───────────────────────────────────────────
    print("\n[2/4] Collecting test arrays from new domains...\n")

    all_arrays = []

    print("  ── sklearn datasets (ecology, housing, digits) ──")
    all_arrays += fetch_sklearn_datasets()

    print("\n  ── Web datasets (wine, medical) ──")
    all_arrays += fetch_web_datasets()

    print("\n  ── Adversarial patterns (designed to break models) ──")
    all_arrays += generate_adversarial_arrays()

    print("\n  ── Large stress arrays (100K–500K) ──")
    all_arrays += generate_large_stress_arrays()

    print(f"\n  Total test arrays: {len(all_arrays)}")
    domains = {}
    for domain, name, arr in all_arrays:
        domains[domain] = domains.get(domain, 0) + 1
    for d, c in sorted(domains.items()):
        print(f"    {d:>15s}: {c:>4d} arrays")

    # ── 3. Test each array ───────────────────────────────────────────────
    print(f"\n[3/4] Testing model on {len(all_arrays)} arrays...\n")

    results = []
    t_start = time.time()

    for i, (domain, name, arr) in enumerate(all_arrays):
        n = len(arr)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 50:
            continue

        # Extract features
        features = extract_features(arr, N_MAX, name)
        feature_vec = [features[f] for f in FEATURE_NAMES]
        X = np.array(feature_vec).reshape(1, -1)

        # Predict
        pred_enc = model.predict(X)[0]
        pred_label = le.inverse_transform([pred_enc])[0]
        proba = model.predict_proba(X)[0]
        confidence = proba.max()

        # Time all 3 algorithms
        times = time_all(arr)
        actual_best = times["best"]
        is_correct = pred_label == actual_best

        # Regret
        pred_time = times[pred_label]
        best_time = times[actual_best]
        regret_us = (pred_time - best_time) * 1e6

        results.append({
            "domain": domain,
            "name": name,
            "n": len(arr),
            "predicted": pred_label,
            "actual": actual_best,
            "correct": is_correct,
            "confidence": confidence,
            "regret_us": regret_us,
            "time_introsort": times["introsort"],
            "time_heapsort": times["heapsort"],
            "time_timsort": times["timsort"],
        })

        # Progress
        symbol = "✅" if is_correct else "❌"
        if (i + 1) % 10 == 0 or not is_correct or n >= 50000:
            print(f"  [{i+1:>4d}/{len(all_arrays)}] {symbol} {name:>40s}  "
                  f"n={n:>7,}  pred={pred_label:>10s}  "
                  f"actual={actual_best:>10s}  regret={regret_us:>8.1f}μs  "
                  f"conf={confidence:.2f}")

    elapsed = time.time() - t_start

    # ── 4. Analysis ──────────────────────────────────────────────────────
    df = pd.DataFrame(results)

    print(f"\n{'='*72}")
    print(f"  STRESS TEST RESULTS")
    print(f"{'='*72}")
    print(f"\n  Total arrays tested: {len(df)}")
    print(f"  Time: {elapsed:.1f}s")

    # Overall accuracy
    acc = df["correct"].mean() * 100
    print(f"\n  ── OVERALL ──")
    print(f"  Accuracy: {acc:.1f}% ({df['correct'].sum()}/{len(df)})")
    print(f"  Mean regret: {df['regret_us'].mean():.2f} μs")
    print(f"  Median regret: {df['regret_us'].median():.2f} μs")
    print(f"  Max regret: {df['regret_us'].max():.1f} μs")
    print(f"  Zero regret: {(df['regret_us'] == 0).sum()}/{len(df)} "
          f"({(df['regret_us'] == 0).mean()*100:.1f}%)")

    # Per-domain accuracy
    print(f"\n  ── BY DOMAIN ──")
    print(f"  {'Domain':>15s}  {'N':>5s}  {'Acc':>7s}  {'Mean Regret':>12s}  "
          f"{'Max Regret':>11s}  {'Zero%':>6s}")
    print(f"  {'─'*15}  {'─'*5}  {'─'*7}  {'─'*12}  {'─'*11}  {'─'*6}")
    for domain in sorted(df["domain"].unique()):
        sub = df[df["domain"] == domain]
        a = sub["correct"].mean() * 100
        mr = sub["regret_us"].mean()
        mx = sub["regret_us"].max()
        zr = (sub["regret_us"] == 0).mean() * 100
        print(f"  {domain:>15s}  {len(sub):>5d}  {a:>6.1f}%  "
              f"{mr:>10.2f}μs  {mx:>9.1f}μs  {zr:>5.1f}%")

    # By size bucket
    print(f"\n  ── BY ARRAY SIZE ──")
    print(f"  {'Size':>12s}  {'N':>5s}  {'Acc':>7s}  {'Mean Regret':>12s}")
    print(f"  {'─'*12}  {'─'*5}  {'─'*7}  {'─'*12}")
    for label, lo, hi in [("< 500", 0, 500), ("500–5K", 500, 5000),
                           ("5K–50K", 5000, 50000), ("50K–200K", 50000, 200000),
                           ("200K+", 200000, 999999999)]:
        sub = df[(df["n"] >= lo) & (df["n"] < hi)]
        if len(sub) == 0:
            continue
        a = sub["correct"].mean() * 100
        mr = sub["regret_us"].mean()
        print(f"  {label:>12s}  {len(sub):>5d}  {a:>6.1f}%  {mr:>10.2f}μs")

    # Confusion matrix
    print(f"\n  ── CONFUSION (what model predicts vs actual best) ──")
    print(f"  {'':>12s}  {'→ introsort':>12s}  {'→ heapsort':>12s}  {'→ timsort':>12s}")
    for actual in ALGORITHMS:
        row = []
        for pred in ALGORITHMS:
            c = len(df[(df["actual"] == actual) & (df["predicted"] == pred)])
            row.append(c)
        print(f"  {actual:>12s}  {row[0]:>12d}  {row[1]:>12d}  {row[2]:>12d}")

    # Regret analysis
    print(f"\n  ── REGRET ANALYSIS ──")
    vbs_total = df[["time_introsort", "time_heapsort", "time_timsort"]].min(axis=1).sum()
    model_total = sum(
        df.iloc[i][f"time_{df.iloc[i]['predicted']}"]
        for i in range(len(df))
    )
    sbs_totals = {a: df[f"time_{a}"].sum() for a in ALGORITHMS}
    sbs_algo = min(sbs_totals, key=sbs_totals.get)
    sbs_total = sbs_totals[sbs_algo]

    gap = sbs_total - vbs_total
    if gap > 0:
        gap_closed = (1 - (model_total - vbs_total) / gap) * 100
    else:
        gap_closed = 100.0

    print(f"  VBS (oracle):     {vbs_total:.4f}s")
    print(f"  Model (XGBv5):    {model_total:.4f}s")
    print(f"  SBS ({sbs_algo:>9s}):  {sbs_total:.4f}s")
    print(f"  VBS-SBS Gap:      {100*gap/sbs_total:.1f}%")
    print(f"  Gap closed:       {gap_closed:.1f}%")

    # Worst misses
    wrong = df[~df["correct"]].sort_values("regret_us", ascending=False)
    if len(wrong) > 0:
        print(f"\n  ── TOP 10 WORST MISSES ──")
        print(f"  {'Name':>40s}  {'N':>8s}  {'Pred':>10s}  {'Actual':>10s}  {'Regret':>10s}")
        print(f"  {'─'*40}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")
        for _, row in wrong.head(10).iterrows():
            print(f"  {row['name']:>40s}  {row['n']:>8,}  {row['predicted']:>10s}  "
                  f"{row['actual']:>10s}  {row['regret_us']:>8.1f}μs")

    # Save results
    out_dir = ROOT / "results" / "stress_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "stress_test_results.csv", index=False)
    print(f"\n  Results saved: results/stress_test/stress_test_results.csv")

    print(f"\n{'='*72}")


if __name__ == "__main__":
    main()
