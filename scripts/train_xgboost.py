#!/usr/bin/env python3
"""
Step 3: XGBoost Multi-Output Regressor for Algorithm Selection
==============================================================

Architecture
------------
- Train one XGBoost regressor per algorithm (3 total).
- Each predicts the sorting time given 16 structural features.
- At inference, pick argmin of the 3 predicted times → selected algorithm.

Evaluation
----------
- Primary metric: per-array prediction accuracy
  (fraction of arrays where selected == best_algorithm)
- Secondary metrics:
  a) Regret = time_selected - time_oracle (μs)
  b) VBS-SBS gap recovery (%)
  c) Per-structure / per-distribution accuracy breakdown
  d) Confusion matrix on algorithm selection

Data splits
-----------
- train.parquet   (216 samples, uniform + normal)
- val.parquet     (72 samples, uniform + normal)
- test_A.parquet  (72 samples, uniform + normal — same distributions)
- test_B.parquet  (360 samples, lognormal + exponential — unseen distributions)
- real-world      (309 truly-real arrays, 7 domains)

Outputs
-------
- models/xgboost_v1/       model artifacts (JSON + metadata)
- results/xgboost_v1/      evaluation CSVs, confusion matrices, plots
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_extraction import FEATURE_NAMES

# ─── Configuration ───────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parent.parent
ALGORITHMS = ["introsort", "heapsort", "timsort"]
TIME_COLS  = [f"time_{a}" for a in ALGORITHMS]
SEED       = 42

# XGBoost hyper-parameters (conservative baseline — tuning comes later)
XGB_PARAMS = dict(
    n_estimators   = 300,
    max_depth      = 6,
    learning_rate  = 0.05,
    subsample      = 0.8,
    colsample_bytree = 0.8,
    reg_alpha      = 0.1,
    reg_lambda     = 1.0,
    random_state   = SEED,
    n_jobs         = -1,
    tree_method    = "hist",          # fast on M4
    objective      = "reg:squarederror",
)

EARLY_STOPPING_ROUNDS = 30

# Output directories — versioned, never overwritten
MODEL_DIR   = ROOT / "models"  / "xgboost_v1"
RESULTS_DIR = ROOT / "results" / "xgboost_v1"


# ─── Helpers ─────────────────────────────────────────────────────────────

def load_split(name: str) -> pd.DataFrame:
    """Load a benchmark split parquet."""
    return pd.read_parquet(ROOT / f"data/benchmark/{name}.parquet")


def load_real_world() -> pd.DataFrame:
    """Load truly-real arrays from v4 combined dataset."""
    df = pd.read_parquet(ROOT / "data/real_world_v4/real_world_v4_combined.parquet")
    return df[~df["domain"].isin(["synthetic", "largescale"])].copy()


def evaluate_selection(
    y_true_times: np.ndarray,   # (N, 3) true times per algorithm
    y_pred_times: np.ndarray,   # (N, 3) predicted times per algorithm
    best_algorithm: pd.Series,  # ground truth names
    label: str,
) -> dict:
    """Compute all evaluation metrics for a split."""
    n = len(y_true_times)

    # Predicted selection: argmin of predicted times
    pred_idx = np.argmin(y_pred_times, axis=1)
    pred_algo = [ALGORITHMS[i] for i in pred_idx]

    # Oracle selection: argmin of true times
    oracle_idx = np.argmin(y_true_times, axis=1)
    oracle_algo = [ALGORITHMS[i] for i in oracle_idx]

    # ── Primary: per-array accuracy ──
    correct = sum(p == t for p, t in zip(pred_algo, oracle_algo))
    accuracy = correct / n

    # ── Regret ──
    oracle_times = y_true_times[np.arange(n), oracle_idx]
    selected_times = y_true_times[np.arange(n), pred_idx]
    regret = selected_times - oracle_times       # per-array (should be >= 0)
    mean_regret   = np.mean(regret)
    median_regret = np.median(regret)
    max_regret    = np.max(regret)

    # ── SBS/VBS gap recovery ──
    # SBS = always pick the algorithm with the lowest mean time
    mean_times = np.mean(y_true_times, axis=0)
    sbs_idx = np.argmin(mean_times)
    sbs_total = np.sum(y_true_times[:, sbs_idx])
    vbs_total = np.sum(oracle_times)
    model_total = np.sum(selected_times)

    sbs_gap_pct = 100.0 * (sbs_total - vbs_total) / vbs_total if vbs_total > 0 else 0.0
    model_gap_pct = 100.0 * (model_total - vbs_total) / vbs_total if vbs_total > 0 else 0.0
    gap_recovery  = 100.0 * (1 - model_gap_pct / sbs_gap_pct) if sbs_gap_pct > 0 else 100.0

    # ── "Acceptable" selections: within 5% of oracle time ──
    acceptable = np.sum(selected_times <= oracle_times * 1.05)
    acceptable_pct = 100.0 * acceptable / n

    # ── Regression quality per algorithm ──
    reg_metrics = {}
    for i, algo in enumerate(ALGORITHMS):
        reg_metrics[algo] = dict(
            mae  = float(mean_absolute_error(y_true_times[:, i], y_pred_times[:, i])),
            rmse = float(np.sqrt(mean_squared_error(y_true_times[:, i], y_pred_times[:, i]))),
            r2   = float(r2_score(y_true_times[:, i], y_pred_times[:, i])),
        )

    # ── Confusion matrix ──
    labels = ALGORITHMS
    cm = confusion_matrix(oracle_algo, pred_algo, labels=labels)

    return dict(
        label          = label,
        n_samples      = n,
        accuracy       = round(accuracy, 4),
        acceptable_pct = round(acceptable_pct, 2),
        mean_regret    = round(float(mean_regret), 6),
        median_regret  = round(float(median_regret), 6),
        max_regret     = round(float(max_regret), 6),
        sbs_algorithm  = ALGORITHMS[sbs_idx],
        sbs_gap_pct    = round(sbs_gap_pct, 2),
        model_gap_pct  = round(model_gap_pct, 2),
        gap_recovery   = round(gap_recovery, 2),
        regression     = reg_metrics,
        confusion_matrix = cm.tolist(),
        confusion_labels = labels,
        pred_algo_dist = {a: pred_algo.count(a) for a in ALGORITHMS},
        true_algo_dist = {a: oracle_algo.count(a) for a in ALGORITHMS},
    )


def breakdown_by_column(
    df: pd.DataFrame,
    y_pred_times: np.ndarray,
    column: str,
) -> dict:
    """Per-group accuracy breakdown by a categorical column."""
    groups = {}
    oracle_idx = np.argmin(df[TIME_COLS].values, axis=1)
    pred_idx   = np.argmin(y_pred_times, axis=1)

    for val, sub in df.groupby(column, observed=True):
        mask = df.index.isin(sub.index)
        n = mask.sum()
        correct = np.sum(oracle_idx[mask] == pred_idx[mask])
        groups[str(val)] = dict(
            n=int(n),
            accuracy=round(correct / n, 4) if n > 0 else 0.0,
        )
    return groups


# ─── Main Training Pipeline ─────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STEP 3: XGBoost Multi-Output Regressor")
    print("=" * 70)
    t0 = time.time()

    # ── 1. Load data ──
    print("\n[1/5] Loading data...")
    train_df = load_split("train")
    val_df   = load_split("val")
    test_a_df = load_split("test_A")
    test_b_df = load_split("test_B")
    real_df   = load_real_world()

    X_train = train_df[FEATURE_NAMES].values
    X_val   = val_df[FEATURE_NAMES].values
    X_test_a = test_a_df[FEATURE_NAMES].values
    X_test_b = test_b_df[FEATURE_NAMES].values
    X_real   = real_df[FEATURE_NAMES].values

    y_train = train_df[TIME_COLS].values
    y_val   = val_df[TIME_COLS].values
    y_test_a = test_a_df[TIME_COLS].values
    y_test_b = test_b_df[TIME_COLS].values
    y_real   = real_df[TIME_COLS].values

    print(f"   Train:    {X_train.shape[0]:>5d} samples")
    print(f"   Val:      {X_val.shape[0]:>5d} samples")
    print(f"   Test A:   {X_test_a.shape[0]:>5d} samples")
    print(f"   Test B:   {X_test_b.shape[0]:>5d} samples")
    print(f"   Real:     {X_real.shape[0]:>5d} samples")

    # ── 2. Train one XGBoost per algorithm ──
    print("\n[2/5] Training XGBoost regressors (3 algorithms)...")
    models = {}
    train_info = {}

    for i, algo in enumerate(ALGORITHMS):
        print(f"\n   Training: {algo}...")
        model = xgb.XGBRegressor(**XGB_PARAMS)

        model.fit(
            X_train, y_train[:, i],
            eval_set=[(X_train, y_train[:, i]), (X_val, y_val[:, i])],
            verbose=False,
        )

        # Best iteration (last, since no early stopping callback that stops)
        train_rmse = np.sqrt(mean_squared_error(y_train[:, i], model.predict(X_train)))
        val_rmse   = np.sqrt(mean_squared_error(y_val[:, i],   model.predict(X_val)))
        val_r2     = r2_score(y_val[:, i], model.predict(X_val))

        train_info[algo] = dict(
            train_rmse = round(float(train_rmse), 6),
            val_rmse   = round(float(val_rmse), 6),
            val_r2     = round(float(val_r2), 4),
            n_estimators_used = model.n_estimators,
        )
        print(f"     train RMSE: {train_rmse:.6f}  |  val RMSE: {val_rmse:.6f}  |  val R²: {val_r2:.4f}")

        models[algo] = model

    # ── 3. Predict on all splits ──
    print("\n[3/5] Generating predictions on all splits...")
    splits = {
        "train":    (train_df,  X_train,  y_train),
        "val":      (val_df,    X_val,    y_val),
        "test_A":   (test_a_df, X_test_a, y_test_a),
        "test_B":   (test_b_df, X_test_b, y_test_b),
        "real":     (real_df,   X_real,   y_real),
    }

    all_results = {}
    for split_name, (df, X, y_true) in splits.items():
        y_pred = np.column_stack([
            models[algo].predict(X) for algo in ALGORITHMS
        ])
        result = evaluate_selection(y_true, y_pred, df["best_algorithm"], split_name)

        # Breakdowns (if columns exist)
        if "structure" in df.columns:
            result["breakdown_structure"] = breakdown_by_column(df, y_pred, "structure")
        if "distribution" in df.columns:
            result["breakdown_distribution"] = breakdown_by_column(df, y_pred, "distribution")
        if "domain" in df.columns:
            result["breakdown_domain"] = breakdown_by_column(df, y_pred, "domain")
        if "n" in df.columns:
            # Bin n into categories
            df_copy = df.copy()
            df_copy["n_bin"] = pd.cut(df_copy["n"], bins=[0, 50_000, 200_000, 1_000_000, float("inf")],
                                       labels=["10K-50K", "50K-200K", "200K-1M", ">1M"])
            result["breakdown_size"] = breakdown_by_column(df_copy, y_pred, "n_bin")

        all_results[split_name] = result

    # ── 4. Print results summary ──
    print("\n" + "=" * 70)
    print("[4/5] RESULTS SUMMARY")
    print("=" * 70)

    for split_name, result in all_results.items():
        print(f"\n--- {split_name.upper()} ({result['n_samples']} samples) ---")
        print(f"   Accuracy:       {result['accuracy']*100:.1f}%")
        print(f"   Acceptable:     {result['acceptable_pct']:.1f}%")
        print(f"   Mean regret:    {result['mean_regret']:.6f}s")
        print(f"   SBS gap:        {result['sbs_gap_pct']:.2f}%")
        print(f"   Model gap:      {result['model_gap_pct']:.2f}%")
        print(f"   Gap recovery:   {result['gap_recovery']:.1f}%")
        print(f"   Predicted dist: {result['pred_algo_dist']}")
        print(f"   True dist:      {result['true_algo_dist']}")

        # Confusion matrix
        cm = np.array(result['confusion_matrix'])
        labels = result['confusion_labels']
        print(f"\n   Confusion matrix (rows=true, cols=predicted):")
        print(f"   {'':>12s}  " + "  ".join(f"{l:>10s}" for l in labels))
        for j, label in enumerate(labels):
            print(f"   {label:>12s}  " + "  ".join(f"{cm[j,k]:>10d}" for k in range(len(labels))))

        # Breakdowns
        for bk in ["breakdown_structure", "breakdown_distribution", "breakdown_domain", "breakdown_size"]:
            if bk in result:
                bk_label = bk.replace("breakdown_", "").capitalize()
                print(f"\n   By {bk_label}:")
                for k, v in sorted(result[bk].items()):
                    print(f"      {k:>20s}: {v['accuracy']*100:5.1f}%  (n={v['n']})")

    # ── 5. Feature importance ──
    print("\n\n--- FEATURE IMPORTANCE (avg gain across 3 models) ---")
    importance_dict = {}
    for algo in ALGORITHMS:
        imp = models[algo].feature_importances_
        for j, feat in enumerate(FEATURE_NAMES):
            importance_dict.setdefault(feat, []).append(imp[j])

    avg_importance = {f: np.mean(v) for f, v in importance_dict.items()}
    sorted_feats = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_feats:
        bar = "█" * int(imp * 200)
        print(f"   {feat:>25s}:  {imp:.4f}  {bar}")

    # ── 6. Save everything ──
    print(f"\n\n[5/5] Saving artifacts...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save models
    for algo in ALGORITHMS:
        model_path = MODEL_DIR / f"xgb_{algo}.json"
        models[algo].save_model(str(model_path))
        print(f"   Saved model: {model_path.relative_to(ROOT)}")

    # Save metadata
    metadata = dict(
        timestamp     = datetime.now().isoformat(),
        seed          = SEED,
        xgb_params    = XGB_PARAMS,
        features      = FEATURE_NAMES,
        algorithms    = ALGORITHMS,
        train_info    = train_info,
        train_size    = X_train.shape[0],
        val_size      = X_val.shape[0],
    )
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"   Saved metadata: models/xgboost_v1/metadata.json")

    # Save results
    results_out = dict(
        timestamp = datetime.now().isoformat(),
        results   = all_results,
        feature_importance = {f: round(float(v), 6) for f, v in sorted_feats},
    )
    (RESULTS_DIR / "evaluation_results.json").write_text(
        json.dumps(results_out, indent=2, default=str)
    )
    print(f"   Saved results: results/xgboost_v1/evaluation_results.json")

    # Save per-split prediction CSVs
    for split_name, (df, X, y_true) in splits.items():
        y_pred = np.column_stack([models[algo].predict(X) for algo in ALGORITHMS])
        pred_idx = np.argmin(y_pred, axis=1)
        oracle_idx = np.argmin(y_true, axis=1)

        id_cols = ["sample_id"] if "sample_id" in df.columns else []
        extra_cols = [c for c in ["domain", "source", "distribution", "structure", "n"] if c in df.columns]
        out = df[id_cols + extra_cols + FEATURE_NAMES + TIME_COLS + ["best_algorithm"]].copy()
        for i, algo in enumerate(ALGORITHMS):
            out[f"pred_time_{algo}"] = y_pred[:, i]
        out["predicted_algorithm"] = [ALGORITHMS[i] for i in pred_idx]
        out["oracle_algorithm"]    = [ALGORITHMS[i] for i in oracle_idx]
        out["correct"]             = out["predicted_algorithm"] == out["oracle_algorithm"]
        out["regret"]              = (
            np.array([y_true[j, pred_idx[j]] for j in range(len(y_true))]) -
            np.array([y_true[j, oracle_idx[j]] for j in range(len(y_true))])
        )

        csv_path = RESULTS_DIR / f"predictions_{split_name}.csv"
        out.to_csv(csv_path, index=False)
        print(f"   Saved predictions: results/xgboost_v1/predictions_{split_name}.csv")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Step 3 complete in {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
