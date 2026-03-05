#!/usr/bin/env python3
"""
Train XGBoost v6 — Honest Diverse Dataset
===========================================
Trains on the new 38K diverse dataset with honest methodology:

  1. Group-aware split: train/test split by source_id so NO leakage
     (all arrays from the same source column stay in the same split)
  2. No heavy undersampling needed — classes already ~18/42/40%
  3. Light sample_weight for the introsort minority
  4. Regret analysis on TEST ONLY (no inflation)

Inputs:  data/diverse_training_data.csv  (from fetch_diverse_data.py)
Outputs: models/xgboost_v6/             (model JSON)
         results/xgboost_v6/            (metrics, predictions, regret)

Usage:
    python3 scripts/train_xgboost_v6.py
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
DATA_CSV    = ROOT / "data" / "diverse_training_data.csv"
MODEL_DIR   = ROOT / "models" / "xgboost_v6"
RESULTS_DIR = ROOT / "results" / "xgboost_v6"

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
SEED = 42

# ── XGBoost hyperparameters ──────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.5,
    reg_lambda=2.0,
    gamma=0.1,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    early_stopping_rounds=50,
)


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights so each class contributes equally to loss."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weight_map = {c: total / (len(classes) * cnt)
                  for c, cnt in zip(classes, counts)}
    return np.array([weight_map[yi] for yi in y], dtype=np.float64)


def evaluate_split(model, X, y_true, le, split_name, df_slice=None):
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
    print(f"    Confusion matrix:")
    print(f"    {'':>12s}  {'pred_intro':>10s} {'pred_heap':>10s} {'pred_tim':>10s}")
    for i, algo in enumerate(ALGORITHMS):
        print(f"    {algo:>12s}  {cm[i][0]:>10,} {cm[i][1]:>10,} {cm[i][2]:>10,}")
    print(f"    Per-class:")
    for algo in ALGORITHMS:
        p = report[algo]["precision"]
        r = report[algo]["recall"]
        s = report[algo]["support"]
        print(f"      {algo:12s}  prec={p*100:.1f}%  recall={r*100:.1f}%  (n={int(s):,})")

    result = dict(
        accuracy=round(acc, 4),
        balanced_accuracy=round(bal_acc, 4),
        confusion_matrix=cm.tolist(),
        confusion_labels=ALGORITHMS,
        classification_report=report,
        n_samples=len(y_true),
    )

    # Regret analysis on this split
    if df_slice is not None and len(df_slice) > 0:
        regret = compute_regret(df_slice, y_pred)
        result["regret"] = regret
        print(f"    Regret: gap_closed={regret['gap_closed_pct']:.1f}%  "
              f"mean_regret={regret['mean_regret_ms']:.4f}ms")

    return result


def compute_regret(df_slice, y_pred):
    """Compute regret statistics — how close to oracle/VBS."""
    time_cols = ["time_introsort", "time_heapsort", "time_timsort"]
    algo_map = {"introsort": 0, "heapsort": 1, "timsort": 2}

    times = df_slice[time_cols].values
    oracle_times = times.min(axis=1)  # VBS: always pick the best

    pred_times = np.array([
        times[i, algo_map[y_pred[i]]] for i in range(len(y_pred))
    ])

    # SBS: single best algorithm (the one with lowest total time)
    total_per_algo = times.sum(axis=0)
    sbs_idx = total_per_algo.argmin()
    sbs_times = times[:, sbs_idx]
    sbs_name = time_cols[sbs_idx].replace("time_", "")

    regret_vs_oracle  = pred_times - oracle_times
    sbs_regret        = sbs_times - oracle_times

    total_oracle  = oracle_times.sum()
    total_pred    = pred_times.sum()
    total_sbs     = sbs_times.sum()

    gap = total_sbs - total_oracle
    closed = gap - (total_pred - total_oracle) if gap > 0 else 0
    gap_closed_pct = (closed / gap * 100) if gap > 0 else 0.0

    return dict(
        sbs_algorithm=sbs_name,
        total_oracle_s=round(total_oracle, 4),
        total_pred_s=round(total_pred, 4),
        total_sbs_s=round(total_sbs, 4),
        gap_closed_pct=round(gap_closed_pct, 2),
        mean_regret_ms=round(regret_vs_oracle.mean() * 1000, 6),
        max_regret_ms=round(regret_vs_oracle.max() * 1000, 6),
        pct_zero_regret=round((regret_vs_oracle == 0).mean() * 100, 2),
        n_arrays=len(y_pred),
    )


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TRAIN XGBoost v6 — Honest Diverse Dataset")
    print("=" * 70)
    t0 = time.time()

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading diverse training dataset...")
    df = pd.read_csv(DATA_CSV)
    print(f"  Loaded: {len(df):,} rows from {df['source_id'].nunique():,} sources")
    print(f"  Domains: {sorted(df['domain'].unique())}")
    print(f"  Class distribution:")
    for algo, cnt in df["best_algorithm"].value_counts().items():
        print(f"    {algo:12s} {cnt:>8,}  ({100*cnt/len(df):.1f}%)")

    # ── 2. Group-aware split by source_id ────────────────────────────────
    print("\n[2/5] Group-aware split (70/15/15 by source_id)...")
    print("  (All arrays from same source stay in same split → no leakage)")

    groups = df["source_id"].values
    X_all = df[FEATURE_NAMES].values
    y_all = df["best_algorithm"].values

    # First split: 70% train+val vs 30% test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    trainval_idx, test_idx = next(gss1.split(X_all, y_all, groups))

    X_trainval = X_all[trainval_idx]
    y_trainval = y_all[trainval_idx]
    groups_trainval = groups[trainval_idx]
    df_trainval = df.iloc[trainval_idx]

    X_test = X_all[test_idx]
    y_test = y_all[test_idx]
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Second split: from trainval, take ~21% as val (= 15% of total)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.21, random_state=SEED)
    train_idx, val_idx = next(gss2.split(X_trainval, y_trainval, groups_trainval))

    X_train = X_trainval[train_idx]
    y_train = y_trainval[train_idx]
    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)

    X_val = X_trainval[val_idx]
    y_val = y_trainval[val_idx]
    df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

    # Verify no source leakage
    train_sources = set(df_train["source_id"])
    val_sources   = set(df_val["source_id"])
    test_sources  = set(df_test["source_id"])
    leak_tv = train_sources & val_sources
    leak_tt = train_sources & test_sources
    leak_vt = val_sources & test_sources

    print(f"  Train: {len(X_train):,} arrays from {len(train_sources):,} sources")
    print(f"  Val:   {len(X_val):,} arrays from {len(val_sources):,} sources")
    print(f"  Test:  {len(X_test):,} arrays from {len(test_sources):,} sources")
    print(f"  Source leakage: train∩val={len(leak_tv)}, "
          f"train∩test={len(leak_tt)}, val∩test={len(leak_vt)}")
    if leak_tv or leak_tt or leak_vt:
        print("  WARNING: Source leakage detected! GroupShuffleSplit should prevent this.")

    print(f"\n  Train class distribution:")
    for algo in ALGORITHMS:
        n = (y_train == algo).sum()
        print(f"    {algo:12s} {n:>8,}  ({100*n/len(y_train):.1f}%)")
    print(f"  Test class distribution:")
    for algo in ALGORITHMS:
        n = (y_test == algo).sum()
        print(f"    {algo:12s} {n:>8,}  ({100*n/len(y_test):.1f}%)")

    # ── 3. Encode + weights ──────────────────────────────────────────────
    print("\n[3/5] Preparing labels and sample weights...")
    le = LabelEncoder().fit(ALGORITHMS)
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    weights = compute_sample_weights(y_train_enc)
    print(f"  Sample weights: min={weights.min():.2f}  max={weights.max():.2f}")

    # ── 4. Train ─────────────────────────────────────────────────────────
    print("\n[4/5] Training XGBoost v6...")
    params = {k: v for k, v in XGB_PARAMS.items() if k != 'early_stopping_rounds'}
    model = xgb.XGBClassifier(
        early_stopping_rounds=XGB_PARAMS['early_stopping_rounds'],
        **params
    )
    model.fit(
        X_train, y_train_enc,
        sample_weight=weights,
        eval_set=[(X_train, y_train_enc), (X_val, y_val_enc)],
        verbose=50,
    )
    best_iter = model.best_iteration
    print(f"  Best iteration: {best_iter}")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / "xgb_v6.json"))
    print(f"  Model saved: {MODEL_DIR / 'xgb_v6.json'}")

    # ── 5. Evaluate ──────────────────────────────────────────────────────
    print("\n[5/5] Evaluating (honest — test sources never seen during training)...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    results["train"] = evaluate_split(model, X_train, y_train, le,
                                       "Train", df_train)
    results["val"]   = evaluate_split(model, X_val, y_val, le,
                                       "Validation", df_val)
    results["test"]  = evaluate_split(model, X_test, y_test, le,
                                       "Test (HONEST — unseen sources)", df_test)

    # Per-domain breakdown on test set
    print("\n  Per-domain test accuracy:")
    y_test_pred = le.inverse_transform(model.predict(X_test))
    domain_results = {}
    for domain in sorted(df_test["domain"].unique()):
        mask = df_test["domain"].values == domain
        if mask.sum() == 0:
            continue
        d_true = y_test[mask]
        d_pred = y_test_pred[mask]
        d_acc = accuracy_score(d_true, d_pred)
        d_bal = balanced_accuracy_score(d_true, d_pred)
        domain_results[domain] = dict(
            n=int(mask.sum()),
            accuracy=round(d_acc, 4),
            balanced_accuracy=round(d_bal, 4),
        )
        print(f"    {domain:<15s}  n={mask.sum():>6,}  "
              f"acc={d_acc*100:.1f}%  bal_acc={d_bal*100:.1f}%")
    results["per_domain_test"] = domain_results

    # Overfitting check
    train_acc = results["train"]["accuracy"]
    test_acc  = results["test"]["accuracy"]
    overfit_gap = train_acc - test_acc
    print(f"\n  Overfitting check:")
    print(f"    Train acc: {train_acc*100:.1f}%")
    print(f"    Test acc:  {test_acc*100:.1f}%")
    print(f"    Gap:       {overfit_gap*100:.1f}pp "
          f"({'OK' if overfit_gap < 0.05 else 'WARNING: overfitting'})")

    # Feature importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(FEATURE_NAMES, importance.tolist()), key=lambda x: -x[1])
    print(f"\n  Feature importance (top 10):")
    for f, imp in feat_imp[:10]:
        bar = "#" * int(imp * 100)
        print(f"    {f:>22s}: {imp:.4f}  {bar}")

    # Save predictions
    pd.DataFrame({
        "file": df_test["file"].values,
        "domain": df_test["domain"].values,
        "source_id": df_test["source_id"].values,
        "n_elements": df_test["n_elements"].values,
        "true": y_test,
        "pred": y_test_pred,
        "correct": (y_test == y_test_pred).astype(int),
    }).to_csv(RESULTS_DIR / "predictions_test.csv", index=False)

    # ── Save all results ─────────────────────────────────────────────────
    output = dict(
        timestamp=datetime.now().isoformat(),
        version="v6",
        dataset=dict(
            source="diverse_training_data.csv",
            total_rows=len(df),
            unique_sources=int(df["source_id"].nunique()),
            domains=sorted(df["domain"].unique().tolist()),
            train_rows=len(X_train),
            val_rows=len(X_val),
            test_rows=len(X_test),
            train_sources=len(train_sources),
            test_sources=len(test_sources),
            source_leakage=dict(
                train_val=len(leak_tv),
                train_test=len(leak_tt),
                val_test=len(leak_vt),
            ),
        ),
        xgb_params={k: str(v) if not isinstance(v, (int, float, bool)) else v
                     for k, v in XGB_PARAMS.items()},
        best_iteration=best_iter,
        features=FEATURE_NAMES,
        algorithms=ALGORITHMS,
        results=results,
        overfit_gap_pp=round(overfit_gap * 100, 2),
        feature_importance=[dict(feature=f, importance=round(i, 6))
                            for f, i in feat_imp],
    )

    results_file = RESULTS_DIR / "evaluation_results.json"
    results_file.write_text(json.dumps(output, indent=2, default=str))

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  XGBoost v6 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
    print(f"  Model:   {MODEL_DIR / 'xgb_v6.json'}")
    print(f"  Results: {results_file}")
    print(f"  Test accuracy: {test_acc*100:.1f}%")
    print(f"  Test gap closed: {results['test']['regret']['gap_closed_pct']:.1f}%")
    print(f"  Overfit gap: {overfit_gap*100:.1f}pp")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
