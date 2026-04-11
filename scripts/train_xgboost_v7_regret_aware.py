#!/usr/bin/env python3
"""
Step 3: Train XGBoost v7 — Regret-Aware Classifier
====================================================
Improvements over v5:
  1. Early stopping (patience=30) — prevents overfitting
  2. Regret-weighted sample weights — misclassifications cost proportional
     to the actual timing penalty, not uniform
  3. Expected-regret inference — at prediction time, pick the algorithm
     that minimises E[regret] rather than argmax(P)
  4. Confidence-threshold fallback — uncertain predictions default to SBS
  5. Lower max_depth (5) + more estimators (1500) for better generalisation
  6. Feature distribution plots use top features by importance (not index)
  7. Cleaned up dead code (TrainingTracker)

Inputs:  data/training_dataset.csv
Outputs: models/xgboost_v7/          (model JSON + cost matrix)
         results/xgboost_v7/         (metrics + figures)

Usage:
    python3 scripts/train_xgboost_v7_regret_aware.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
DATA_CSV    = ROOT / "data" / "training_dataset.csv"
MODEL_DIR   = ROOT / "models" / "xgboost_v7"
RESULTS_DIR = ROOT / "results" / "xgboost_v7"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
TIME_COLS  = ["time_introsort", "time_heapsort", "time_timsort"]
SEED = 42

# ── Hyperparameters (tuned) ─────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=1500,       # more trees, but early stopping will prune
    max_depth=5,             # shallower → better generalisation
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
    early_stopping_rounds=30,  # ← NEW: stop when val loss plateaus
)


# ── Balancing ────────────────────────────────────────────────────────────

def balanced_undersample(df: pd.DataFrame, label_col: str,
                         max_ratio: float = 3.0) -> pd.DataFrame:
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
    return result.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def compute_regret_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Regret-weighted sample weights.

    For each sample, the weight = base_class_weight × (1 + normalised_regret).
    - base_class_weight: inverse-frequency (so minority classes aren't drowned out)
    - normalised_regret: how much timing gap exists between the best and
      second-best algorithm.  Samples where the winner is clear get higher
      weight; ambiguous samples (all algorithms ~equal) get lower weight.

    This teaches the model to focus on samples where getting it wrong is costly.
    """
    times = df[TIME_COLS].values
    best_times = times.min(axis=1)
    sorted_times = np.sort(times, axis=1)
    second_times = sorted_times[:, 1]

    # Regret = (second_best - best) / best — fractional slowdown of a wrong pick
    regret = (second_times - best_times) / (best_times + 1e-15)

    # Normalise regret to [0, 1] range using 95th percentile clipping
    p95 = np.percentile(regret, 95)
    regret_norm = np.clip(regret / (p95 + 1e-15), 0, 1)

    # Inverse-frequency class weights
    labels = df["best_algorithm"].values
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    class_weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    class_weights = np.array([class_weight_map[l] for l in labels], dtype=np.float64)

    # Combined: class balance × (1 + regret importance)
    weights = class_weights * (1.0 + regret_norm)
    return weights


# ── Expected-regret inference ────────────────────────────────────────────

def build_cost_matrix(df: pd.DataFrame, le: LabelEncoder) -> np.ndarray:
    """
    Build a 3×3 cost matrix C where C[i,j] = average extra time (µs) when
    you pick algorithm j but the true best is i.

    Used at inference: pick j that minimises  Σ_i P(class=i) × C[i,j]
    """
    times = df[TIME_COLS].values  # (N, 3)
    labels_enc = le.transform(df["best_algorithm"].values)

    C = np.zeros((3, 3), dtype=np.float64)
    for true_class in range(3):
        mask = labels_enc == true_class
        if mask.sum() == 0:
            continue
        best_time = times[mask, true_class]  # time of the true-best algorithm
        for pred_class in range(3):
            pred_time = times[mask, pred_class]
            # Average regret in µs
            C[true_class, pred_class] = np.mean(pred_time - best_time) * 1e6
    return C


def predict_min_regret(model, X: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """
    Instead of argmax(P), pick the algorithm with lowest expected regret:
        pred_j = argmin_j  Σ_i P(i|x) × C[i,j]
    """
    proba = model.predict_proba(X)  # (N, 3)
    expected_cost = proba @ cost_matrix  # (N, 3)
    return np.argmin(expected_cost, axis=1)


def predict_with_confidence(model, X: np.ndarray, cost_matrix: np.ndarray,
                            sbs_class: int, threshold: float = 0.45) -> np.ndarray:
    """
    If max probability < threshold, fall back to SBS (safest single algorithm).
    Otherwise use expected-regret prediction.
    """
    proba = model.predict_proba(X)
    max_prob = proba.max(axis=1)

    expected_cost = proba @ cost_matrix
    preds = np.argmin(expected_cost, axis=1)

    # Fallback uncertain predictions to SBS
    uncertain = max_prob < threshold
    preds[uncertain] = sbs_class
    return preds, uncertain.sum()


# ── Evaluation helpers ───────────────────────────────────────────────────

def compute_regret_metrics(y_true_enc, y_pred_enc, times, le):
    """Compute regret-based metrics (the ones that actually matter)."""
    n = len(y_true_enc)
    best_times = times.min(axis=1)

    # Model total time
    model_times = np.array([times[i, y_pred_enc[i]] for i in range(n)])
    model_total = model_times.sum()

    # VBS total (oracle)
    vbs_total = best_times.sum()

    # SBS total (always pick the single best)
    sbs_idx = np.argmin(times.sum(axis=0))
    sbs_total = times[:, sbs_idx].sum()

    # Per-instance regret
    regret = model_times - best_times
    regret_us = regret * 1e6

    gap_closed = 1.0 - (model_total - vbs_total) / (sbs_total - vbs_total + 1e-15)
    perfect_picks = (regret < 1e-15).mean()

    return {
        "vbs_total_s": float(vbs_total),
        "sbs_total_s": float(sbs_total),
        "sbs_algorithm": ALGORITHMS[sbs_idx],
        "model_total_s": float(model_total),
        "gap_closed_pct": float(gap_closed * 100),
        "perfect_picks_pct": float(perfect_picks * 100),
        "mean_regret_us": float(regret_us.mean()),
        "median_regret_us": float(np.median(regret_us)),
        "p95_regret_us": float(np.percentile(regret_us, 95)),
        "p99_regret_us": float(np.percentile(regret_us, 99)),
        "max_regret_us": float(regret_us.max()),
    }


# ── Figure generation ────────────────────────────────────────────────────

def plot_training_history(model, output_path):
    """Loss curves from XGBoost eval_result."""
    results = model.evals_result()
    train_loss = results["validation_0"]["mlogloss"]
    val_loss = results["validation_1"]["mlogloss"]
    epochs = list(range(len(train_loss)))

    best_epoch = np.argmin(val_loss)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2, alpha=0.8)
    ax.plot(epochs, val_loss, label="Val Loss", linewidth=2, alpha=0.8)
    ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.6,
               label=f"Best epoch ({best_epoch}, val={val_loss[best_epoch]:.4f})")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss (mlogloss)", fontsize=12, fontweight="bold")
    ax.set_title("Training History with Early Stopping", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_path / "01_training_history.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_roc_curves(y_true, y_proba, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, algo in enumerate(ALGORITHMS):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{algo} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax.set_title("ROC Curves (Test Set)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_path / "02_roc_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pr_curves(y_true, y_proba, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, algo in enumerate(ALGORITHMS):
        y_bin = (y_true == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, i])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, linewidth=2, label=f"{algo} (AP = {pr_auc:.3f})")
    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall Curves (Test Set)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_path / "03_precision_recall_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_confusion_matrices(y_true, y_pred_std, y_pred_regret, output_path):
    """Side-by-side: standard argmax vs regret-aware predictions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    cm1 = confusion_matrix(y_true, y_pred_std)
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues",
                xticklabels=ALGORITHMS, yticklabels=ALGORITHMS, ax=ax1)
    ax1.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax1.set_title("Standard (argmax P)", fontsize=13, fontweight="bold")

    cm2 = confusion_matrix(y_true, y_pred_regret)
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens",
                xticklabels=ALGORITHMS, yticklabels=ALGORITHMS, ax=ax2)
    ax2.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax2.set_title("Regret-Aware (min E[regret])", fontsize=13, fontweight="bold")

    plt.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_path / "04_confusion_matrices.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_feature_importance(model, feature_names, output_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    x_pos = np.arange(len(indices))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    ax.barh(x_pos, importances[indices[::-1]], color=colors)
    ax.set_yticks(x_pos)
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel("Importance (gain)", fontsize=12, fontweight="bold")
    ax.set_title("Feature Importance", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    out = output_path / "05_feature_importance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")
    return indices  # return sorted indices for downstream plots


def plot_feature_distributions(X, y, feature_names, importance_indices, output_path):
    """Box plots of TOP 6 features BY IMPORTANCE (not by index)."""
    top6 = importance_indices[:6]
    top_names = [feature_names[i] for i in top6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (feat_idx, feat_name) in enumerate(zip(top6, top_names)):
        data_by_class = [X[y == i, feat_idx] for i in range(len(ALGORITHMS))]
        bp = axes[idx].boxplot(data_by_class, labels=ALGORITHMS, patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[idx].set_ylabel("Value", fontsize=10, fontweight="bold")
        axes[idx].set_title(feat_name, fontsize=11, fontweight="bold")
        axes[idx].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Feature Distributions (Top 6 by Importance)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_path / "06_feature_distributions.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_regret_comparison(regret_std, regret_regret, regret_conf, output_path):
    """Compare regret distributions across the 3 inference strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    data = [regret_std, regret_regret, regret_conf]
    labels = ["Standard\n(argmax P)", "Regret-Aware\n(min E[regret])", "Confidence\nThreshold"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                    whis=[5, 95])
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Per-Instance Regret (µs)", fontsize=12, fontweight="bold")
    ax.set_title("Regret Distribution: 3 Inference Strategies", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Add mean annotations
    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.annotate(f"mean={mean_val:.3f}µs", xy=(i + 1, mean_val),
                    fontsize=9, ha="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out = output_path / "07_regret_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_confidence_analysis(model, X, y_true, cost_matrix, sbs_class, output_path):
    """Sweep confidence thresholds and plot accuracy/regret trade-off."""
    proba = model.predict_proba(X)
    max_prob = proba.max(axis=1)

    thresholds = np.arange(0.30, 0.80, 0.02)
    accs = []
    fallback_pcts = []

    expected_cost = proba @ cost_matrix

    for t in thresholds:
        preds = np.argmin(expected_cost, axis=1).copy()
        uncertain = max_prob < t
        preds[uncertain] = sbs_class
        accs.append(accuracy_score(y_true, preds) * 100)
        fallback_pcts.append(uncertain.mean() * 100)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(thresholds, accs, "b-o", linewidth=2, markersize=4, label="Accuracy (%)")
    ax1.set_xlabel("Confidence Threshold", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, fallback_pcts, "r--s", linewidth=2, markersize=4, label="Fallback %")
    ax2.set_ylabel("Fallback to SBS (%)", fontsize=12, fontweight="bold", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center right")
    ax1.set_title("Confidence Threshold Sweep", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_path / "08_confidence_threshold_sweep.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_cost_matrix(cost_matrix, output_path):
    """Visualise the learned cost matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cost_matrix, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=[f"pick {a}" for a in ALGORITHMS],
                yticklabels=[f"true={a}" for a in ALGORITHMS], ax=ax)
    ax.set_xlabel("Predicted Algorithm", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Best Algorithm", fontsize=12, fontweight="bold")
    ax.set_title("Cost Matrix — Avg Regret (µs)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_path / "09_cost_matrix.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("XGBOOST v7 — Regret-Aware Classifier")
    print("=" * 80)

    start_time = time.time()

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"\n[LOAD] {DATA_CSV}...")
    df = pd.read_csv(DATA_CSV)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Classes: {dict(df['best_algorithm'].value_counts())}")

    # ── Balance ───────────────────────────────────────────────────────────
    print(f"\n[BALANCE] Undersampling with max_ratio=3.0...")
    df_balanced = balanced_undersample(df, "best_algorithm", max_ratio=3.0)
    print(f"  After undersampling: {len(df_balanced):,} rows")
    print(f"  Classes: {dict(df_balanced['best_algorithm'].value_counts())}")

    # ── Split 70/15/15 ────────────────────────────────────────────────────
    print(f"\n[SPLIT] Stratified 70/15/15...")
    X = df_balanced[FEATURE_NAMES].values
    y = df_balanced["best_algorithm"].values
    times_all = df_balanced[TIME_COLS].values

    X_train, X_temp, y_train, y_temp, t_train, t_temp = train_test_split(
        X, y, times_all, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test, t_val, t_test = train_test_split(
        X_temp, y_temp, t_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ── Encode labels ─────────────────────────────────────────────────────
    le = LabelEncoder().fit(ALGORITHMS)
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    # ── Regret-weighted sample weights ────────────────────────────────────
    print(f"\n[WEIGHTS] Computing regret-weighted sample weights...")
    train_df_for_weights = pd.DataFrame(
        np.column_stack([y_train, t_train]),
        columns=["best_algorithm"] + TIME_COLS,
    )
    # Fix dtypes (column_stack makes everything object)
    for c in TIME_COLS:
        train_df_for_weights[c] = train_df_for_weights[c].astype(float)
    weights = compute_regret_weights(train_df_for_weights)
    print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}], mean={weights.mean():.3f}")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n[TRAIN] XGBoost with early stopping (patience=30)...")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train_enc,
        sample_weight=weights,
        eval_set=[(X_train, y_train_enc), (X_val, y_val_enc)],
        verbose=False,
    )

    best_iter = model.best_iteration
    results_evals = model.evals_result()
    best_val_loss = results_evals["validation_1"]["mlogloss"][best_iter]
    print(f"  Stopped at epoch {best_iter}/{XGB_PARAMS['n_estimators']}")
    print(f"  Best val loss: {best_val_loss:.5f}")
    print(f"  Training time: {time.time() - start_time:.1f}s")

    # ── Build cost matrix ─────────────────────────────────────────────────
    print(f"\n[COST] Building cost matrix from training data...")
    cost_df = pd.DataFrame(
        np.column_stack([y_train, t_train]),
        columns=["best_algorithm"] + TIME_COLS,
    )
    for c in TIME_COLS:
        cost_df[c] = cost_df[c].astype(float)
    cost_matrix = build_cost_matrix(cost_df, le)
    print(f"  Cost matrix (µs):")
    for i, algo in enumerate(ALGORITHMS):
        row = "    " + "  ".join(f"{cost_matrix[i, j]:8.1f}" for j in range(3))
        print(f"  true={algo:10s} → {row}")

    # ── Evaluate 3 strategies ─────────────────────────────────────────────
    print(f"\n[EVAL] Comparing 3 inference strategies on test set...")

    # Strategy 1: standard argmax
    y_pred_std = model.predict(X_test)

    # Strategy 2: expected-regret minimisation
    y_pred_regret = predict_min_regret(model, X_test, cost_matrix)

    # Strategy 3: confidence threshold + regret
    sbs_idx = np.argmin(t_test.sum(axis=0))
    y_pred_conf, n_fallback = predict_with_confidence(
        model, X_test, cost_matrix, sbs_class=sbs_idx, threshold=0.45
    )

    strategies = {
        "standard_argmax": y_pred_std,
        "regret_aware": y_pred_regret,
        "confidence_threshold": y_pred_conf,
    }

    eval_results = {}
    for name, preds in strategies.items():
        acc = accuracy_score(y_test_enc, preds)
        bal_acc = balanced_accuracy_score(y_test_enc, preds)
        regret = compute_regret_metrics(y_test_enc, preds, t_test, le)
        eval_results[name] = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            **regret,
        }
        print(f"\n  [{name}]")
        print(f"    Accuracy:      {acc:.4f}")
        print(f"    Gap closed:    {regret['gap_closed_pct']:.2f}%")
        print(f"    Perfect picks: {regret['perfect_picks_pct']:.2f}%")
        print(f"    Mean regret:   {regret['mean_regret_us']:.3f} µs")
        if name == "confidence_threshold":
            print(f"    Fallbacks:     {n_fallback} ({n_fallback/len(X_test)*100:.1f}%)")

    # Also evaluate on train/val for overfitting check
    for split_name, X_s, y_s, t_s in [
        ("train", X_train, y_train_enc, t_train),
        ("val", X_val, y_val_enc, t_val),
    ]:
        preds = predict_min_regret(model, X_s, cost_matrix)
        acc = accuracy_score(y_s, preds)
        regret = compute_regret_metrics(y_s, preds, t_s, le)
        eval_results[f"{split_name}_regret_aware"] = {
            "accuracy": float(acc),
            "gap_closed_pct": regret["gap_closed_pct"],
            "mean_regret_us": regret["mean_regret_us"],
        }
        print(f"\n  [{split_name} regret-aware]  acc={acc:.4f}  gap_closed={regret['gap_closed_pct']:.2f}%  mean_regret={regret['mean_regret_us']:.3f}µs")

    # ── Figures ───────────────────────────────────────────────────────────
    print(f"\n[FIGURES] Generating...")

    y_proba_test = model.predict_proba(X_test)

    plot_training_history(model, FIGURES_DIR)
    plot_roc_curves(y_test_enc, y_proba_test, FIGURES_DIR)
    plot_pr_curves(y_test_enc, y_proba_test, FIGURES_DIR)
    plot_confusion_matrices(y_test_enc, y_pred_std, y_pred_regret, FIGURES_DIR)
    imp_indices = plot_feature_importance(model, FEATURE_NAMES, FIGURES_DIR)
    plot_feature_distributions(X_test, y_test_enc, FEATURE_NAMES, imp_indices, FIGURES_DIR)

    # Regret distributions for comparison
    regret_std = np.array([(t_test[i, y_pred_std[i]] - t_test[i].min()) * 1e6
                           for i in range(len(y_pred_std))])
    regret_reg = np.array([(t_test[i, y_pred_regret[i]] - t_test[i].min()) * 1e6
                           for i in range(len(y_pred_regret))])
    regret_conf = np.array([(t_test[i, y_pred_conf[i]] - t_test[i].min()) * 1e6
                            for i in range(len(y_pred_conf))])
    plot_regret_comparison(regret_std, regret_reg, regret_conf, FIGURES_DIR)

    plot_confidence_analysis(model, X_test, y_test_enc, cost_matrix, sbs_idx, FIGURES_DIR)
    plot_cost_matrix(cost_matrix, FIGURES_DIR)

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n[SAVE]...")

    model_file = MODEL_DIR / "xgb_v7.json"
    model.get_booster().save_model(str(model_file))
    print(f"  Model: {model_file}")

    cost_file = MODEL_DIR / "cost_matrix.npy"
    np.save(cost_file, cost_matrix)
    print(f"  Cost matrix: {cost_file}")

    # Full results JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "version": "v7_regret_aware",
        "xgb_params": {k: v for k, v in XGB_PARAMS.items()},
        "best_iteration": best_iter,
        "best_val_loss": best_val_loss,
        "features": FEATURE_NAMES,
        "algorithms": ALGORITHMS,
        "dataset": {
            "total_raw": len(df),
            "after_undersample": len(df_balanced),
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "cost_matrix": cost_matrix.tolist(),
        "eval_results": eval_results,
        "feature_importance": [
            {"feature": FEATURE_NAMES[i], "importance": float(model.feature_importances_[i])}
            for i in imp_indices
        ],
    }

    results_file = RESULTS_DIR / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results: {results_file}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"DONE in {elapsed:.1f}s — Figures in {FIGURES_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
