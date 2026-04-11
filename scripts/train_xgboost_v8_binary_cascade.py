#!/usr/bin/env python3
"""
XGBoost v8 — Binary Cascade Classifier
========================================
v7 analysis showed introsort↔heapsort are interchangeable (cost ≈ -1.5µs).
The only decision that matters: timsort vs non-timsort (36-50µs penalty).

Architecture:
  Stage 1: Binary XGBoost — timsort vs {introsort+heapsort}
  Stage 2: For non-timsort predictions, pick heapsort (SBS of the pair)
           OR optionally train a second binary classifier

Keeps v5's proven setup: max_depth=7, margin filter removed.
Adds: early stopping, regret evaluation, comprehensive figures.

Inputs:  data/training_dataset.csv
Outputs: models/xgboost_v8/
         results/xgboost_v8/

Usage:
    python3 scripts/train_xgboost_v8_binary_cascade.py
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
    f1_score,
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
MODEL_DIR   = ROOT / "models" / "xgboost_v8"
RESULTS_DIR = ROOT / "results" / "xgboost_v8"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
TIME_COLS  = ["time_introsort", "time_heapsort", "time_timsort"]
SEED = 42

# ── Stage 1 params (binary: timsort vs rest) ────────────────────────────
STAGE1_PARAMS = dict(
    n_estimators=1000,
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
    objective="binary:logistic",
    eval_metric="logloss",
    early_stopping_rounds=30,
)

# ── Stage 2 params (binary: introsort vs heapsort, for non-timsort) ─────
STAGE2_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
    objective="binary:logistic",
    eval_metric="logloss",
    early_stopping_rounds=20,
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


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[yi] for yi in y], dtype=np.float64)


# ── Regret evaluation ───────────────────────────────────────────────────

def compute_regret_metrics(y_pred_3class, times):
    """Compute regret using 3-class predictions and actual timing data."""
    n = len(y_pred_3class)
    best_times = times.min(axis=1)

    model_times = np.array([times[i, y_pred_3class[i]] for i in range(n)])
    model_total = model_times.sum()
    vbs_total = best_times.sum()

    sbs_idx = np.argmin(times.sum(axis=0))
    sbs_total = times[:, sbs_idx].sum()

    regret = model_times - best_times
    regret_us = regret * 1e6

    gap_closed = 1.0 - (model_total - vbs_total) / (sbs_total - vbs_total + 1e-15)
    perfect_picks = (regret < 1e-15).mean()

    return {
        "vbs_total_s": round(float(vbs_total), 6),
        "sbs_total_s": round(float(sbs_total), 6),
        "sbs_algorithm": ALGORITHMS[sbs_idx],
        "model_total_s": round(float(model_total), 6),
        "gap_closed_pct": round(float(gap_closed * 100), 4),
        "perfect_picks_pct": round(float(perfect_picks * 100), 4),
        "mean_regret_us": round(float(regret_us.mean()), 4),
        "median_regret_us": round(float(np.median(regret_us)), 4),
        "p95_regret_us": round(float(np.percentile(regret_us, 95)), 4),
        "p99_regret_us": round(float(np.percentile(regret_us, 99)), 4),
        "max_regret_us": round(float(regret_us.max()), 4),
    }


# ── Cascade prediction ──────────────────────────────────────────────────

def cascade_predict(stage1_model, stage2_model, X, default_non_timsort="heapsort"):
    """
    Stage 1: predict timsort (1) vs non-timsort (0)
    Stage 2: for non-timsort, predict introsort (0) vs heapsort (1)

    Returns 3-class predictions: 0=introsort, 1=heapsort, 2=timsort
    """
    # Stage 1: binary timsort vs rest
    s1_pred = stage1_model.predict(X)  # 1=timsort, 0=non-timsort
    s1_proba = stage1_model.predict_proba(X)[:, 1]  # P(timsort)

    # Start with all predictions
    preds_3class = np.full(len(X), -1, dtype=int)

    # Timsort predictions
    timsort_mask = s1_pred == 1
    preds_3class[timsort_mask] = 2  # timsort

    # Non-timsort: use stage 2
    non_timsort_mask = ~timsort_mask
    if non_timsort_mask.sum() > 0 and stage2_model is not None:
        X_nontim = X[non_timsort_mask]
        s2_pred = stage2_model.predict(X_nontim)  # 0=introsort, 1=heapsort
        preds_3class[non_timsort_mask] = s2_pred  # 0 or 1 maps directly
    elif non_timsort_mask.sum() > 0:
        # Fallback: always pick heapsort (SBS of the pair)
        preds_3class[non_timsort_mask] = 1

    return preds_3class, s1_proba


def cascade_predict_heapsort_default(stage1_model, X):
    """Simpler cascade: timsort vs heapsort only (no stage 2)."""
    s1_pred = stage1_model.predict(X)
    preds_3class = np.where(s1_pred == 1, 2, 1)  # timsort=2, heapsort=1
    s1_proba = stage1_model.predict_proba(X)[:, 1]
    return preds_3class, s1_proba


# ── Figures ──────────────────────────────────────────────────────────────

def plot_training_history(model, name, output_path, filename):
    results = model.evals_result()
    train_loss = results["validation_0"]["logloss"]
    val_loss = results["validation_1"]["logloss"]
    epochs = list(range(len(train_loss)))
    best_epoch = np.argmin(val_loss)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2, alpha=0.8)
    ax.plot(epochs, val_loss, label="Val Loss", linewidth=2, alpha=0.8)
    ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.6,
               label=f"Best epoch ({best_epoch}, val={val_loss[best_epoch]:.4f})")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss (logloss)", fontsize=12, fontweight="bold")
    ax.set_title(f"Training History: {name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_path / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {out}")


def plot_stage1_roc(y_true, y_proba, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2.5, color="#2ca02c",
            label=f"Timsort vs Rest (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("Stage 1: ROC — Timsort vs Non-Timsort", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_path / "02_stage1_roc.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {out}")
    return roc_auc


def plot_stage2_roc(y_true, y_proba, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2.5, color="#1f77b4",
            label=f"Introsort vs Heapsort (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("Stage 2: ROC — Introsort vs Heapsort", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_path / "03_stage2_roc.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {out}")
    return roc_auc


def plot_3class_confusion(y_true_3class, preds_cascade, preds_heapdefault, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    cm1 = confusion_matrix(y_true_3class, preds_cascade, labels=[0, 1, 2])
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues",
                xticklabels=ALGORITHMS, yticklabels=ALGORITHMS, ax=ax1)
    ax1.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax1.set_title("Full Cascade (Stage1 + Stage2)", fontsize=13, fontweight="bold")

    cm2 = confusion_matrix(y_true_3class, preds_heapdefault, labels=[0, 1, 2])
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens",
                xticklabels=ALGORITHMS, yticklabels=ALGORITHMS, ax=ax2)
    ax2.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax2.set_title("Stage1 + Heapsort Default", fontsize=13, fontweight="bold")

    plt.suptitle("3-Class Confusion Matrices — Test Set", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_path / "04_confusion_matrices.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {out}")


def plot_feature_importance_both(s1_model, s2_model, feature_names, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    for ax, model, title in [
        (ax1, s1_model, "Stage 1: Timsort vs Rest"),
        (ax2, s2_model, "Stage 2: Introsort vs Heapsort"),
    ]:
        imp = model.feature_importances_
        indices = np.argsort(imp)[::-1]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
        x_pos = np.arange(len(indices))
        ax.barh(x_pos, imp[indices[::-1]], color=colors)
        ax.set_yticks(x_pos)
        ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
        ax.set_xlabel("Importance (gain)", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Feature Importance — Both Stages", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = output_path / "05_feature_importance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {out}")


def plot_regret_comparison_all(regret_dict, output_path):
    """Compare regret across strategies."""
    fig, ax = plt.subplots(figsize=(14, 6))

    names = list(regret_dict.keys())
    data = [regret_dict[n] for n in names]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    bp = ax.boxplot(data, labels=[n.replace("_", "\n") for n in names],
                    patch_artist=True, showfliers=False, whis=[5, 95])
    for patch, color in zip(bp["boxes"], colors[:len(names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Per-Instance Regret (µs)", fontsize=12, fontweight="bold")
    ax.set_title("Regret Distribution: v8 Strategies vs v5 Baseline", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for i, d in enumerate(data):
        mean_val = np.mean(d)
        ax.annotate(f"mean={mean_val:.3f}µs", xy=(i + 1, mean_val),
                    fontsize=9, ha="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out = output_path / "06_regret_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {out}")


def plot_timsort_probability_distribution(s1_proba, y_true_binary, output_path):
    """Show how well Stage 1 separates timsort from rest."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(s1_proba[y_true_binary == 0], bins=50, alpha=0.6, color="#1f77b4",
            label="Non-Timsort (true)", density=True)
    ax.hist(s1_proba[y_true_binary == 1], bins=50, alpha=0.6, color="#2ca02c",
            label="Timsort (true)", density=True)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=2, label="Decision boundary")
    ax.set_xlabel("P(timsort)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax.set_title("Stage 1: Timsort Probability Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_path / "07_timsort_probability_dist.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {out}")


def plot_threshold_sweep(stage1_model, stage2_model, X_test, y_test_3class, t_test, output_path):
    """Sweep Stage 1 threshold and plot gap_closed vs threshold."""
    proba_tim = stage1_model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.20, 0.80, 0.02)
    gap_closeds = []
    mean_regrets = []
    timsort_pcts = []

    for t in thresholds:
        s1_pred = (proba_tim >= t).astype(int)
        preds_3class = np.full(len(X_test), -1, dtype=int)
        timsort_mask = s1_pred == 1
        preds_3class[timsort_mask] = 2

        non_tim = ~timsort_mask
        if non_tim.sum() > 0 and stage2_model is not None:
            s2_pred = stage2_model.predict(X_test[non_tim])
            preds_3class[non_tim] = s2_pred
        else:
            preds_3class[non_tim] = 1

        metrics = compute_regret_metrics(preds_3class, t_test)
        gap_closeds.append(metrics["gap_closed_pct"])
        mean_regrets.append(metrics["mean_regret_us"])
        timsort_pcts.append(timsort_mask.mean() * 100)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(thresholds, gap_closeds, "b-o", linewidth=2, markersize=4, label="Gap Closed (%)")
    ax1.set_xlabel("P(timsort) Threshold", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Gap Closed (%)", fontsize=12, fontweight="bold", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, mean_regrets, "r--s", linewidth=2, markersize=4, label="Mean Regret (µs)")
    ax2.set_ylabel("Mean Regret (µs)", fontsize=12, fontweight="bold", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="center right")
    ax1.set_title("Stage 1 Threshold Sweep — Gap Closed vs Regret", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_path / "08_threshold_sweep.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("XGBOOST v8 — Binary Cascade Classifier")
    print("  Stage 1: Timsort vs {Introsort + Heapsort}")
    print("  Stage 2: Introsort vs Heapsort")
    print("=" * 80)

    t0 = time.time()

    # ── Load ──────────────────────────────────────────────────────────────
    print(f"\n[LOAD] {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    print(f"  {len(df):,} rows")
    print(f"  3-class: {dict(df['best_algorithm'].value_counts())}")

    # ── Balance (same as v5) ──────────────────────────────────────────────
    print(f"\n[BALANCE] max_ratio=3.0")
    df_bal = balanced_undersample(df, "best_algorithm", max_ratio=3.0)
    print(f"  {len(df_bal):,} rows after undersample")

    # ── Split 70/15/15 ────────────────────────────────────────────────────
    X = df_bal[FEATURE_NAMES].values
    y_3class = df_bal["best_algorithm"].values
    times_all = df_bal[TIME_COLS].values

    X_train, X_temp, y_train, y_temp, t_train, t_temp = train_test_split(
        X, y_3class, times_all, test_size=0.30, stratify=y_3class, random_state=SEED
    )
    X_val, X_test, y_val, y_temp2, t_val, t_test = train_test_split(
        X_temp, y_temp, t_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    print(f"  Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    # ── Encode ────────────────────────────────────────────────────────────
    le = LabelEncoder().fit(ALGORITHMS)
    y_train_3 = le.transform(y_train)  # 0=introsort, 1=heapsort, 2=timsort
    y_val_3   = le.transform(y_val)
    y_test_3  = le.transform(y_temp2)

    # Binary labels: 1=timsort, 0=non-timsort
    y_train_bin = (y_train_3 == 2).astype(int)
    y_val_bin   = (y_val_3 == 2).astype(int)
    y_test_bin  = (y_test_3 == 2).astype(int)

    # ── STAGE 1: Timsort vs Rest ──────────────────────────────────────────
    print(f"\n[STAGE 1] Timsort vs Non-Timsort")
    print(f"  Train: {y_train_bin.sum():,} timsort / {(~y_train_bin.astype(bool)).sum():,} rest")

    w1 = compute_sample_weights(y_train_bin)
    s1_model = xgb.XGBClassifier(**STAGE1_PARAMS)
    s1_model.fit(
        X_train, y_train_bin,
        sample_weight=w1,
        eval_set=[(X_train, y_train_bin), (X_val, y_val_bin)],
        verbose=False,
    )
    print(f"  Stopped: epoch {s1_model.best_iteration}/{STAGE1_PARAMS['n_estimators']}")
    s1_val_pred = s1_model.predict(X_val)
    s1_val_acc = accuracy_score(y_val_bin, s1_val_pred)
    s1_val_f1 = f1_score(y_val_bin, s1_val_pred)
    print(f"  Val acc={s1_val_acc:.4f}  F1={s1_val_f1:.4f}")

    # ── STAGE 2: Introsort vs Heapsort (non-timsort only) ────────────────
    print(f"\n[STAGE 2] Introsort vs Heapsort (non-timsort subset)")

    # Filter to non-timsort samples
    nontim_train = y_train_3 != 2
    nontim_val   = y_val_3 != 2
    nontim_test  = y_test_3 != 2

    X_train_s2 = X_train[nontim_train]
    y_train_s2 = y_train_3[nontim_train]  # 0=introsort, 1=heapsort
    X_val_s2   = X_val[nontim_val]
    y_val_s2   = y_val_3[nontim_val]

    print(f"  Train: {(y_train_s2==0).sum():,} introsort / {(y_train_s2==1).sum():,} heapsort")

    w2 = compute_sample_weights(y_train_s2)
    s2_model = xgb.XGBClassifier(**STAGE2_PARAMS)
    s2_model.fit(
        X_train_s2, y_train_s2,
        sample_weight=w2,
        eval_set=[(X_train_s2, y_train_s2), (X_val_s2, y_val_s2)],
        verbose=False,
    )
    print(f"  Stopped: epoch {s2_model.best_iteration}/{STAGE2_PARAMS['n_estimators']}")
    s2_val_pred = s2_model.predict(X_val_s2)
    s2_val_acc = accuracy_score(y_val_s2, s2_val_pred)
    s2_val_f1 = f1_score(y_val_s2, s2_val_pred, average="binary")
    print(f"  Val acc={s2_val_acc:.4f}  F1={s2_val_f1:.4f}")

    # ── Evaluate on test set ──────────────────────────────────────────────
    print(f"\n[EVAL] Test set — 3 strategies")

    # Strategy 1: Full cascade (Stage1 + Stage2)
    preds_cascade, s1_proba = cascade_predict(s1_model, s2_model, X_test)
    r_cascade = compute_regret_metrics(preds_cascade, t_test)

    # Strategy 2: Stage1 + heapsort default (no Stage2)
    preds_heapdef, _ = cascade_predict_heapsort_default(s1_model, X_test)
    r_heapdef = compute_regret_metrics(preds_heapdef, t_test)

    # Strategy 3: v5-style 3-class for comparison (retrain quick 3-class)
    print(f"\n[BASELINE] Training v5-style 3-class for fair comparison...")
    v5_params = dict(
        n_estimators=500, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
        n_jobs=-1, tree_method="hist", objective="multi:softprob",
        num_class=3, eval_metric="mlogloss",
    )
    v5_model = xgb.XGBClassifier(**v5_params)
    w_v5 = compute_sample_weights(y_train_3)
    v5_model.fit(X_train, y_train_3, sample_weight=w_v5,
                 eval_set=[(X_val, y_val_3)], verbose=False)
    preds_v5 = v5_model.predict(X_test)
    r_v5 = compute_regret_metrics(preds_v5, t_test)

    # Print comparison
    strategies = {
        "v5_3class": (preds_v5, r_v5),
        "v8_cascade": (preds_cascade, r_cascade),
        "v8_heap_default": (preds_heapdef, r_heapdef),
    }

    print(f"\n{'Strategy':<20} {'Acc':>6} {'Gap%':>7} {'Perfect%':>9} {'MeanReg':>9} {'P99Reg':>9}")
    print("-" * 62)
    for name, (preds, r) in strategies.items():
        acc = accuracy_score(y_test_3, preds)
        print(f"{name:<20} {acc:>5.1%} {r['gap_closed_pct']:>6.2f}% {r['perfect_picks_pct']:>8.2f}% "
              f"{r['mean_regret_us']:>8.3f}µ {r['p99_regret_us']:>8.2f}µ")

    # Train/val regret for overfitting check
    for split, Xs, ts in [("train", X_train, t_train), ("val", X_val, t_val)]:
        p, _ = cascade_predict(s1_model, s2_model, Xs)
        r = compute_regret_metrics(p, ts)
        print(f"  [{split}] cascade gap_closed={r['gap_closed_pct']:.2f}% mean_regret={r['mean_regret_us']:.3f}µs")

    # ── Figures ───────────────────────────────────────────────────────────
    print(f"\n[FIGURES]")

    plot_training_history(s1_model, "Stage 1 (Timsort vs Rest)", FIGURES_DIR, "01a_stage1_training.png")
    plot_training_history(s2_model, "Stage 2 (Introsort vs Heapsort)", FIGURES_DIR, "01b_stage2_training.png")

    s1_test_proba = s1_model.predict_proba(X_test)[:, 1]
    s1_auc = plot_stage1_roc(y_test_bin, s1_test_proba, FIGURES_DIR)

    # Stage 2 ROC on non-timsort test subset
    X_test_s2 = X_test[nontim_test]
    y_test_s2 = y_test_3[nontim_test]
    s2_test_proba = s2_model.predict_proba(X_test_s2)[:, 1]
    s2_auc = plot_stage2_roc(y_test_s2, s2_test_proba, FIGURES_DIR)

    plot_3class_confusion(y_test_3, preds_cascade, preds_heapdef, FIGURES_DIR)
    plot_feature_importance_both(s1_model, s2_model, FEATURE_NAMES, FIGURES_DIR)

    # Regret distributions
    regret_v5 = np.array([(t_test[i, preds_v5[i]] - t_test[i].min()) * 1e6
                          for i in range(len(preds_v5))])
    regret_cascade = np.array([(t_test[i, preds_cascade[i]] - t_test[i].min()) * 1e6
                               for i in range(len(preds_cascade))])
    regret_heapdef = np.array([(t_test[i, preds_heapdef[i]] - t_test[i].min()) * 1e6
                               for i in range(len(preds_heapdef))])
    plot_regret_comparison_all(
        {"v5_3class": regret_v5, "v8_cascade": regret_cascade, "v8_heap_default": regret_heapdef},
        FIGURES_DIR,
    )

    plot_timsort_probability_distribution(s1_test_proba, y_test_bin, FIGURES_DIR)
    plot_threshold_sweep(s1_model, s2_model, X_test, y_test_3, t_test, FIGURES_DIR)

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n[SAVE]")

    s1_model.get_booster().save_model(str(MODEL_DIR / "stage1_timsort_vs_rest.json"))
    s2_model.get_booster().save_model(str(MODEL_DIR / "stage2_introsort_vs_heapsort.json"))
    print(f"  Models saved to {MODEL_DIR}")

    output = {
        "timestamp": datetime.now().isoformat(),
        "version": "v8_binary_cascade",
        "architecture": "Stage1: timsort vs rest → Stage2: introsort vs heapsort",
        "stage1_params": STAGE1_PARAMS,
        "stage2_params": STAGE2_PARAMS,
        "stage1_best_iteration": s1_model.best_iteration,
        "stage2_best_iteration": s2_model.best_iteration,
        "stage1_auc": round(s1_auc, 4),
        "stage2_auc": round(s2_auc, 4),
        "dataset": {
            "total_raw": len(df),
            "after_undersample": len(df_bal),
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
            "stage2_train": len(X_train_s2),
        },
        "eval_results": {
            name: {
                "accuracy": round(float(accuracy_score(y_test_3, preds)), 4),
                **r,
            }
            for name, (preds, r) in strategies.items()
        },
        "stage1_feature_importance": [
            {"feature": FEATURE_NAMES[i], "importance": round(float(s1_model.feature_importances_[i]), 6)}
            for i in np.argsort(s1_model.feature_importances_)[::-1]
        ],
        "stage2_feature_importance": [
            {"feature": FEATURE_NAMES[i], "importance": round(float(s2_model.feature_importances_[i]), 6)}
            for i in np.argsort(s2_model.feature_importances_)[::-1]
        ],
    }

    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results → {RESULTS_DIR / 'evaluation_results.json'}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"DONE. {elapsed:.1f}s. Figures → {FIGURES_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
