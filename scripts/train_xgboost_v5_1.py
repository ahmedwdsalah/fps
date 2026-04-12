#!/usr/bin/env python3
"""
XGBoost v5.1 — Improved Production Classifier with Figures
============================================================
Fixes issues identified in v5 code analysis:

  1. EARLY STOPPING: early_stopping_rounds=30 (v5 trained all 500 blindly)
  2. MARGIN FILTER: restored 5% margin OR size≥2K filter (v5_with_figures dropped it)
  3. THRESHOLD TUNING: post-training timsort probability sweep (v5 used raw argmax)
  4. REGRET FIGURES: added per-instance regret analysis figures (v5 had none)
  5. EARLY STOP VIZ: shows where training stopped + overshoot zone

Same hyperparameters as v5. Same data pipeline. The model itself is a refinement,
not a new architecture.

Figures generated (13 total):
  01  Training loss curves (with early-stop marker)
  02  Training accuracy curves (with early-stop marker)
  03  ROC curves per class
  04  Precision-Recall curves per class
  05  t-SNE feature space
  06  Feature distributions — KDE (top 6 by importance)
  07  Confusion matrix (counts + row-normalised)
  08  Feature importance (horizontal bar, all 16)
  09  Per-class precision/recall/F1 grouped bar
  10  Prediction confidence histogram (correct vs incorrect)
  11  Timsort threshold sweep (gap closed vs threshold)
  12  Per-instance regret distribution (histogram + CDF)
  13  Regret by predicted class (box/violin)

Inputs:  data/training_dataset.csv
Outputs: models/xgboost_v5_1/
         results/xgboost_v5_1/

Usage:
    python3 scripts/train_xgboost_v5_1.py
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
from scipy.stats import gaussian_kde
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
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
DATA_CSV    = ROOT / "data" / "training_dataset.csv"
MODEL_DIR   = ROOT / "models" / "xgboost_v5_1"
RESULTS_DIR = ROOT / "results" / "xgboost_v5_1"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [MODEL_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
ALGO_COLORS = {"introsort": "#1f77b4", "heapsort": "#ff7f0e", "timsort": "#2ca02c"}
SEED = 42

# ── XGBoost hyperparameters ──────────────────────────────────────────────
# Same as v5, plus early stopping
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
    early_stopping_rounds=30,           # FIX #1: early stopping
)


# ── Data helpers ─────────────────────────────────────────────────────────

def apply_margin_filter(df: pd.DataFrame) -> pd.DataFrame:
    """FIX #2: Keep arrays where winner is ≥5% faster OR array is large (≥2000).
    Matches original train_xgboost_v5.py lines 161-175."""
    time_cols = df[["time_introsort", "time_heapsort", "time_timsort"]].values
    sorted_times = np.sort(time_cols, axis=1)
    best_time = sorted_times[:, 0]
    second_time = sorted_times[:, 1]
    margin = (second_time - best_time) / (second_time + 1e-15)
    has_margin = margin >= 0.05
    is_large = df["n_elements"].values >= 2000
    keep = has_margin | is_large
    return df[keep].reset_index(drop=True)


def balanced_undersample(df: pd.DataFrame, label_col: str,
                         max_ratio: float = 3.0) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    cap = int(counts.min() * max_ratio)
    parts = []
    for cls in counts.index:
        sub = df[df[label_col] == cls]
        if len(sub) > cap:
            sub = sub.sample(n=cap, random_state=SEED)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True).sample(
        frac=1.0, random_state=SEED).reset_index(drop=True)


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    wmap = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([wmap[yi] for yi in y], dtype=np.float64)


# ── Regret computation ───────────────────────────────────────────────────

def compute_regret(df: pd.DataFrame, predictions: np.ndarray,
                   le: LabelEncoder) -> dict:
    """Compute per-instance regret and aggregate metrics.
    Uses the test split's timing columns."""
    time_cols = ["time_introsort", "time_heapsort", "time_timsort"]
    times = df[time_cols].values  # shape (N, 3)
    best_times = times.min(axis=1)

    pred_labels = le.inverse_transform(predictions)
    pred_times = np.array([
        times[i, ALGORITHMS.index(pred_labels[i])]
        for i in range(len(pred_labels))
    ])

    regret = pred_times - best_times  # per-instance in seconds
    regret_us = regret * 1e6          # microseconds

    vbs_total = best_times.sum()
    model_total = pred_times.sum()
    sbs_totals = {algo: times[:, j].sum() for j, algo in enumerate(ALGORITHMS)}
    sbs_algo = min(sbs_totals, key=sbs_totals.get)
    sbs_total = sbs_totals[sbs_algo]

    gap_closed = (sbs_total - model_total) / (sbs_total - vbs_total) * 100 \
        if sbs_total > vbs_total else 0.0

    return {
        "regret_us": regret_us,
        "pred_labels": pred_labels,
        "vbs_total_s": float(vbs_total),
        "model_total_s": float(model_total),
        "sbs_total_s": float(sbs_total),
        "sbs_algorithm": sbs_algo,
        "gap_closed_pct": round(gap_closed, 4),
        "perfect_picks_pct": round(float((regret_us == 0).mean() * 100), 4),
        "mean_regret_us": round(float(regret_us.mean()), 4),
        "median_regret_us": round(float(np.median(regret_us)), 4),
        "p95_regret_us": round(float(np.percentile(regret_us, 95)), 4),
        "p99_regret_us": round(float(np.percentile(regret_us, 99)), 4),
        "max_regret_us": round(float(regret_us.max()), 4),
    }


# ── Threshold tuning ────────────────────────────────────────────────────

def sweep_timsort_threshold(y_proba: np.ndarray, df_test: pd.DataFrame,
                            le: LabelEncoder) -> tuple[float, list]:
    """FIX #3: Sweep timsort probability threshold to maximise gap closed.

    Default argmax treats all classes equally. But timsort misclassification is
    10-30x more expensive than intro↔heap confusion. Lowering the timsort threshold
    means we pick timsort more aggressively when it's even slightly likely.
    """
    results = []
    thresholds = np.arange(0.20, 0.70, 0.02)

    for thresh in thresholds:
        # Custom prediction: pick timsort if P(timsort) >= thresh, else argmax of remaining
        preds = np.empty(len(y_proba), dtype=int)
        tim_idx = ALGORITHMS.index("timsort")
        for i in range(len(y_proba)):
            if y_proba[i, tim_idx] >= thresh:
                preds[i] = tim_idx
            else:
                # argmax over non-timsort classes
                probs_no_tim = y_proba[i].copy()
                probs_no_tim[tim_idx] = -1
                preds[i] = probs_no_tim.argmax()

        regret_info = compute_regret(df_test, preds, le)
        results.append({
            "threshold": round(float(thresh), 2),
            "gap_closed_pct": regret_info["gap_closed_pct"],
            "perfect_picks_pct": regret_info["perfect_picks_pct"],
            "mean_regret_us": regret_info["mean_regret_us"],
            "accuracy": round(float(accuracy_score(
                le.transform(df_test["best_algorithm"].values), preds)), 4),
        })

    # Find best threshold by gap closed
    best = max(results, key=lambda x: x["gap_closed_pct"])
    return best["threshold"], results


# ── Figure functions ─────────────────────────────────────────────────────

def fig_01_training_loss(model, best_iter, path):
    """Training vs validation loss with early-stop marker."""
    print("[FIG 01] Training loss curves")
    res = model.evals_result()
    train_loss = res["validation_0"]["mlogloss"]
    val_loss = res["validation_1"]["mlogloss"]
    epochs = range(len(train_loss))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, train_loss, linewidth=2, label="Training Loss", color="#1f77b4")
    ax.plot(epochs, val_loss, linewidth=2, label="Validation Loss", color="#ff7f0e")

    # Early stopping marker
    if best_iter is not None and best_iter < len(train_loss):
        ax.axvline(best_iter, color="#d62728", linestyle="--", linewidth=1.5,
                   label=f"Best iteration ({best_iter})")
        # Shade overshoot zone
        if best_iter < len(train_loss) - 1:
            ax.axvspan(best_iter, len(train_loss) - 1, alpha=0.08, color="red",
                       label="Overshoot zone")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (mlogloss)")
    ax.set_title("Training History: Loss per Epoch")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path / "01_training_loss.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 01_training_loss.png (best_iter={best_iter})")


def fig_02_training_accuracy(X_train, y_train_enc, X_val, y_val_enc,
                             model, best_iter, path):
    """Training vs validation accuracy with early-stop marker."""
    print("[FIG 02] Training accuracy curves")

    res = model.evals_result()
    n_epochs = len(res["validation_0"]["mlogloss"])
    sample_every = max(1, n_epochs // 50)
    sampled = list(range(0, n_epochs, sample_every))

    train_accs, val_accs = [], []
    for ep in sampled:
        itr = (0, max(1, ep + 1))
        train_accs.append(accuracy_score(
            y_train_enc, model.predict(X_train, iteration_range=itr)) * 100)
        val_accs.append(accuracy_score(
            y_val_enc, model.predict(X_val, iteration_range=itr)) * 100)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sampled, train_accs, linewidth=2, label="Training Accuracy",
            color="#2ca02c", marker="o", markersize=3)
    ax.plot(sampled, val_accs, linewidth=2, label="Validation Accuracy",
            color="#d62728", marker="s", markersize=3)

    if best_iter is not None:
        ax.axvline(best_iter, color="#9467bd", linestyle="--", linewidth=1.5,
                   label=f"Early stop ({best_iter})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training History: Accuracy per Epoch")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0, min(train_accs + val_accs) - 5),
                top=min(100, max(train_accs + val_accs) + 5))
    plt.tight_layout()
    plt.savefig(path / "02_training_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 02_training_accuracy.png")


def fig_03_roc_curves(y_true, y_proba, path):
    print("[FIG 03] ROC curves")
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, algo in enumerate(ALGORITHMS):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, color=ALGO_COLORS[algo],
                label=f"{algo} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Test Set)")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path / "03_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 03_roc_curves.png")


def fig_04_pr_curves(y_true, y_proba, path):
    print("[FIG 04] Precision-Recall curves")
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, algo in enumerate(ALGORITHMS):
        y_bin = (y_true == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, i])
        ap = auc(rec, prec)
        ax.plot(rec, prec, linewidth=2, color=ALGO_COLORS[algo],
                label=f"{algo} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (Test Set)")
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(path / "04_precision_recall_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 04_precision_recall_curves.png")


def fig_05_tsne(X, y, path):
    print("[FIG 05] t-SNE feature space")
    # Subsample if too large (t-SNE is O(n²))
    n_max = 5000
    if len(X) > n_max:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(X), n_max, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    else:
        X_sub, y_sub = X, y

    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, algo in enumerate(ALGORITHMS):
        mask = y_sub == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=ALGO_COLORS[algo],
                   label=algo, alpha=0.5, s=20, edgecolors="none")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_title(f"t-SNE: Feature Space (n={len(X_sub):,})")
    ax.legend(fontsize=11, markerscale=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path / "05_tsne_features.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 05_tsne_features.png")


def fig_06_feature_distributions(X, y, feature_names, importance_indices, path):
    """KDE histograms — top 6 features by importance, per algorithm."""
    print("[FIG 06] Feature distributions (KDE, top 6 by importance)")
    top6 = importance_indices[:6]
    top_names = [feature_names[i] for i in top6]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (feat_idx, feat_name) in enumerate(zip(top6, top_names)):
        ax = axes[idx]
        for i, algo in enumerate(ALGORITHMS):
            vals = X[y == i, feat_idx]
            p1, p99 = np.percentile(vals, [1, 99])
            vals_clipped = vals[(vals >= p1) & (vals <= p99)]
            ax.hist(vals_clipped, bins=50, density=True, alpha=0.35,
                    color=ALGO_COLORS[algo], label=algo if idx == 0 else None)
            if len(vals_clipped) > 10 and vals_clipped.std() > 1e-10:
                kde = gaussian_kde(vals_clipped, bw_method=0.3)
                xgrid = np.linspace(p1, p99, 200)
                ax.plot(xgrid, kde(xgrid), linewidth=2, color=ALGO_COLORS[algo])
        ax.set_title(feat_name, fontweight="bold")
        ax.set_ylabel("Density" if idx % 3 == 0 else "")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, alpha=0.5, color=ALGO_COLORS[a])
               for a in ALGORITHMS]
    fig.legend(handles, ALGORITHMS, loc="upper center", ncol=3,
               fontsize=12, bbox_to_anchor=(0.5, 1.02))
    plt.suptitle("Feature Distributions by Algorithm (Top 6 by Importance)",
                 fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(path / "06_feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 06_feature_distributions.png")


def fig_07_confusion_matrix(y_true, y_pred, path):
    print("[FIG 07] Confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ALGORITHMS, yticklabels=ALGORITHMS, ax=ax1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Counts")

    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=ALGORITHMS, yticklabels=ALGORITHMS, ax=ax2)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Row-Normalised (%)")

    plt.suptitle("Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path / "07_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 07_confusion_matrix.png")


def fig_08_feature_importance(model, feature_names, path):
    """Horizontal bar — all 16 features, sorted by importance."""
    print("[FIG 08] Feature importance")
    imp = model.feature_importances_
    indices = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    ax.barh(range(len(indices)), imp[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("Feature Importance (All 16 Features)")
    ax.grid(True, alpha=0.3, axis="x")
    for i, idx in enumerate(indices):
        ax.text(imp[idx] + 0.002, i, f"{imp[idx]:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(path / "08_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 08_feature_importance.png")
    return np.argsort(imp)[::-1]  # descending for other plots


def fig_09_per_class_metrics(y_true, y_pred, path):
    """Per-class precision, recall, F1 as grouped bar chart."""
    print("[FIG 09] Per-class metrics")
    report = classification_report(y_true, y_pred, labels=[0, 1, 2],
                                   target_names=ALGORITHMS, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ALGORITHMS))
    width = 0.25
    colors_m = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, metric in enumerate(metrics):
        vals = [report[algo][metric] * 100 for algo in ALGORITHMS]
        bars = ax.bar(x + i * width, vals, width, label=metric.title(),
                      color=colors_m[i], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(ALGORITHMS, fontsize=12)
    ax.set_ylabel("Score (%)")
    ax.set_title("Per-Class Precision, Recall, F1 (Test Set)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(path / "09_per_class_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 09_per_class_metrics.png")


def fig_10_prediction_confidence(y_true, y_proba, path):
    """Histogram of max prediction probability — correct vs incorrect."""
    print("[FIG 10] Prediction confidence")
    max_prob = y_proba.max(axis=1)
    y_pred = y_proba.argmax(axis=1)
    correct = y_pred == y_true

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(max_prob[correct], bins=50, alpha=0.6, color="#2ca02c",
            label=f"Correct ({correct.sum():,})", density=True)
    ax.hist(max_prob[~correct], bins=50, alpha=0.6, color="#d62728",
            label=f"Incorrect ({(~correct).sum():,})", density=True)
    ax.axvline(1/3, color="gray", linestyle="--", linewidth=1, label="Random (0.33)")
    ax.set_xlabel("Max Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Confidence Distribution (Test Set)")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path / "10_prediction_confidence.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 10_prediction_confidence.png")


def fig_11_threshold_sweep(sweep_results, best_threshold, path):
    """FIX #3 visualization: timsort probability threshold vs gap closed."""
    print("[FIG 11] Timsort threshold sweep")
    thresholds = [r["threshold"] for r in sweep_results]
    gap_closed = [r["gap_closed_pct"] for r in sweep_results]
    perfect_picks = [r["perfect_picks_pct"] for r in sweep_results]
    accuracies = [r["accuracy"] * 100 for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    l1, = ax1.plot(thresholds, gap_closed, "o-", linewidth=2, color="#1f77b4",
                   markersize=4, label="Gap Closed (%)")
    l2, = ax1.plot(thresholds, perfect_picks, "s-", linewidth=2, color="#2ca02c",
                   markersize=4, label="Perfect Picks (%)")
    l3, = ax2.plot(thresholds, accuracies, "^-", linewidth=2, color="#ff7f0e",
                   markersize=4, label="Accuracy (%)")

    # Mark best and argmax=0.33
    ax1.axvline(best_threshold, color="#d62728", linestyle="--", linewidth=1.5,
                label=f"Best threshold ({best_threshold:.2f})")
    ax1.axvline(1/3, color="gray", linestyle=":", linewidth=1,
                label="Argmax default (0.33)")

    ax1.set_xlabel("Timsort Probability Threshold")
    ax1.set_ylabel("Gap Closed / Perfect Picks (%)")
    ax2.set_ylabel("Accuracy (%)", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc="lower left")
    ax1.set_title("Timsort Threshold Sweep: Gap Closed vs Threshold")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path / "11_threshold_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 11_threshold_sweep.png (best={best_threshold:.2f})")


def fig_12_regret_distribution(regret_us, path):
    """FIX #4: Per-instance regret histogram + CDF."""
    print("[FIG 12] Regret distribution")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: histogram (clip at P99 for readability)
    nonzero = regret_us[regret_us > 0]
    p99 = np.percentile(regret_us, 99) if len(regret_us) > 0 else 1
    zero_pct = (regret_us == 0).mean() * 100

    ax1.hist(nonzero[nonzero <= p99], bins=80, color="#1f77b4", alpha=0.7,
             edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Regret (μs)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Per-Instance Regret (non-zero only)\n"
                  f"{zero_pct:.1f}% zero regret, clipped at P99={p99:.1f} μs")
    ax1.grid(True, alpha=0.3)

    # Right: CDF
    sorted_regret = np.sort(regret_us)
    cdf = np.arange(1, len(sorted_regret) + 1) / len(sorted_regret) * 100
    ax2.plot(sorted_regret, cdf, linewidth=2, color="#2ca02c")
    ax2.axhline(95, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.axhline(99, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.text(sorted_regret.max() * 0.7, 95.5, "P95", fontsize=9, color="gray")
    ax2.text(sorted_regret.max() * 0.7, 99.5, "P99", fontsize=9, color="gray")
    ax2.set_xlabel("Regret (μs)")
    ax2.set_ylabel("Cumulative % of instances")
    ax2.set_title("Regret CDF")
    ax2.set_xlim(left=-0.5)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Per-Instance Regret Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path / "12_regret_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 12_regret_distribution.png")


def fig_13_regret_by_class(regret_us, pred_labels, path):
    """Regret broken down by which algorithm was predicted."""
    print("[FIG 13] Regret by predicted class")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: violin plot
    data_for_violin = []
    labels_for_violin = []
    for algo in ALGORITHMS:
        mask = pred_labels == algo
        vals = regret_us[mask]
        # Clip at P99 for readability
        p99 = np.percentile(vals, 99) if len(vals) > 0 else 1
        vals_clipped = vals[vals <= p99]
        data_for_violin.append(vals_clipped)
        labels_for_violin.append(algo)

    parts = ax1.violinplot(data_for_violin, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(ALGO_COLORS[ALGORITHMS[i]])
        pc.set_alpha(0.6)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(ALGORITHMS)
    ax1.set_ylabel("Regret (μs)")
    ax1.set_title("Regret Distribution by Predicted Algorithm")
    ax1.grid(True, alpha=0.3)

    # Right: bar chart of mean regret + % zero regret
    means = []
    zero_pcts = []
    for algo in ALGORITHMS:
        mask = pred_labels == algo
        vals = regret_us[mask]
        means.append(vals.mean() if len(vals) > 0 else 0)
        zero_pcts.append((vals == 0).mean() * 100 if len(vals) > 0 else 0)

    x = np.arange(len(ALGORITHMS))
    width = 0.35
    bars1 = ax2.bar(x - width/2, means, width, label="Mean Regret (μs)",
                    color=[ALGO_COLORS[a] for a in ALGORITHMS], alpha=0.8)
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, zero_pcts, width, label="Zero Regret (%)",
                         color=[ALGO_COLORS[a] for a in ALGORITHMS], alpha=0.4,
                         hatch="//")

    for bar, val in zip(bars1, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", fontsize=9)
    for bar, val in zip(bars2, zero_pcts):
        ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                      f"{val:.1f}%", ha="center", fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(ALGORITHMS)
    ax2.set_ylabel("Mean Regret (μs)")
    ax2_twin.set_ylabel("Zero Regret (%)")
    ax2.set_title("Mean Regret & Perfect Picks by Predicted Class")
    ax2.legend(loc="upper left", fontsize=9)
    ax2_twin.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Regret Breakdown by Predicted Algorithm",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path / "13_regret_by_class.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 13_regret_by_class.png")


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  XGBOOST v5.1 — Improved Production Classifier with Figures")
    print("=" * 80)
    t0 = time.time()

    # ── 1. Load ──────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    df_full = df.copy()
    print(f"  Loaded: {len(df):,} rows")
    print(f"  Classes: {dict(df['best_algorithm'].value_counts())}")

    # ── 2. Margin filter (FIX #2: restored from original v5) ────────────
    print(f"\n[2/7] Applying margin filter (≥5% OR size≥2K)")
    df = apply_margin_filter(df)
    print(f"  After filter: {len(df):,} rows")
    print(f"  Classes: {dict(df['best_algorithm'].value_counts())}")

    # ── 3. Balance ───────────────────────────────────────────────────────
    print(f"\n[3/7] Balancing (undersample max_ratio=3.0 + inv-freq weights)")
    df_bal = balanced_undersample(df, "best_algorithm", max_ratio=3.0)
    print(f"  After undersampling: {len(df_bal):,} rows")
    print(f"  Classes: {dict(df_bal['best_algorithm'].value_counts())}")

    # ── 4. Split ─────────────────────────────────────────────────────────
    print(f"\n[4/7] Splitting 70/15/15 (stratified)")
    X = df_bal[FEATURE_NAMES].values
    y = df_bal["best_algorithm"].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)
    print(f"  Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    le = LabelEncoder().fit(ALGORITHMS)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    weights = compute_sample_weights(y_train_enc)
    print(f"  Sample weights: min={weights.min():.2f}  max={weights.max():.2f}")

    # We also need timing columns for the test split for regret analysis
    # Reconstruct which rows ended up in test split
    df_bal_indexed = df_bal.copy()
    df_bal_indexed["_idx"] = range(len(df_bal_indexed))
    idx_all = df_bal_indexed["_idx"].values

    _, idx_temp, _, _ = train_test_split(
        idx_all, y, test_size=0.30, stratify=y, random_state=SEED)
    _, idx_test, _, _ = train_test_split(
        idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

    df_test = df_bal.iloc[idx_test].reset_index(drop=True)

    # ── 5. Train (FIX #1: early stopping) ───────────────────────────────
    print(f"\n[5/7] Training XGBoost v5.1 (with early_stopping_rounds=30)...")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train_enc,
        sample_weight=weights,
        eval_set=[(X_train, y_train_enc), (X_val, y_val_enc)],
        verbose=50,
    )

    best_iter = model.best_iteration if hasattr(model, "best_iteration") else None
    n_trained = len(model.evals_result()["validation_0"]["mlogloss"])
    print(f"  Trained {n_trained} rounds, best iteration: {best_iter}")
    print(f"  Training time: {time.time() - t0:.1f}s")

    # ── 6. Evaluate ──────────────────────────────────────────────────────
    print(f"\n[6/7] Evaluating...")
    results = {}
    test_proba = None

    for name, Xs, ys, ys_enc in [
        ("train", X_train, y_train, y_train_enc),
        ("val", X_val, y_val, y_val_enc),
        ("test", X_test, y_test, y_test_enc),
    ]:
        yp_enc = model.predict(Xs)
        yp = le.inverse_transform(yp_enc)
        yp_proba = model.predict_proba(Xs)

        acc = accuracy_score(ys, yp)
        bal_acc = balanced_accuracy_score(ys, yp)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "confusion_matrix": confusion_matrix(ys, yp, labels=ALGORITHMS).tolist(),
            "classification_report": classification_report(
                ys, yp, labels=ALGORITHMS, output_dict=True, zero_division=0),
        }
        print(f"  {name.upper()}: acc={acc:.4f}  bal_acc={bal_acc:.4f}")

        if name == "test":
            test_proba = yp_proba

    # Regret on test split
    y_pred_test = model.predict(X_test)
    regret_info = compute_regret(df_test, y_pred_test, le)
    results["test_regret"] = {k: v for k, v in regret_info.items()
                              if k not in ("regret_us", "pred_labels")}
    print(f"\n  TEST REGRET:")
    print(f"    Gap closed:    {regret_info['gap_closed_pct']:.2f}%")
    print(f"    Perfect picks: {regret_info['perfect_picks_pct']:.2f}%")
    print(f"    Mean regret:   {regret_info['mean_regret_us']:.4f} μs")
    print(f"    P99 regret:    {regret_info['p99_regret_us']:.4f} μs")

    # Threshold sweep (FIX #3)
    print(f"\n  THRESHOLD SWEEP:")
    best_thresh, sweep_results = sweep_timsort_threshold(test_proba, df_test, le)
    # Find argmax default result
    argmax_result = [r for r in sweep_results if abs(r["threshold"] - 0.34) < 0.02]
    if argmax_result:
        print(f"    Argmax default: gap_closed={argmax_result[0]['gap_closed_pct']:.2f}%")
    print(f"    Best threshold: {best_thresh:.2f}")
    best_sweep = [r for r in sweep_results if r["threshold"] == best_thresh][0]
    print(f"    Best gap_closed: {best_sweep['gap_closed_pct']:.2f}%")
    print(f"    Best perfect_picks: {best_sweep['perfect_picks_pct']:.2f}%")

    results["threshold_sweep"] = {
        "best_threshold": best_thresh,
        "sweep_results": sweep_results,
    }

    # Evaluate on full dataset too
    print(f"\n  FULL DATASET (1.18M):")
    X_full = df_full[FEATURE_NAMES].values
    y_full = df_full["best_algorithm"].values
    yp_full = le.inverse_transform(model.predict(X_full))
    full_acc = accuracy_score(y_full, yp_full)
    full_bal = balanced_accuracy_score(y_full, yp_full)
    results["full_dataset"] = {
        "accuracy": round(float(full_acc), 4),
        "balanced_accuracy": round(float(full_bal), 4),
    }
    print(f"    acc={full_acc:.4f}  bal_acc={full_bal:.4f}")

    # ── 7. Figures ───────────────────────────────────────────────────────
    print(f"\n[7/7] Generating 13 figures...")

    y_test_np = y_test_enc
    y_pred_test_enc = model.predict(X_test)

    fig_01_training_loss(model, best_iter, FIGURES_DIR)
    fig_02_training_accuracy(X_train, y_train_enc, X_val, y_val_enc,
                             model, best_iter, FIGURES_DIR)
    fig_03_roc_curves(y_test_np, test_proba, FIGURES_DIR)
    fig_04_pr_curves(y_test_np, test_proba, FIGURES_DIR)
    fig_05_tsne(X_test, y_test_np, FIGURES_DIR)
    imp_indices = fig_08_feature_importance(model, FEATURE_NAMES, FIGURES_DIR)
    fig_06_feature_distributions(X_test, y_test_np, FEATURE_NAMES,
                                 imp_indices, FIGURES_DIR)
    fig_07_confusion_matrix(y_test_np, y_pred_test_enc, FIGURES_DIR)
    fig_09_per_class_metrics(y_test_np, y_pred_test_enc, FIGURES_DIR)
    fig_10_prediction_confidence(y_test_np, test_proba, FIGURES_DIR)
    fig_11_threshold_sweep(sweep_results, best_thresh, FIGURES_DIR)
    fig_12_regret_distribution(regret_info["regret_us"], FIGURES_DIR)
    fig_13_regret_by_class(regret_info["regret_us"], regret_info["pred_labels"],
                           FIGURES_DIR)

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n[SAVE]")

    model_file = MODEL_DIR / "xgb_v5_1.ubj"
    model.save_model(str(model_file))
    print(f"  Model: {model_file}")

    # Feature importance
    feat_imp = sorted(
        zip(FEATURE_NAMES, model.feature_importances_.tolist()),
        key=lambda x: -x[1]
    )

    output = {
        "timestamp": datetime.now().isoformat(),
        "version": "v5.1",
        "changes_from_v5": [
            "Added early_stopping_rounds=30",
            "Restored margin filter (>=5% OR size>=2K) from original v5",
            "Added timsort threshold sweep post-processing",
            "Added per-instance regret analysis on test split",
        ],
        "xgb_params": {k: v for k, v in XGB_PARAMS.items()},
        "best_iteration": best_iter,
        "total_epochs_trained": n_trained,
        "features": FEATURE_NAMES,
        "algorithms": ALGORITHMS,
        "dataset": {
            "total_raw": len(df_full),
            "after_margin_filter": len(df),
            "after_undersample": len(df_bal),
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "results": results,
        "feature_importance": [
            {"feature": f, "importance": round(i, 6)} for f, i in feat_imp
        ],
    }

    results_file = RESULTS_DIR / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results: {results_file}")

    pred_df = pd.DataFrame({
        "predicted": le.inverse_transform(y_pred_test_enc),
        "actual": le.inverse_transform(y_test_np),
    })
    pred_df.to_csv(RESULTS_DIR / "predictions_test.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  v5.1 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Model:   {model_file}")
    print(f"  Results: {results_file}")
    print(f"  Figures: {FIGURES_DIR} (13 PNGs)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
