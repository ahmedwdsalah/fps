#!/usr/bin/env python3
"""
Step 3: Train XGBoost v5 WITH FIGURES — Balanced Real-World Classifier
========================================================================
Identical model to train_xgboost_v5.py, with thesis-quality figures.

Balance strategy (3-pronged):
  1. Undersample majority: cap timsort at ~3× the minority class count
  2. sample_weight: inverse-frequency weighting on the undersampled set
  3. eval_metric: mlogloss (proper multi-class metric)

Split: 70% train / 15% val / 15% test (stratified)

Figures generated (10 total):
  01  Training loss curves (train vs val)
  02  Training accuracy curves (train vs val)
  03  ROC curves (per-class)
  04  Precision-Recall curves (per-class)
  05  t-SNE feature space
  06  Feature distributions — KDE ridge plot (top 6 by importance)
  07  Confusion matrix heatmap
  08  Feature importance (horizontal bar, all 16)
  09  Per-class accuracy bar chart
  10  Prediction confidence histogram

Inputs:  data/training_dataset.csv
Outputs: models/xgboost_v5_with_figures/
         results/xgboost_v5_with_figures/

Usage:
    python3 scripts/train_xgboost_v5_with_figures.py
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
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
DATA_CSV    = ROOT / "data" / "training_dataset.csv"
MODEL_DIR   = ROOT / "models" / "xgboost_v5_with_figures"
RESULTS_DIR = ROOT / "results" / "xgboost_v5_with_figures"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
ALGO_COLORS = {"introsort": "#1f77b4", "heapsort": "#ff7f0e", "timsort": "#2ca02c"}
SEED = 42

# ── XGBoost hyperparameters (identical to v5) ───────────────────────────
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


# ── Balance ──────────────────────────────────────────────────────────────

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


# ── Figure functions ─────────────────────────────────────────────────────

def fig_01_training_loss(model, path):
    """Training vs validation loss — separate, clean figure."""
    print("[FIG 01] Training loss curves")
    res = model.evals_result()
    train_loss = res["validation_0"]["mlogloss"]
    val_loss = res["validation_1"]["mlogloss"]
    epochs = range(len(train_loss))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, train_loss, linewidth=2, label="Training Loss", color="#1f77b4")
    ax.plot(epochs, val_loss, linewidth=2, label="Validation Loss", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (mlogloss)")
    ax.set_title("Training History: Loss per Epoch")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = path / "01_training_loss.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def fig_02_training_accuracy(X_train, y_train_enc, X_val, y_val_enc, model, path):
    """Training vs validation accuracy — separate figure, no dual axis."""
    print("[FIG 02] Training accuracy curves")

    res = model.evals_result()
    n_epochs = len(res["validation_0"]["mlogloss"])
    sample_every = max(1, n_epochs // 50)
    sampled = list(range(0, n_epochs, sample_every))

    train_accs, val_accs = [], []
    for ep in sampled:
        itr = (0, max(1, ep + 1))
        train_accs.append(accuracy_score(y_train_enc, model.predict(X_train, iteration_range=itr)) * 100)
        val_accs.append(accuracy_score(y_val_enc, model.predict(X_val, iteration_range=itr)) * 100)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sampled, train_accs, linewidth=2, label="Training Accuracy",
            color="#2ca02c", marker="o", markersize=3)
    ax.plot(sampled, val_accs, linewidth=2, label="Validation Accuracy",
            color="#d62728", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training History: Accuracy per Epoch")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0, min(train_accs + val_accs) - 5),
                top=min(100, max(train_accs + val_accs) + 5))
    plt.tight_layout()
    out = path / "02_training_accuracy.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


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
    out = path / "03_roc_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


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
    out = path / "04_precision_recall_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def fig_05_tsne(X, y, path):
    print("[FIG 05] t-SNE feature space")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, algo in enumerate(ALGORITHMS):
        mask = y == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=ALGO_COLORS[algo],
                   label=algo, alpha=0.5, s=20, edgecolors="none")
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_title("t-SNE: Feature Space (Test Set)")
    ax.legend(fontsize=11, markerscale=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = path / "05_tsne_features.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def fig_06_feature_distributions(X, y, feature_names, importance_indices, path):
    """KDE ridge plot — top 6 features by importance, per algorithm."""
    print("[FIG 06] Feature distributions (KDE, top 6 by importance)")

    top6 = importance_indices[:6]
    top_names = [feature_names[i] for i in top6]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (feat_idx, feat_name) in enumerate(zip(top6, top_names)):
        ax = axes[idx]
        for i, algo in enumerate(ALGORITHMS):
            vals = X[y == i, feat_idx]
            # Clip extreme outliers for cleaner KDE
            p1, p99 = np.percentile(vals, [1, 99])
            vals_clipped = vals[(vals >= p1) & (vals <= p99)]
            ax.hist(vals_clipped, bins=50, density=True, alpha=0.35,
                    color=ALGO_COLORS[algo], label=algo if idx == 0 else None)
            # KDE overlay
            from scipy.stats import gaussian_kde
            if len(vals_clipped) > 10 and vals_clipped.std() > 1e-10:
                kde = gaussian_kde(vals_clipped, bw_method=0.3)
                xgrid = np.linspace(p1, p99, 200)
                ax.plot(xgrid, kde(xgrid), linewidth=2, color=ALGO_COLORS[algo])

        ax.set_title(feat_name, fontweight="bold")
        ax.set_ylabel("Density" if idx % 3 == 0 else "")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=9)

    # Shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, alpha=0.5, color=ALGO_COLORS[a])
               for a in ALGORITHMS]
    fig.legend(handles, ALGORITHMS, loc="upper center", ncol=3,
               fontsize=12, bbox_to_anchor=(0.5, 1.02))

    plt.suptitle("Feature Distributions by Algorithm (Top 6 by Importance)",
                 fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    out = path / "06_feature_distributions.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def fig_07_confusion_matrix(y_true, y_pred, path):
    print("[FIG 07] Confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    # Also compute normalised version
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
    out = path / "07_confusion_matrix.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def fig_08_feature_importance(model, feature_names, path):
    """Horizontal bar — all 16 features, sorted by importance."""
    print("[FIG 08] Feature importance")
    imp = model.feature_importances_
    indices = np.argsort(imp)  # ascending for horizontal bar

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    ax.barh(range(len(indices)), imp[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("Feature Importance (All 16 Features)")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, idx in enumerate(indices):
        ax.text(imp[idx] + 0.002, i, f"{imp[idx]:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    out = path / "08_feature_importance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")
    return np.argsort(imp)[::-1]  # return descending order for other plots


def fig_09_per_class_accuracy(y_true, y_pred, path):
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
    out = path / "09_per_class_metrics.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


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
    out = path / "10_prediction_confidence.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("XGBOOST v5 WITH FIGURES — Balanced Real-World Classifier")
    print("=" * 80)

    start_time = time.time()

    # Load
    print(f"\n[LOAD] {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    print(f"  {len(df):,} rows")
    print(f"  Classes: {dict(df['best_algorithm'].value_counts())}")

    # Balance (same as v5 — NO margin filter)
    print(f"\n[BALANCE] max_ratio=3.0")
    df_bal = balanced_undersample(df, "best_algorithm", max_ratio=3.0)
    print(f"  {len(df_bal):,} rows after undersampling")
    print(f"  Classes: {dict(df_bal['best_algorithm'].value_counts())}")

    # Features + labels
    X = df_bal[FEATURE_NAMES].values
    y = df_bal["best_algorithm"].values

    # Split 70/15/15
    print(f"\n[SPLIT] 70/15/15 stratified")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)
    print(f"  Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    # Encode
    le = LabelEncoder().fit(ALGORITHMS)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    weights = compute_sample_weights(y_train_enc)

    # Train
    print(f"\n[TRAIN] XGBoost v5...")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train_enc,
        sample_weight=weights,
        eval_set=[(X_train, y_train_enc), (X_val, y_val_enc)],
        verbose=False,
    )
    print(f"  Done in {time.time() - start_time:.1f}s")

    # Evaluate
    print(f"\n[EVAL]")
    results = {}
    predictions = {}

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
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "confusion_matrix": confusion_matrix(ys, yp, labels=ALGORITHMS).tolist(),
            "classification_report": classification_report(
                ys, yp, labels=ALGORITHMS, output_dict=True, zero_division=0),
        }
        predictions[name] = {
            "y_true_enc": ys_enc,
            "y_pred_enc": yp_enc,
            "y_proba": yp_proba,
        }
        print(f"  {name.upper()}: acc={acc:.3f}  balanced_acc={bal_acc:.3f}")

    # ── Figures ───────────────────────────────────────────────────────────
    print(f"\n[FIGURES]")

    y_test_np = predictions["test"]["y_true_enc"]
    y_pred_test = predictions["test"]["y_pred_enc"]
    y_proba_test = predictions["test"]["y_proba"]

    fig_01_training_loss(model, FIGURES_DIR)
    fig_02_training_accuracy(X_train, y_train_enc, X_val, y_val_enc, model, FIGURES_DIR)
    fig_03_roc_curves(y_test_np, y_proba_test, FIGURES_DIR)
    fig_04_pr_curves(y_test_np, y_proba_test, FIGURES_DIR)
    fig_05_tsne(X_test, y_test_np, FIGURES_DIR)
    imp_indices = fig_08_feature_importance(model, FEATURE_NAMES, FIGURES_DIR)
    fig_06_feature_distributions(X_test, y_test_np, FEATURE_NAMES, imp_indices, FIGURES_DIR)
    fig_07_confusion_matrix(y_test_np, y_pred_test, FIGURES_DIR)
    fig_09_per_class_accuracy(y_test_np, y_pred_test, FIGURES_DIR)
    fig_10_prediction_confidence(y_test_np, y_proba_test, FIGURES_DIR)

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n[SAVE]")

    model_file = MODEL_DIR / "xgb_v5_with_figures.json"
    model.get_booster().save_model(str(model_file))
    print(f"  Model: {model_file}")

    output = {
        "timestamp": datetime.now().isoformat(),
        "xgb_params": XGB_PARAMS,
        "features": FEATURE_NAMES,
        "algorithms": ALGORITHMS,
        "dataset": {
            "total_raw": len(df),
            "after_undersample": len(df_bal),
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "results": results,
        "feature_importance": [
            {"feature": name, "importance": float(score)}
            for name, score in sorted(
                zip(FEATURE_NAMES, model.feature_importances_),
                key=lambda x: x[1], reverse=True
            )
        ],
    }

    results_file = RESULTS_DIR / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results: {results_file}")

    pred_df = pd.DataFrame({
        "predicted": le.inverse_transform(y_pred_test),
        "actual": le.inverse_transform(y_test_np),
    })
    pred_file = RESULTS_DIR / "predictions_test.csv"
    pred_df.to_csv(pred_file, index=False)
    print(f"  Predictions: {pred_file}")

    print(f"\n{'='*80}")
    print(f"DONE. Figures → {FIGURES_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
