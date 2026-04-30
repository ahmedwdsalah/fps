#!/usr/bin/env python3
"""
Train XGBoost v5 for the dedicated F1-only dataset, with figures.

Uses the same core training logic as train_xgboost_v5.py:
  - noisy-label filter
  - balanced undersampling
  - inverse-frequency sample weights
  - 70/15/15 stratified split

Inputs:
  /Volumes/k/thesis_data/f1_only/training_dataset.csv

Outputs:
  models/xgboost_v5_f1/xgb_v5_f1.json
  results/xgboost_v5_f1/evaluation_results.json
  results/xgboost_v5_f1/predictions_test.csv
  results/xgboost_v5_f1/figures/*.png

Usage:
    python3 scripts/train_xgboost_v5_f1.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12})

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_CSV = Path("/Volumes/k/thesis_data/f1_only/training_dataset.csv")
MODEL_DIR = ROOT / "models" / "xgboost_v5_f1"
RESULTS_DIR = ROOT / "results" / "xgboost_v5_f1"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
ALGO_COLORS = {"introsort": "#1f77b4", "heapsort": "#ff7f0e", "timsort": "#2ca02c"}
SEED = 42

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


def _validate_training_dataframe(df: pd.DataFrame, label_col: str) -> None:
    required_cols = set(FEATURE_NAMES) | {
        label_col,
        "n_elements",
        "time_introsort",
        "time_heapsort",
        "time_timsort",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Training dataset is empty")
    if df[label_col].isna().any():
        raise ValueError(f"{label_col} contains NaN labels")
    invalid = sorted(set(df[label_col].unique()) - set(ALGORITHMS))
    if invalid:
        raise ValueError(f"Unexpected labels in {label_col}: {invalid}")


def balanced_undersample(
    df: pd.DataFrame,
    label_col: str,
    max_ratio: float = 3.0,
    *,
    random_state: int = SEED,
) -> pd.DataFrame:
    if label_col not in df.columns:
        raise KeyError(f"Missing label column: {label_col}")
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    if max_ratio < 1.0:
        raise ValueError("max_ratio must be >= 1.0")
    if df[label_col].isna().any():
        raise ValueError(f"{label_col} contains NaN labels")

    counts = df[label_col].value_counts(sort=False, dropna=False)
    if counts.empty:
        raise ValueError("No class counts available for undersampling")

    min_count = int(counts.min())
    cap = int(min_count * max_ratio)
    if cap <= 0:
        raise ValueError("Computed undersampling cap must be positive")

    parts = []
    for _, subset in df.groupby(label_col, sort=False, dropna=False):
        n = min(len(subset), cap)
        parts.append(subset.sample(n=n, random_state=random_state))

    return pd.concat(parts, ignore_index=True).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[yi] for yi in y], dtype=np.float64)


def fig_01_training_loss(model, path: Path) -> None:
    res = model.evals_result()
    train_loss = res["validation_0"]["mlogloss"]
    val_loss = res["validation_1"]["mlogloss"]
    epochs = range(len(train_loss))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, train_loss, linewidth=2, label="Training Loss", color="#1f77b4")
    ax.plot(epochs, val_loss, linewidth=2, label="Validation Loss", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (mlogloss)")
    ax.set_title("F1-only XGBoost v5: Training Loss")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path / "01_training_loss.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_02_training_accuracy(X_train, y_train_enc, X_val, y_val_enc, model, path: Path) -> None:
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
    ax.plot(sampled, train_accs, linewidth=2, label="Training Accuracy", color="#2ca02c")
    ax.plot(sampled, val_accs, linewidth=2, label="Validation Accuracy", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("F1-only XGBoost v5: Training Accuracy")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path / "02_training_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_03_roc_curves(y_true, y_proba, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, algo in enumerate(ALGORITHMS):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, color=ALGO_COLORS[algo], label=f"{algo} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("F1-only ROC Curves")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path / "03_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_04_pr_curves(y_true, y_proba, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, algo in enumerate(ALGORITHMS):
        y_bin = (y_true == i).astype(int)
        prec, rec, _ = precision_recall_curve(y_bin, y_proba[:, i])
        ap = auc(rec, prec)
        ax.plot(rec, prec, linewidth=2, color=ALGO_COLORS[algo], label=f"{algo} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("F1-only Precision-Recall Curves")
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(path / "04_precision_recall_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_05_tsne(X, y, path: Path) -> None:
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, algo in enumerate(ALGORITHMS):
        mask = y == i
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=ALGO_COLORS[algo],
            label=algo,
            alpha=0.5,
            s=20,
            edgecolors="none",
        )
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    ax.set_title("F1-only Feature Space (t-SNE)")
    ax.legend(fontsize=11, markerscale=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path / "05_tsne_features.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_06_feature_distributions(X, y, feature_names, importance_indices, path: Path) -> None:
    from scipy.stats import gaussian_kde

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
            ax.hist(vals_clipped, bins=50, density=True, alpha=0.35, color=ALGO_COLORS[algo], label=algo if idx == 0 else None)
            if len(vals_clipped) > 10 and vals_clipped.std() > 1e-10:
                kde = gaussian_kde(vals_clipped, bw_method=0.3)
                xgrid = np.linspace(p1, p99, 200)
                ax.plot(xgrid, kde(xgrid), linewidth=2, color=ALGO_COLORS[algo])
        ax.set_title(feat_name, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, alpha=0.5, color=ALGO_COLORS[a]) for a in ALGORITHMS]
    fig.legend(handles, ALGORITHMS, loc="upper center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, 1.02))
    plt.suptitle("F1-only Feature Distributions (Top 6 by Importance)", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(path / "06_feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_07_confusion_matrix(y_true, y_pred, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=ALGORITHMS, yticklabels=ALGORITHMS, ax=ax1)
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues", xticklabels=ALGORITHMS, yticklabels=ALGORITHMS, ax=ax2)
    ax1.set_title("Counts")
    ax2.set_title("Row-Normalized (%)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    plt.suptitle("F1-only Confusion Matrix", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path / "07_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_08_feature_importance(model, feature_names, path: Path):
    imp = model.feature_importances_
    indices = np.argsort(imp)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    ax.barh(range(len(indices)), imp[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("F1-only Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(path / "08_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    return np.argsort(imp)[::-1]


def fig_09_per_class_metrics(y_true, y_pred, path: Path) -> None:
    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=ALGORITHMS, output_dict=True
    )
    metrics = ["precision", "recall", "f1-score"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ALGORITHMS))
    width = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, metric in enumerate(metrics):
        vals = [report[algo][metric] * 100 for algo in ALGORITHMS]
        ax.bar(x + i * width, vals, width, label=metric.title(), color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(ALGORITHMS)
    ax.set_ylabel("Score (%)")
    ax.set_title("F1-only Per-Class Metrics")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(path / "09_per_class_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_10_prediction_confidence(y_true, y_proba, path: Path) -> None:
    max_prob = y_proba.max(axis=1)
    y_pred = y_proba.argmax(axis=1)
    correct = y_pred == y_true

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(max_prob[correct], bins=50, alpha=0.6, color="#2ca02c", label=f"Correct ({correct.sum():,})", density=True)
    ax.hist(max_prob[~correct], bins=50, alpha=0.6, color="#d62728", label=f"Incorrect ({(~correct).sum():,})", density=True)
    ax.axvline(1 / 3, color="gray", linestyle="--", linewidth=1, label="Random (0.33)")
    ax.set_xlabel("Max Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("F1-only Prediction Confidence")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path / "10_prediction_confidence.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("=" * 80)
    print("XGBOOST v5 F1-ONLY")
    print("=" * 80)
    print(f"\n[LOAD] {DATA_CSV}")

    if not DATA_CSV.exists():
        raise SystemExit(f"Training dataset not found: {DATA_CSV}")

    start_time = time.time()
    df = pd.read_csv(DATA_CSV)
    _validate_training_dataframe(df, "best_algorithm")
    total_raw = len(df)
    print(f"  Loaded: {len(df):,} rows")

    print("\n[FILTER] margin>=5% OR size>=2K")
    time_cols = df[["time_introsort", "time_heapsort", "time_timsort"]].values
    sorted_times = np.sort(time_cols, axis=1)
    best_time = sorted_times[:, 0]
    second_time = sorted_times[:, 1]
    margin = (second_time - best_time) / (second_time + 1e-15)
    has_margin = margin >= 0.05
    is_large = df["n_elements"].values >= 2000
    keep = has_margin | is_large
    df = df[keep].reset_index(drop=True)
    _validate_training_dataframe(df, "best_algorithm")
    df_eval_full = df.copy()
    print(f"  After filter: {len(df):,} rows")

    classes_present = sorted(df["best_algorithm"].dropna().unique().tolist())
    if set(classes_present) != set(ALGORITHMS):
        raise SystemExit(
            "F1 training dataset does not yet contain all 3 classes. "
            f"Found: {classes_present}"
        )

    print("\n[SPLIT] 70/15/15 stratified")
    df_train, df_temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["best_algorithm"],
        random_state=SEED,
    )
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        stratify=df_temp["best_algorithm"],
        random_state=SEED,
    )
    print(f"  Train(raw)={len(df_train):,}  Val={len(df_val):,}  Test={len(df_test):,}")

    print("\n[BALANCE] train only, max_ratio=3.0")
    df_train_bal = balanced_undersample(df_train, "best_algorithm", max_ratio=3.0)
    print(f"  Train(after undersampling)={len(df_train_bal):,} rows")

    X_train = df_train_bal[FEATURE_NAMES].values
    y_train = df_train_bal["best_algorithm"].values
    X_val = df_val[FEATURE_NAMES].values
    y_val = df_val["best_algorithm"].values
    X_test = df_test[FEATURE_NAMES].values
    y_test = df_test["best_algorithm"].values

    le = LabelEncoder().fit(ALGORITHMS)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    weights = compute_sample_weights(y_train_enc)

    print("\n[TRAIN] XGBoost")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train,
        y_train_enc,
        sample_weight=weights,
        eval_set=[(X_train, y_train_enc), (X_val, y_val_enc)],
        verbose=False,
    )

    print("\n[EVAL]")
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
        results[name] = {
            "accuracy": float(accuracy_score(ys, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(ys, yp)),
            "confusion_matrix": confusion_matrix(ys, yp, labels=ALGORITHMS).tolist(),
            "classification_report": classification_report(
                ys, yp, labels=ALGORITHMS, output_dict=True, zero_division=0
            ),
        }
        predictions[name] = {
            "y_true_enc": ys_enc,
            "y_pred_enc": yp_enc,
            "y_proba": yp_proba,
        }
        print(
            f"  {name.upper()}: "
            f"acc={results[name]['accuracy']:.3f}  "
            f"balanced_acc={results[name]['balanced_accuracy']:.3f}"
        )

    print("\n[EVAL FULL]")
    X_full = df_eval_full[FEATURE_NAMES].values
    y_full = df_eval_full["best_algorithm"].values
    y_full_pred = le.inverse_transform(model.predict(X_full))
    results["full_dataset"] = {
        "accuracy": float(accuracy_score(y_full, y_full_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_full, y_full_pred)),
        "confusion_matrix": confusion_matrix(y_full, y_full_pred, labels=ALGORITHMS).tolist(),
        "classification_report": classification_report(
            y_full, y_full_pred, labels=ALGORITHMS, output_dict=True, zero_division=0
        ),
    }

    print("\n[FIGURES]")
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
    fig_09_per_class_metrics(y_test_np, y_pred_test, FIGURES_DIR)
    fig_10_prediction_confidence(y_test_np, y_proba_test, FIGURES_DIR)

    print("\n[SAVE]")
    model_file = MODEL_DIR / "xgb_v5_f1.json"
    model.get_booster().save_model(str(model_file))

    output = {
        "timestamp": datetime.now().isoformat(),
        "xgb_params": XGB_PARAMS,
        "features": FEATURE_NAMES,
        "algorithms": ALGORITHMS,
        "dataset": {
            "input_csv": str(DATA_CSV),
            "total_raw": total_raw,
            "after_filter": len(df),
            "train_raw": len(df_train),
            "train_after_undersample": len(df_train_bal),
            "val": len(df_val),
            "test": len(df_test),
            "class_distribution_after_filter": df["best_algorithm"].value_counts().to_dict(),
            "class_distribution_train_raw": df_train["best_algorithm"].value_counts().to_dict(),
            "class_distribution_train_balanced": df_train_bal["best_algorithm"].value_counts().to_dict(),
        },
        "results": results,
        "feature_importance": [
            {"feature": name, "importance": float(score)}
            for name, score in sorted(
                zip(FEATURE_NAMES, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        ],
    }

    results_file = RESULTS_DIR / "evaluation_results.json"
    results_file.write_text(json.dumps(output, indent=2))

    pred_file = RESULTS_DIR / "predictions_test.csv"
    pd.DataFrame(
        {
            "predicted": le.inverse_transform(y_pred_test),
            "actual": le.inverse_transform(y_test_np),
        }
    ).to_csv(pred_file, index=False)

    elapsed = time.time() - start_time
    print(f"  Model: {model_file}")
    print(f"  Results: {results_file}")
    print(f"  Figures: {FIGURES_DIR}")
    print(f"\nDONE in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
