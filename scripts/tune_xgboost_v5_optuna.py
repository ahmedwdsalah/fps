#!/usr/bin/env python3
"""
Optuna tuning for XGBoost v5 baseline.

Uses same core data preparation as train_xgboost_v5.py:
- label-noise filter (margin>=5% OR n_elements>=2000)
- undersample majority with max_ratio
- stratified 70/15/15 split
- inverse-frequency sample weights
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_CSV = ROOT / "data" / "training_dataset.csv"
MODEL_DIR = ROOT / "models" / "xgboost_v5_optuna"
RESULTS_DIR = ROOT / "results" / "xgboost_v5_optuna"
STUDY_DB = RESULTS_DIR / "optuna_study.db"
FIG_DIR = RESULTS_DIR / "figures"

SEED = 42
ALGORITHMS = ["introsort", "heapsort", "timsort"]
FEATURE_NAMES = [
    "length_norm",
    "adj_sorted_ratio",
    "duplicate_ratio",
    "dispersion_ratio",
    "runs_ratio",
    "inversion_ratio",
    "entropy_ratio",
    "skewness_t",
    "kurtosis_excess_t",
    "longest_run_ratio",
    "iqr_norm",
    "mad_norm",
    "top1_freq_ratio",
    "top5_freq_ratio",
    "outlier_ratio",
    "mean_abs_diff_norm",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--timeout-sec", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--max-ratio", type=float, default=3.0, help="Undersample cap: max class <= ratio * minority")
    return p.parse_args()


def balanced_undersample(df: pd.DataFrame, label_col: str, max_ratio: float = 3.0) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    min_count = counts.min()
    cap = int(min_count * max_ratio)
    parts = []
    for cls in counts.index:
        sub = df[df[label_col] == cls]
        if len(sub) > cap:
            sub = sub.sample(n=cap, random_state=SEED)
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    w = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([w[i] for i in y], dtype=np.float64)


def prepare_data(max_ratio: float):
    df = pd.read_csv(DATA_CSV)
    # v5 filter logic
    tcols = df[["time_introsort", "time_heapsort", "time_timsort"]].values
    st = np.sort(tcols, axis=1)
    margin = (st[:, 1] - st[:, 0]) / (st[:, 1] + 1e-15)
    keep = (margin >= 0.05) | (df["n_elements"].values >= 2000)
    df = df[keep].reset_index(drop=True)
    df = balanced_undersample(df, "best_algorithm", max_ratio=max_ratio)

    X = df[FEATURE_NAMES].values
    y = df["best_algorithm"].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    le = LabelEncoder().fit(ALGORITHMS)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    return df, le, X_train, X_val, X_test, y_train, y_val, y_test, y_train_enc, y_val_enc, y_test_enc


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OPTUNA TUNE XGBOOST V5")
    print("=" * 80)
    print(f"Data: {DATA_CSV}")
    print(f"Trials: {args.trials}")
    print(f"Workers: {args.workers}")
    print(f"Undersample max_ratio: {args.max_ratio}")

    t0 = time.time()
    df, le, X_train, X_val, X_test, y_train, y_val, y_test, y_train_enc, y_val_enc, y_test_enc = prepare_data(args.max_ratio)
    weights = compute_sample_weights(y_train_enc)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": SEED,
            "n_jobs": args.workers,
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        }
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train,
            y_train_enc,
            sample_weight=weights,
            eval_set=[(X_val, y_val_enc)],
            verbose=False,
        )
        yp = le.inverse_transform(model.predict(X_val))
        return float(balanced_accuracy_score(y_val, yp))

    study = optuna.create_study(
        study_name="xgboost_v5_optuna",
        storage=f"sqlite:///{STUDY_DB}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials, timeout=(args.timeout_sec or None))

    best_params = study.best_trial.params
    final_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": args.workers,
        **best_params,
    }
    model = xgb.XGBClassifier(**final_params)
    model.fit(
        X_train,
        y_train_enc,
        sample_weight=weights,
        eval_set=[(X_train, y_train_enc), (X_val, y_val_enc)],
        verbose=False,
    )

    def eval_split(name: str, X, y_true):
        yp_enc = model.predict(X)
        yp = le.inverse_transform(yp_enc)
        return {
            "accuracy": float(accuracy_score(y_true, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, yp)),
            "confusion_matrix": confusion_matrix(y_true, yp, labels=ALGORITHMS).tolist(),
            "classification_report": classification_report(
                y_true, yp, labels=ALGORITHMS, output_dict=True, zero_division=0
            ),
        }

    train_res = eval_split("train", X_train, y_train)
    val_res = eval_split("val", X_val, y_val)
    test_res = eval_split("test", X_test, y_test)

    model_path = MODEL_DIR / "xgb_v5_optuna.json"
    model.get_booster().save_model(str(model_path))

    # figures
    plt.figure(figsize=(10, 5))
    plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_optimization_history.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_param_importances.png", dpi=220)
    plt.close()

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(DATA_CSV),
        "rows_after_v5_filter_and_balance": int(len(df)),
        "split_sizes": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "best_trial_balanced_accuracy": float(study.best_value),
        "best_params": best_params,
        "results": {
            "train": train_res,
            "val": val_res,
            "test": test_res,
        },
        "study_db": str(STUDY_DB),
        "model_path": str(model_path),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (RESULTS_DIR / "evaluation_results.json").write_text(json.dumps(report, indent=2))

    print(f"Best val bal_acc: {study.best_value:.4f}")
    print(f"Test acc: {test_res['accuracy']:.4f}")
    print(f"Test bal_acc: {test_res['balanced_accuracy']:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved report: {RESULTS_DIR / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()

