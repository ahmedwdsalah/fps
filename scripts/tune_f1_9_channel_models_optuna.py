#!/usr/bin/env python3
"""
Optuna tuning + training for 9 channel-specific F1 models.

Pipeline per channel:
1) load channel subset
2) drop ultra-rare classes
3) Optuna tune on validation balanced accuracy
4) retrain best model
5) save model + params + metrics
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
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

SEED = 42
MIN_CLASS_COUNT = 3

DATA_CSV = Path("/Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2_hard_m0p05_balanced.csv")
INDEX_CSV = Path("/Volumes/k/thesis_data/f1_only_1m_packed/index.csv")

ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ROOT / "models" / "f1_9_channel_models_optuna"
RESULTS_ROOT = ROOT / "results" / "f1_9_channel_models_optuna"
STUDY_DB = RESULTS_ROOT / "optuna_studies.db"
FIG_ROOT = RESULTS_ROOT / "figures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--timeout-sec", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    return p.parse_args()


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    w = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([w[i] for i in y], dtype=np.float64)


def feature_cols(df: pd.DataFrame) -> list[str]:
    banned = {
        "file",
        "domain",
        "channel",
        "best_algorithm",
        "best_algorithm_v2",
        "winner_margin_v2",
        "year",
        "round",
        "session",
        "driver",
        "lap",
        "event",
        "dtype",
    }
    out = []
    for c in df.columns:
        if c in banned or c.startswith("time_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    if not out:
        raise ValueError("No numeric feature columns found.")
    return out


def main() -> None:
    args = parse_args()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    if not DATA_CSV.exists():
        raise SystemExit(f"Missing dataset: {DATA_CSV}")
    if not INDEX_CSV.exists():
        raise SystemExit(f"Missing index: {INDEX_CSV}")

    print("=" * 80)
    print("OPTUNA TUNE F1 9 CHANNEL MODELS")
    print("=" * 80)
    print(f"Data: {DATA_CSV}")
    print(f"Trials/channel: {args.trials}")
    print(f"Workers: {args.workers}")

    t0 = time.time()
    df = pd.read_csv(DATA_CSV)
    if "channel" not in df.columns:
        idx = pd.read_csv(INDEX_CSV, usecols=["file", "channel"])
        df = df.merge(idx, on="file", how="left", validate="many_to_one")
    if "best_algorithm_v2" not in df.columns:
        raise SystemExit("Missing best_algorithm_v2.")
    if df["channel"].isna().any():
        raise SystemExit("Missing channel labels after join.")

    cols = feature_cols(df)
    channels = sorted(df["channel"].unique().tolist())

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(DATA_CSV),
        "feature_columns": cols,
        "trials_per_channel": args.trials,
        "channels": {},
    }

    for ch in channels:
        sub = df[df["channel"] == ch].copy()
        raw_counts = sub["best_algorithm_v2"].value_counts()
        keep = raw_counts[raw_counts >= MIN_CLASS_COUNT].index.tolist()
        dropped = sorted(set(raw_counts.index.tolist()) - set(keep))
        if keep:
            sub = sub[sub["best_algorithm_v2"].isin(keep)].copy()
        classes = sorted(sub["best_algorithm_v2"].unique().tolist())
        if len(classes) < 2:
            manifest["channels"][ch] = {"status": "skipped", "reason": "need >=2 classes", "dropped_rare_classes": dropped}
            print(f"[{ch}] skipped")
            continue

        X = sub[cols].values
        y = sub["best_algorithm_v2"].values
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
        )

        le = LabelEncoder().fit(classes)
        y_train_enc = le.transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "multi:softprob",
                "num_class": len(classes),
                "eval_metric": "mlogloss",
                "tree_method": "hist",
                "random_state": SEED,
                "n_jobs": args.workers,
                "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
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
                sample_weight=compute_sample_weights(y_train_enc),
                eval_set=[(X_val, y_val_enc)],
                verbose=False,
            )
            yp = le.inverse_transform(model.predict(X_val))
            return float(balanced_accuracy_score(y_val, yp))

        study_name = f"f1_{ch}_algo_v2"
        study = optuna.create_study(
            study_name=study_name,
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
            "num_class": len(classes),
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": SEED,
            "n_jobs": args.workers,
            **best_params,
        }
        final_model = xgb.XGBClassifier(**final_params)
        final_model.fit(
            X_train,
            y_train_enc,
            sample_weight=compute_sample_weights(y_train_enc),
            eval_set=[(X_train, y_train_enc), (X_val, y_val_enc)],
            verbose=False,
        )

        yp_enc = final_model.predict(X_test)
        yp = le.inverse_transform(yp_enc)
        acc = float(accuracy_score(y_test, yp))
        bacc = float(balanced_accuracy_score(y_test, yp))

        ch_model_dir = MODEL_ROOT / ch
        ch_result_dir = RESULTS_ROOT / ch
        ch_fig_dir = FIG_ROOT / ch
        ch_model_dir.mkdir(parents=True, exist_ok=True)
        ch_result_dir.mkdir(parents=True, exist_ok=True)
        ch_fig_dir.mkdir(parents=True, exist_ok=True)

        model_path = ch_model_dir / "xgb_model.json"
        classes_path = ch_model_dir / "classes.json"
        final_model.get_booster().save_model(str(model_path))
        classes_path.write_text(json.dumps(classes, indent=2))

        report = {
            "channel": ch,
            "rows": int(len(sub)),
            "classes": classes,
            "dropped_rare_classes": dropped,
            "best_trial_value_balanced_accuracy": float(study.best_value),
            "best_params": best_params,
            "test_metrics": {
                "accuracy": acc,
                "balanced_accuracy": bacc,
                "confusion_matrix": confusion_matrix(y_test, yp, labels=classes).tolist(),
                "classification_report": classification_report(
                    y_test, yp, labels=classes, output_dict=True, zero_division=0
                ),
            },
            "model_path": str(model_path),
            "classes_path": str(classes_path),
        }
        (ch_result_dir / "evaluation_results.json").write_text(json.dumps(report, indent=2))

        # Figures: optimization history / parameter importances / slice / parallel coordinate
        try:
            plt.figure(figsize=(10, 5))
            plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(ch_fig_dir / "01_optimization_history.png", dpi=200)
            plt.close()

            plt.figure(figsize=(10, 6))
            plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(ch_fig_dir / "02_param_importances.png", dpi=200)
            plt.close()

            plt.figure(figsize=(10, 6))
            plot_slice(study)
            plt.tight_layout()
            plt.savefig(ch_fig_dir / "03_slice.png", dpi=200)
            plt.close()

            plt.figure(figsize=(11, 7))
            plot_parallel_coordinate(study)
            plt.tight_layout()
            plt.savefig(ch_fig_dir / "04_parallel_coordinate.png", dpi=200)
            plt.close()
        except Exception:
            pass

        # Figures: confusion matrix + per-class F1 on test
        cm = np.array(report["test_metrics"]["confusion_matrix"], dtype=float)
        cm_row = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_row,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
        )
        plt.title(f"{ch} - Test Confusion Matrix (Row-normalized)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(ch_fig_dir / "05_confusion_matrix_test.png", dpi=220)
        plt.close()

        rep = report["test_metrics"]["classification_report"]
        f1_vals = [rep[c]["f1-score"] for c in classes]
        plt.figure(figsize=(10, 4))
        plt.bar(classes, f1_vals, color="#1f77b4")
        plt.ylim(0, 1)
        plt.title(f"{ch} - Test Per-Class F1")
        plt.ylabel("F1-score")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(ch_fig_dir / "06_per_class_f1_test.png", dpi=220)
        plt.close()

        manifest["channels"][ch] = {
            "status": "trained",
            "rows": int(len(sub)),
            "classes": classes,
            "best_trial_balanced_accuracy": float(study.best_value),
            "test_accuracy": acc,
            "test_balanced_accuracy": bacc,
            "best_params": best_params,
            "model_path": str(model_path),
        }
        print(f"[{ch}] test_acc={acc:.3f} test_bal_acc={bacc:.3f} best_val_bal_acc={study.best_value:.3f}")

    manifest["elapsed_sec"] = round(time.time() - t0, 2)
    manifest["study_db"] = str(STUDY_DB)
    (RESULTS_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Global comparison figure: tuned test balanced accuracy across channels
    tuned_channels = [
        (ch, d["test_balanced_accuracy"])
        for ch, d in manifest["channels"].items()
        if d.get("status") == "trained"
    ]
    if tuned_channels:
        tuned_channels.sort(key=lambda x: x[1], reverse=True)
        labels = [c for c, _ in tuned_channels]
        vals = [v for _, v in tuned_channels]
        plt.figure(figsize=(11, 5))
        plt.bar(labels, vals, color="#2ca02c")
        plt.ylim(0, 1)
        plt.title("Tuned Test Balanced Accuracy by Channel")
        plt.ylabel("Balanced Accuracy")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(FIG_ROOT / "00_balanced_accuracy_by_channel.png", dpi=220)
        plt.close()

    print(f"\nSaved manifest: {RESULTS_ROOT / 'manifest.json'}")
    print(f"Study DB: {STUDY_DB}")


if __name__ == "__main__":
    main()
