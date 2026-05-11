#!/usr/bin/env python3
"""
Train 9 F1 channel models with channel-specific class sets.

Key idea:
- One model per channel.
- Each channel predicts only algorithms with enough support in that channel.
- Output still one algorithm, but candidate set is dynamic per channel.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 42

DATA_CSV = Path("/Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2_hard_m0p05_balanced.csv")
INDEX_CSV = Path("/Volumes/k/thesis_data/f1_only_1m_packed/index.csv")

ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ROOT / "models" / "f1_9_channel_models_dynamic_v2"
RESULTS_ROOT = ROOT / "results" / "f1_9_channel_models_dynamic_v2"

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
    eval_metric="mlogloss",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train per-channel models with dynamic class sets.")
    p.add_argument("--data", type=Path, default=DATA_CSV)
    p.add_argument("--index", type=Path, default=INDEX_CSV)
    p.add_argument("--min-class-count", type=int, default=20)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--val-frac-of-temp", type=float, default=0.50)
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

    print("=" * 80)
    print("TRAIN F1 9 CHANNEL MODELS (DYNAMIC CLASSES V2)")
    print("=" * 80)
    print(f"Data:  {args.data}")
    print(f"Index: {args.index}")
    print(f"Min class count per channel: {args.min_class_count}")

    if not args.data.exists():
        raise SystemExit(f"Missing data: {args.data}")
    if not args.index.exists():
        raise SystemExit(f"Missing index: {args.index}")

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = pd.read_csv(args.data)
    if "channel" not in df.columns:
        idx = pd.read_csv(args.index, usecols=["file", "channel"])
        df = df.merge(idx, on="file", how="left", validate="many_to_one")
    if "best_algorithm_v2" not in df.columns:
        raise SystemExit("Missing best_algorithm_v2 in dataset.")
    if df["channel"].isna().any():
        raise SystemExit("Some rows have missing channel after join.")

    cols = feature_cols(df)
    channels = sorted(df["channel"].unique().tolist())

    manifest: dict[str, object] = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(args.data),
        "index": str(args.index),
        "min_class_count": int(args.min_class_count),
        "feature_columns": cols,
        "channels": {},
    }

    for ch in channels:
        sub = df[df["channel"] == ch].copy()
        raw_counts = sub["best_algorithm_v2"].value_counts()
        keep = raw_counts[raw_counts >= args.min_class_count].index.tolist()
        dropped = sorted(set(raw_counts.index.tolist()) - set(keep))

        if keep:
            sub = sub[sub["best_algorithm_v2"].isin(keep)].copy()
        classes = sorted(sub["best_algorithm_v2"].unique().tolist())

        if len(classes) < 2:
            manifest["channels"][ch] = {
                "status": "skipped",
                "reason": "need >=2 supported classes",
                "rows_raw": int(len(df[df["channel"] == ch])),
                "rows_used": int(len(sub)),
                "class_counts_before_drop": {k: int(v) for k, v in raw_counts.to_dict().items()},
                "kept_classes": classes,
                "dropped_classes": dropped,
            }
            print(f"[{ch}] skipped: supported classes={classes}")
            continue

        min_count = int(sub["best_algorithm_v2"].value_counts().min())
        if min_count < 3:
            manifest["channels"][ch] = {
                "status": "skipped",
                "reason": "insufficient support for stratified splits",
                "rows_raw": int(len(df[df['channel'] == ch])),
                "rows_used": int(len(sub)),
                "class_counts_before_drop": {k: int(v) for k, v in raw_counts.to_dict().items()},
                "kept_classes": classes,
                "dropped_classes": dropped,
            }
            print(f"[{ch}] skipped: min class count after drop={min_count}")
            continue

        X = sub[cols].values
        y = sub["best_algorithm_v2"].values

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=args.test_size, stratify=y, random_state=SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=args.val_frac_of_temp, stratify=y_temp, random_state=SEED
        )

        le = LabelEncoder().fit(classes)
        y_train_enc = le.transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)

        params = dict(XGB_PARAMS)
        params["num_class"] = len(classes)
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train,
            y_train_enc,
            sample_weight=compute_sample_weights(y_train_enc),
            eval_set=[(X_train, y_train_enc), (X_val, y_val_enc)],
            verbose=False,
        )

        yp_enc = model.predict(X_test)
        yp = le.inverse_transform(yp_enc)
        acc = float(accuracy_score(y_test, yp))
        bacc = float(balanced_accuracy_score(y_test, yp))

        ch_model_dir = MODEL_ROOT / ch
        ch_result_dir = RESULTS_ROOT / ch
        ch_model_dir.mkdir(parents=True, exist_ok=True)
        ch_result_dir.mkdir(parents=True, exist_ok=True)

        model_path = ch_model_dir / "xgb_model.json"
        classes_path = ch_model_dir / "classes.json"
        model.get_booster().save_model(str(model_path))
        classes_path.write_text(json.dumps(classes, indent=2))

        report = {
            "channel": ch,
            "rows_raw": int(len(df[df["channel"] == ch])),
            "rows_used": int(len(sub)),
            "kept_classes": classes,
            "dropped_classes": dropped,
            "class_counts_before_drop": {k: int(v) for k, v in raw_counts.to_dict().items()},
            "metrics": {
                "test_accuracy": acc,
                "test_balanced_accuracy": bacc,
                "confusion_matrix": confusion_matrix(y_test, yp, labels=classes).tolist(),
                "classification_report": classification_report(
                    y_test, yp, labels=classes, output_dict=True, zero_division=0
                ),
            },
            "model_path": str(model_path),
            "classes_path": str(classes_path),
        }
        (ch_result_dir / "evaluation_results.json").write_text(json.dumps(report, indent=2))

        manifest["channels"][ch] = {
            "status": "trained",
            "rows_raw": int(len(df[df["channel"] == ch])),
            "rows_used": int(len(sub)),
            "kept_classes": classes,
            "dropped_classes": dropped,
            "class_counts_before_drop": {k: int(v) for k, v in raw_counts.to_dict().items()},
            "test_accuracy": acc,
            "test_balanced_accuracy": bacc,
            "model_path": str(model_path),
            "classes_path": str(classes_path),
        }
        print(f"[{ch}] classes={classes} acc={acc:.3f} bal_acc={bacc:.3f}")

    manifest["elapsed_sec"] = round(time.time() - t0, 2)
    (RESULTS_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved: {RESULTS_ROOT / 'manifest.json'}")


if __name__ == "__main__":
    main()

