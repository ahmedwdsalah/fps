#!/usr/bin/env python3
"""
Strict evaluation for dynamic per-channel routing.

Protocol:
1) Split dataset once into train/test.
2) Train channel models on train split only.
3) Evaluate routed inference on untouched test split.
4) Compare against channel-SBS and global-SBS (computed from train split only).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
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
MODEL_ROOT = ROOT / "models" / "f1_9_channel_models_dynamic_v2_strict"
RESULTS_ROOT = ROOT / "results" / "f1_9_channel_models_dynamic_v2_strict"

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
    p = argparse.ArgumentParser(description="Strict train+eval for dynamic per-channel router.")
    p.add_argument("--data", type=Path, default=DATA_CSV)
    p.add_argument("--index", type=Path, default=INDEX_CSV)
    p.add_argument("--min-class-count", type=int, default=20)
    p.add_argument("--test-size", type=float, default=0.30)
    p.add_argument("--val-frac-of-train", type=float, default=0.20)
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
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STRICT TRAIN+EVAL: F1 DYNAMIC ROUTER V2")
    print("=" * 80)
    print(f"Data:  {args.data}")
    print(f"Index: {args.index}")
    print(f"Min class count per channel: {args.min_class_count}")

    if not args.data.exists():
        raise SystemExit(f"Missing data: {args.data}")
    if not args.index.exists():
        raise SystemExit(f"Missing index: {args.index}")

    t0 = time.time()

    df = pd.read_csv(args.data)
    if "channel" not in df.columns:
        idx = pd.read_csv(args.index, usecols=["file", "channel"])
        df = df.merge(idx, on="file", how="left", validate="many_to_one")
    if df["channel"].isna().any():
        raise SystemExit("Missing channel after join.")

    cols = feature_cols(df)

    # Global strict split once
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df["best_algorithm_v2"],
        random_state=SEED,
    )

    # Baselines from train split only
    global_sbs = train_df["best_algorithm_v2"].mode().iloc[0]
    channel_sbs = (
        train_df.groupby("channel")["best_algorithm_v2"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )

    channels = sorted(df["channel"].unique().tolist())
    channel_models: dict[str, tuple[xgb.XGBClassifier, list[str]]] = {}
    channel_manifest: dict[str, dict] = {}

    # Train per-channel models from train split only
    for ch in channels:
        sub = train_df[train_df["channel"] == ch].copy()
        raw_counts = sub["best_algorithm_v2"].value_counts()
        keep = raw_counts[raw_counts >= args.min_class_count].index.tolist()
        dropped = sorted(set(raw_counts.index.tolist()) - set(keep))

        if keep:
            sub = sub[sub["best_algorithm_v2"].isin(keep)].copy()
        classes = sorted(sub["best_algorithm_v2"].unique().tolist())

        if len(classes) < 2:
            channel_manifest[ch] = {
                "status": "skipped",
                "reason": "need >=2 supported classes",
                "rows_train_raw": int(len(train_df[train_df["channel"] == ch])),
                "rows_train_used": int(len(sub)),
                "kept_classes": classes,
                "dropped_classes": dropped,
                "class_counts_before_drop": {k: int(v) for k, v in raw_counts.to_dict().items()},
            }
            print(f"[{ch}] skipped")
            continue

        y_counts = sub["best_algorithm_v2"].value_counts()
        if int(y_counts.min()) < 3:
            channel_manifest[ch] = {
                "status": "skipped",
                "reason": "insufficient support for stratified train/val split",
                "rows_train_raw": int(len(train_df[train_df["channel"] == ch])),
                "rows_train_used": int(len(sub)),
                "kept_classes": classes,
                "dropped_classes": dropped,
                "class_counts_before_drop": {k: int(v) for k, v in raw_counts.to_dict().items()},
            }
            print(f"[{ch}] skipped (min class < 3)")
            continue

        X = sub[cols].values
        y = sub["best_algorithm_v2"].values
        X_tr, X_val, y_tr, y_val = train_test_split(
            X,
            y,
            test_size=args.val_frac_of_train,
            stratify=y,
            random_state=SEED,
        )

        le = LabelEncoder().fit(classes)
        y_tr_enc = le.transform(y_tr)
        y_val_enc = le.transform(y_val)

        params = dict(XGB_PARAMS)
        params["num_class"] = len(classes)
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr,
            y_tr_enc,
            sample_weight=compute_sample_weights(y_tr_enc),
            eval_set=[(X_tr, y_tr_enc), (X_val, y_val_enc)],
            verbose=False,
        )

        channel_models[ch] = (model, classes)
        ch_model_dir = MODEL_ROOT / ch
        ch_model_dir.mkdir(parents=True, exist_ok=True)
        model.get_booster().save_model(str(ch_model_dir / "xgb_model.json"))
        (ch_model_dir / "classes.json").write_text(json.dumps(classes, indent=2))

        channel_manifest[ch] = {
            "status": "trained",
            "rows_train_raw": int(len(train_df[train_df["channel"] == ch])),
            "rows_train_used": int(len(sub)),
            "kept_classes": classes,
            "dropped_classes": dropped,
            "class_counts_before_drop": {k: int(v) for k, v in raw_counts.to_dict().items()},
            "model_path": str(ch_model_dir / "xgb_model.json"),
            "classes_path": str(ch_model_dir / "classes.json"),
        }
        print(f"[{ch}] trained classes={classes}")

    # Strict test evaluation
    y_true = test_df["best_algorithm_v2"].tolist()
    routed_pred: list[str] = []
    channel_sbs_pred: list[str] = []
    global_sbs_pred: list[str] = []
    fallback_counter: Counter[str] = Counter()

    for _, row in test_df.iterrows():
        ch = row["channel"]
        x = row[cols].values.astype(np.float64).reshape(1, -1)
        ch_default = channel_sbs.get(ch, global_sbs)
        channel_sbs_pred.append(ch_default)
        global_sbs_pred.append(global_sbs)

        if ch in channel_models:
            model, classes = channel_models[ch]
            pred_raw = model.predict(x)
            # XGBoost may return either class indices (shape: [n])
            # or probability vectors (shape: [n, num_class]), depending on setup.
            if np.ndim(pred_raw) == 1:
                pred_idx = int(pred_raw[0])
            elif np.ndim(pred_raw) == 2:
                pred_idx = int(np.argmax(pred_raw[0]))
            else:
                pred_idx = -1
            if 0 <= pred_idx < len(classes):
                routed_pred.append(classes[pred_idx])
            else:
                routed_pred.append(ch_default)
                fallback_counter["bad_class_index"] += 1
        else:
            routed_pred.append(ch_default)
            fallback_counter["missing_channel_model"] += 1

    metrics = {
        "routed": {
            "accuracy": float(accuracy_score(y_true, routed_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, routed_pred)),
            "confusion_matrix": confusion_matrix(y_true, routed_pred).tolist(),
            "classification_report": classification_report(y_true, routed_pred, output_dict=True, zero_division=0),
        },
        "channel_sbs": {
            "accuracy": float(accuracy_score(y_true, channel_sbs_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, channel_sbs_pred)),
        },
        "global_sbs": {
            "accuracy": float(accuracy_score(y_true, global_sbs_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, global_sbs_pred)),
        },
    }

    out = {
        "timestamp": datetime.now().isoformat(),
        "dataset": str(args.data),
        "index": str(args.index),
        "seed": SEED,
        "min_class_count": int(args.min_class_count),
        "split": {
            "test_size": float(args.test_size),
            "val_frac_of_train": float(args.val_frac_of_train),
            "n_total": int(len(df)),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        },
        "global_sbs_class": global_sbs,
        "fallback_counts": dict(fallback_counter),
        "channel_manifest": channel_manifest,
        "metrics": metrics,
        "elapsed_sec": round(time.time() - t0, 2),
    }

    out_path = RESULTS_ROOT / "strict_router_eval.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\nStrict test metrics:")
    print(f"  routed      acc={metrics['routed']['accuracy']:.3f}  bal_acc={metrics['routed']['balanced_accuracy']:.3f}")
    print(f"  channel_sbs acc={metrics['channel_sbs']['accuracy']:.3f}  bal_acc={metrics['channel_sbs']['balanced_accuracy']:.3f}")
    print(f"  global_sbs  acc={metrics['global_sbs']['accuracy']:.3f}  bal_acc={metrics['global_sbs']['balanced_accuracy']:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
