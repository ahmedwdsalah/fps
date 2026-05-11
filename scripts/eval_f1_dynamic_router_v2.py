#!/usr/bin/env python3
"""
Evaluate dynamic per-channel router models on a unified holdout split.

Compares:
1) Routed model predictions (channel-specific model; fallback = channel-SBS)
2) Channel-SBS baseline (most frequent class per channel from train split)
3) Global-SBS baseline (most frequent class globally from train split)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42

DATA_CSV = Path("/Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2_hard_m0p05_balanced.csv")
INDEX_CSV = Path("/Volumes/k/thesis_data/f1_only_1m_packed/index.csv")
MODEL_ROOT = Path("/Users/ahmed/Desktop/thesis/My-Master-thesis/models/f1_9_channel_models_dynamic_v2")
RESULTS_ROOT = Path("/Users/ahmed/Desktop/thesis/My-Master-thesis/results/f1_9_channel_models_dynamic_v2")
OUT_JSON = RESULTS_ROOT / "router_eval.json"


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
    cols = []
    for c in df.columns:
        if c in banned or c.startswith("time_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def load_channel_models() -> dict[str, tuple[xgb.XGBClassifier, list[str]]]:
    out: dict[str, tuple[xgb.XGBClassifier, list[str]]] = {}
    if not MODEL_ROOT.exists():
        return out
    for ch_dir in MODEL_ROOT.iterdir():
        if not ch_dir.is_dir():
            continue
        model_path = ch_dir / "xgb_model.json"
        classes_path = ch_dir / "classes.json"
        if not model_path.exists() or not classes_path.exists():
            continue
        classes = json.loads(classes_path.read_text())
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        out[ch_dir.name] = (model, classes)
    return out


def main() -> None:
    print("=" * 80)
    print("EVAL F1 DYNAMIC ROUTER V2")
    print("=" * 80)
    print(f"Data:   {DATA_CSV}")
    print(f"Index:  {INDEX_CSV}")
    print(f"Models: {MODEL_ROOT}")

    if not DATA_CSV.exists():
        raise SystemExit(f"Missing data: {DATA_CSV}")
    if not INDEX_CSV.exists():
        raise SystemExit(f"Missing index: {INDEX_CSV}")

    df = pd.read_csv(DATA_CSV)
    if "channel" not in df.columns:
        idx = pd.read_csv(INDEX_CSV, usecols=["file", "channel"])
        df = df.merge(idx, on="file", how="left", validate="many_to_one")
    if df["channel"].isna().any():
        raise SystemExit("Missing channel after join.")

    cols = feature_cols(df)
    y = df["best_algorithm_v2"].values

    train_df, test_df = train_test_split(
        df, test_size=0.30, stratify=df["best_algorithm_v2"], random_state=SEED
    )

    # Baselines from training split only (honest setup)
    global_sbs = train_df["best_algorithm_v2"].mode().iloc[0]
    channel_sbs = (
        train_df.groupby("channel")["best_algorithm_v2"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )

    models = load_channel_models()
    if not models:
        raise SystemExit(f"No channel models found in {MODEL_ROOT}")

    routed_pred: list[str] = []
    channel_sbs_pred: list[str] = []
    global_sbs_pred: list[str] = []
    fallback_counter: Counter[str] = Counter()

    for _, row in test_df.iterrows():
        ch = row["channel"]
        x = row[cols].values.astype(np.float64).reshape(1, -1)

        # Baselines
        ch_sbs = channel_sbs.get(ch, global_sbs)
        channel_sbs_pred.append(ch_sbs)
        global_sbs_pred.append(global_sbs)

        # Router prediction
        if ch in models:
            model, classes = models[ch]
            pred_raw = model.predict(x)
            if np.ndim(pred_raw) == 1:
                pred_idx = int(pred_raw[0])
            elif np.ndim(pred_raw) == 2:
                pred_idx = int(np.argmax(pred_raw[0]))
            else:
                pred_idx = -1
            if 0 <= pred_idx < len(classes):
                routed_pred.append(classes[pred_idx])
            else:
                routed_pred.append(ch_sbs)
                fallback_counter["bad_class_index"] += 1
        else:
            routed_pred.append(ch_sbs)
            fallback_counter["missing_channel_model"] += 1

    y_true = test_df["best_algorithm_v2"].tolist()

    res = {
        "dataset": str(DATA_CSV),
        "index": str(INDEX_CSV),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "models_loaded": sorted(models.keys()),
        "global_sbs_class": global_sbs,
        "fallback_counts": dict(fallback_counter),
        "metrics": {
            "routed": {
                "accuracy": float(accuracy_score(y_true, routed_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, routed_pred)),
            },
            "channel_sbs": {
                "accuracy": float(accuracy_score(y_true, channel_sbs_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, channel_sbs_pred)),
            },
            "global_sbs": {
                "accuracy": float(accuracy_score(y_true, global_sbs_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, global_sbs_pred)),
            },
        },
    }

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2))

    print(f"Test rows: {len(test_df):,}")
    print("Metrics:")
    print(f"  routed      acc={res['metrics']['routed']['accuracy']:.3f}  bal_acc={res['metrics']['routed']['balanced_accuracy']:.3f}")
    print(f"  channel_sbs acc={res['metrics']['channel_sbs']['accuracy']:.3f}  bal_acc={res['metrics']['channel_sbs']['balanced_accuracy']:.3f}")
    print(f"  global_sbs  acc={res['metrics']['global_sbs']['accuracy']:.3f}  bal_acc={res['metrics']['global_sbs']['balanced_accuracy']:.3f}")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
