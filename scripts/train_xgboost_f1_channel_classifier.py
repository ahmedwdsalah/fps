#!/usr/bin/env python3
"""
Train one XGBoost multiclass model to predict F1 channel (9 classes).

Data source:
  - Features: /Volumes/k/thesis_data/f1_only_1m_packed/training_dataset.csv
  - Channel label: joined from /Volumes/k/thesis_data/f1_only_1m_packed/index.csv via `file`
"""

from __future__ import annotations

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

DATA_CSV = Path("/Volumes/k/thesis_data/f1_only_1m_packed/training_dataset.csv")
INDEX_CSV = Path("/Volumes/k/thesis_data/f1_only_1m_packed/index.csv")

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "xgboost_f1_channel"
RESULTS_DIR = ROOT / "results" / "xgboost_f1_channel"

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=8,
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


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    w = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([w[i] for i in y], dtype=np.float64)


def resolve_feature_columns(df: pd.DataFrame) -> list[str]:
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
    cols: list[str] = []
    for c in df.columns:
        if c in banned:
            continue
        if c.startswith("time_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("No numeric feature columns found.")
    return cols


def main() -> None:
    print("=" * 80)
    print("XGBOOST F1 CHANNEL CLASSIFIER")
    print("=" * 80)
    print(f"Features file: {DATA_CSV}")
    print(f"Index file:    {INDEX_CSV}")

    if not DATA_CSV.exists():
        raise SystemExit(f"Missing features file: {DATA_CSV}")
    if not INDEX_CSV.exists():
        raise SystemExit(f"Missing index file: {INDEX_CSV}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df_feat = pd.read_csv(DATA_CSV)
    df_idx = pd.read_csv(INDEX_CSV, usecols=["file", "channel"])
    df = df_feat.merge(df_idx, on="file", how="inner", validate="one_to_one")

    if df.empty:
        raise SystemExit("Merged dataframe is empty.")
    if df["channel"].isna().any():
        raise SystemExit("Found NaN in channel labels after merge.")

    feature_cols = resolve_feature_columns(df)
    classes = sorted(df["channel"].unique().tolist())
    print(f"Rows: {len(df):,}")
    print(f"Classes: {classes}")
    print(f"Features: {len(feature_cols)}")

    X = df[feature_cols].values
    y = df["channel"].values

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

    def eval_split(Xs, ys):
        yp_enc = model.predict(Xs)
        yp = le.inverse_transform(yp_enc)
        return {
            "accuracy": float(accuracy_score(ys, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(ys, yp)),
            "confusion_matrix": confusion_matrix(ys, yp, labels=classes).tolist(),
            "classification_report": classification_report(
                ys, yp, labels=classes, output_dict=True, zero_division=0
            ),
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "features_csv": str(DATA_CSV),
            "index_csv": str(INDEX_CSV),
            "rows_total": int(len(df)),
            "rows_train": int(len(y_train)),
            "rows_val": int(len(y_val)),
            "rows_test": int(len(y_test)),
            "class_distribution": {k: int(v) for k, v in df["channel"].value_counts().to_dict().items()},
        },
        "features": feature_cols,
        "classes": classes,
        "xgb_params": params,
        "results": {
            "train": eval_split(X_train, y_train),
            "val": eval_split(X_val, y_val),
            "test": eval_split(X_test, y_test),
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }

    model_path = MODEL_DIR / "xgb_channel_classifier.json"
    model.get_booster().save_model(str(model_path))
    (RESULTS_DIR / "evaluation_results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame(
        {
            "actual": y_test,
            "predicted": le.inverse_transform(model.predict(X_test)),
        }
    ).to_csv(RESULTS_DIR / "predictions_test.csv", index=False)

    print(
        f"TEST: acc={results['results']['test']['accuracy']:.3f}  "
        f"bal_acc={results['results']['test']['balanced_accuracy']:.3f}"
    )
    print(f"Model:   {model_path}")
    print(f"Results: {RESULTS_DIR / 'evaluation_results.json'}")
    print(f"Done in {results['elapsed_sec']}s")


if __name__ == "__main__":
    main()

