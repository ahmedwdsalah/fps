#!/usr/bin/env python3
"""
Export v5-optuna global model assets for the WebSHAP demo.

Outputs:
  webshap/examples/demo/public/models/sorting/v5_global.onnx
  webshap/examples/demo/public/data/sorting-v5.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import xgboost as xgb
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES


ROOT = SCRIPT_DIR.parent
PUBLIC_ROOT = ROOT / "webshap" / "examples" / "demo" / "public"
MODEL_OUT = PUBLIC_ROOT / "models" / "sorting" / "v5_global.onnx"
DATA_OUT = PUBLIC_ROOT / "data" / "sorting-v5.json"
DATA_CSV = ROOT / "data" / "training_dataset.csv"
MODEL_IN = ROOT / "models" / "xgboost_v5_optuna" / "xgb_v5_optuna.json"
SEED = 42

ALGORITHMS = ["introsort", "heapsort", "timsort"]
TIME_COLS = {
    "introsort": "time_introsort",
    "heapsort": "time_heapsort",
    "timsort": "time_timsort",
}

FEATURE_GROUPS = {
    "length_norm": "size",
    "adj_sorted_ratio": "ordering",
    "runs_ratio": "ordering",
    "inversion_ratio": "ordering",
    "longest_run_ratio": "ordering",
    "mean_abs_diff_norm": "ordering",
    "duplicate_ratio": "repetition",
    "top1_freq_ratio": "repetition",
    "top5_freq_ratio": "repetition",
    "dispersion_ratio": "distribution",
    "entropy_ratio": "distribution",
    "skewness_t": "distribution",
    "kurtosis_excess_t": "distribution",
    "outlier_ratio": "distribution",
    "iqr_norm": "robust scale",
    "mad_norm": "robust scale",
}

FEATURE_DISPLAY = {
    "length_norm": "Length norm",
    "adj_sorted_ratio": "Adjacent sorted ratio",
    "duplicate_ratio": "Duplicate ratio",
    "dispersion_ratio": "Dispersion ratio",
    "runs_ratio": "Runs ratio",
    "inversion_ratio": "Inversion ratio",
    "entropy_ratio": "Entropy ratio",
    "skewness_t": "Skewness",
    "kurtosis_excess_t": "Kurtosis excess",
    "longest_run_ratio": "Longest run ratio",
    "iqr_norm": "IQR norm",
    "mad_norm": "MAD norm",
    "top1_freq_ratio": "Top-1 frequency",
    "top5_freq_ratio": "Top-5 frequency",
    "outlier_ratio": "Outlier ratio",
    "mean_abs_diff_norm": "Mean absolute diff",
}


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


def prepare_v5_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    tcols = df[[TIME_COLS[a] for a in ALGORITHMS]].values
    st = np.sort(tcols, axis=1)
    margin = (st[:, 1] - st[:, 0]) / (st[:, 1] + 1e-15)
    keep = (margin >= 0.05) | (df["n_elements"].values >= 2000)
    df = df[keep].reset_index(drop=True)
    df = balanced_undersample(df, "best_algorithm", max_ratio=3.0)
    return df


def export_onnx_model(model_path: Path, out_path: Path, n_features: int) -> None:
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    onx = convert_xgboost(
        model,
        initial_types=[("float_input", FloatTensorType([None, n_features]))],
        target_opset=15,
    )
    onnx.checker.check_model(onx)
    out_path.write_bytes(onx.SerializeToString())


def read_ort_probs(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    outs = session.run(None, {"float_input": x.astype(np.float32)})
    for out in outs:
        arr = np.array(out)
        if arr.ndim == 2 and arr.shape[1] == len(ALGORITHMS):
            return arr.astype(np.float64)
    raise RuntimeError("Could not find probability output with shape [N, num_class].")


def validate_model(model_path: Path, onnx_path: Path, x_sample: np.ndarray) -> float:
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    xgb_probs = model.predict_proba(x_sample.astype(np.float32)).astype(np.float64)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_probs = read_ort_probs(session, x_sample)
    return float(np.max(np.abs(xgb_probs - ort_probs)))


def make_background(df: pd.DataFrame, cols: list[str]) -> list[list[float]]:
    # Small real subset median + mean for stable SHAP background.
    med = df[cols].median(numeric_only=True).astype(float).tolist()
    mean = df[cols].mean(numeric_only=True).astype(float).tolist()
    return [med, mean]


def make_rows(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    rows = []
    # Preserve real domain mix: up to 8 rows per domain.
    for domain, sub in df.groupby("domain", sort=True):
        take = sub.sample(n=min(8, len(sub)), random_state=SEED)
        for _, row in take.iterrows():
            rows.append(
                {
                    "id": str(row["file"]),
                    "file": str(row["file"]),
                    "domain": str(domain),
                    "n_elements": int(row["n_elements"]),
                    "trueLabel": str(row["best_algorithm"]),
                    "x": [float(row[c]) for c in cols],
                    "supportedLabels": ALGORITHMS,
                    "timesSec": {a: float(row[TIME_COLS[a]]) for a in ALGORITHMS},
                }
            )
    return rows


def compute_runtime_metrics(df: pd.DataFrame) -> dict:
    totals = {a: float(df[TIME_COLS[a]].sum()) for a in ALGORITHMS}
    sbs_algo = min(totals, key=totals.get)
    t_sbs = float(df[TIME_COLS[sbs_algo]].mean())
    vbs = df[[TIME_COLS[a] for a in ALGORITHMS]].min(axis=1).astype(float).to_numpy()
    t_vbs = float(vbs.mean())
    return {
        "rows": int(len(df)),
        "sbs_algorithm": sbs_algo,
        "t_sbs_sec": t_sbs,
        "t_vbs_sec": t_vbs,
    }


def main() -> None:
    if not MODEL_IN.exists():
        raise SystemExit(f"Missing v5 model: {MODEL_IN}")

    PUBLIC_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)

    df = prepare_v5_dataset()
    cols = list(FEATURE_NAMES)

    export_onnx_model(MODEL_IN, MODEL_OUT, len(cols))

    sample = df[cols].head(256).to_numpy(dtype=np.float32)
    max_abs_diff = validate_model(MODEL_IN, MODEL_OUT, sample)

    payload = {
        "mode": "v5_global",
        "featureOrder": cols,
        "featureInfo": {
            c: {
                "displayName": FEATURE_DISPLAY.get(c, c),
                "description": FEATURE_GROUPS.get(c, "feature"),
                "group": FEATURE_GROUPS.get(c, "feature"),
                "requiresInt": False,
            }
            for c in cols
        },
        "classLabels": ALGORITHMS,
        "model": "models/sorting/v5_global.onnx",
        "runtimeMetrics": compute_runtime_metrics(df),
        "backgroundData": make_background(df, cols),
        "rows": make_rows(df, cols),
        "validation": {
            "max_abs_probability_diff": max_abs_diff,
            "rows_after_v5_filter_and_balance": int(len(df)),
        },
    }

    DATA_OUT.write_text(json.dumps(payload, indent=2))

    print("=" * 80)
    print("WEBSHAP V5 ASSET EXPORT COMPLETE")
    print("=" * 80)
    print(f"Model: {MODEL_OUT}")
    print(f"Data:  {DATA_OUT}")
    print(f"Rows exported: {len(payload['rows'])}")
    print(f"Validation max_abs_probability_diff: {max_abs_diff:.8f}")


if __name__ == "__main__":
    main()
