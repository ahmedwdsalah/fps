#!/usr/bin/env python3
"""
Export the F1 channel XGBoost models and metadata for the WebSHAP demo.

The browser demo consumes:
  webshap/examples/demo/public/models/sorting/*.onnx
  webshap/examples/demo/public/data/sorting-f1.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import xgboost as xgb
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "f1_9_channel_models_optuna" / "manifest.json"
INDEX_CSV = Path("/Volumes/k/thesis_data/f1_only_1m_packed/index.csv")
PUBLIC_ROOT = ROOT / "webshap" / "examples" / "demo" / "public"
MODEL_OUT = PUBLIC_ROOT / "models" / "sorting"
DATA_OUT = PUBLIC_ROOT / "data" / "sorting-f1.json"

CHANNEL_FILE_NAMES = {
    "DRS": "drs.onnx",
    "Distance": "distance.onnx",
    "RPM": "rpm.onnx",
    "Speed": "speed.onnx",
    "Throttle": "throttle.onnx",
    "X": "x.onnx",
    "Y": "y.onnx",
    "Z": "z.onnx",
    "nGear": "gear.onnx",
}

FEATURE_GROUPS = {
    "n_elements": "size",
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
    "n_elements": "Array length",
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

ALGO_TIME_COL = {
    "quick_sort": "time_quick_sort",
    "introsort": "time_introsort",
    "merge_sort": "time_merge_sort",
    "heap_sort": "time_heap_sort",
    "shell_sort": "time_shell_sort",
}


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "channel" not in df.columns:
        idx = pd.read_csv(INDEX_CSV, usecols=["file", "channel"])
        df = df.merge(idx, on="file", how="left", validate="many_to_one")
    if df["channel"].isna().any():
        missing = int(df["channel"].isna().sum())
        raise SystemExit(f"Missing channel after index join: {missing}")
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


def validate_model(model_path: Path, onnx_path: Path, x_sample: np.ndarray) -> float:
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    xgb_probs = model.predict_proba(x_sample.astype(np.float32))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_probs = session.run(None, {"float_input": x_sample.astype(np.float32)})[1]
    return float(np.max(np.abs(xgb_probs - ort_probs)))


def make_background(sub: pd.DataFrame, cols: list[str]) -> list[list[float]]:
    med = sub[cols].median(numeric_only=True).astype(float).tolist()
    return [med]


def make_examples(sub: pd.DataFrame, cols: list[str], classes: list[str]) -> list[dict]:
    examples = []
    sample = sub.sample(n=min(8, len(sub)), random_state=42)
    for _, row in sample.iterrows():
        examples.append(
            {
                "file": str(row["file"]),
                "trueLabel": str(row["best_algorithm_v2"]),
                "x": [float(row[c]) for c in cols],
                "supportedLabels": classes,
                "timesSec": {
                    a: float(row[ALGO_TIME_COL[a]])
                    for a in classes
                },
            }
        )
    return examples


def compute_runtime_metrics(sub: pd.DataFrame, cols: list[str], classes: list[str], model_path: Path) -> dict:
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    x = sub[cols].to_numpy(dtype=np.float32)
    probs = clf.predict_proba(x)
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [classes[int(i)] for i in pred_idx]

    def row_time(row: pd.Series, label: str) -> float:
        return float(row[ALGO_TIME_COL[label]])

    selected_times = np.array([row_time(row, label) for (_, row), label in zip(sub.iterrows(), pred_labels)], dtype=float)
    vbs_times = np.array(
        [min(float(row[ALGO_TIME_COL[a]]) for a in classes) for _, row in sub.iterrows()],
        dtype=float,
    )

    totals_by_algo = {
        a: float(sub[ALGO_TIME_COL[a]].sum())
        for a in classes
    }
    sbs_algo = min(totals_by_algo, key=totals_by_algo.get)
    sbs_times = sub[ALGO_TIME_COL[sbs_algo]].to_numpy(dtype=float)

    t_selector = float(selected_times.mean())
    t_sbs = float(sbs_times.mean())
    t_vbs = float(vbs_times.mean())
    speedup_vs_sbs = float((t_sbs - t_selector) / t_sbs) if t_sbs > 0 else 0.0
    regret_vs_vbs = float(t_selector - t_vbs)
    denom = (t_sbs - t_vbs)
    gap_closed = float((t_sbs - t_selector) / denom) if abs(denom) > 1e-18 else 0.0

    return {
        "rows": int(len(sub)),
        "sbs_algorithm": sbs_algo,
        "t_selector_sec": t_selector,
        "t_sbs_sec": t_sbs,
        "t_vbs_sec": t_vbs,
        "speedup_vs_sbs": speedup_vs_sbs,
        "regret_vs_vbs_sec": regret_vs_vbs,
        "gap_closed": gap_closed,
    }


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}")

    manifest = json.loads(MANIFEST.read_text())
    dataset = Path(manifest["dataset"])
    cols = list(manifest["feature_columns"])
    df = load_dataset(dataset)

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)

    exported_channels = {}
    validation = {}

    for channel, info in manifest["channels"].items():
        if info.get("status") != "trained":
            continue
        model_path = Path(info["model_path"])
        out_name = CHANNEL_FILE_NAMES[channel]
        out_path = MODEL_OUT / out_name
        sub = df[df["channel"] == channel].copy()
        classes = list(info["classes"])
        sub = sub[sub["best_algorithm_v2"].isin(classes)].copy()
        if sub.empty:
            raise SystemExit(f"No data rows for channel {channel}")

        export_onnx_model(model_path, out_path, len(cols))
        x_sample = sub[cols].head(16).to_numpy(dtype=np.float32)
        max_abs_diff = validate_model(model_path, out_path, x_sample)
        validation[channel] = {
            "model": str(out_path.relative_to(PUBLIC_ROOT)),
            "max_abs_probability_diff": max_abs_diff,
        }

        runtime_metrics = compute_runtime_metrics(sub, cols, classes, model_path)

        exported_channels[channel] = {
            "model": f"models/sorting/{out_name}",
            "classes": classes,
            "rows": int(len(sub)),
            "runtimeMetrics": runtime_metrics,
            "backgroundData": make_background(sub, cols),
            "examples": make_examples(sub, cols, classes),
        }

    metadata = {
        "defaultChannel": "Speed",
        "featureNames": cols,
        "featureInfo": {
            c: {
                "displayName": FEATURE_DISPLAY.get(c, c),
                "description": FEATURE_GROUPS.get(c, "feature"),
                "group": FEATURE_GROUPS.get(c, "feature"),
                "requiresInt": c == "n_elements",
            }
            for c in cols
        },
        "channels": exported_channels,
        "validation": validation,
    }
    DATA_OUT.write_text(json.dumps(metadata, indent=2))

    print("=" * 80)
    print("WEBSHAP SORTING ASSET EXPORT COMPLETE")
    print("=" * 80)
    print(f"Models: {MODEL_OUT}")
    print(f"Data:   {DATA_OUT}")
    for channel, result in validation.items():
        print(f"{channel:9s} max_abs_probability_diff={result['max_abs_probability_diff']:.8f}")


if __name__ == "__main__":
    main()
