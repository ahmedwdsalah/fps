#!/usr/bin/env python3
"""
Export a compact F1 + XGBoost explainer asset for the cnn-explainer frontend.

The asset is precomputed from:
- the existing general v5 XGBoost model
- the independent F1-only training dataset
- real F1 raw arrays

Output:
    /Users/ahmed/Desktop/cnn-explainer/public/assets/data/f1_xgb_explainer.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

MODEL_PATH = ROOT / "models" / "xgboost_v5" / "xgb_v5.json"
EVAL_RESULTS_PATH = ROOT / "results" / "xgboost_v5" / "evaluation_results.json"
F1_DATASET_PATH = Path("/Volumes/k/thesis_data/f1_only/training_dataset.csv")
F1_RAW_DIR = Path("/Volumes/k/thesis_data/f1_only/raw")
OUTPUT_PATH = Path("/Users/ahmed/Desktop/cnn-explainer/public/assets/data/f1_xgb_explainer.json")

DISPLAY_CLASSES = ["introsort", "heapsort", "timsort"]
LE = LabelEncoder().fit(["introsort", "heapsort", "timsort"])
MODEL_CLASSES = list(LE.classes_)
DISPLAY_INDEX = [MODEL_CLASSES.index(name) for name in DISPLAY_CLASSES]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES


FEATURE_META = {
    "length_norm": {
        "label": "Length",
        "group": "Scale",
        "description": "Array length normalized by the 100k cap.",
    },
    "adj_sorted_ratio": {
        "label": "Adjacent Sorted",
        "group": "Sortedness",
        "description": "Fraction of adjacent pairs already in ascending order.",
    },
    "duplicate_ratio": {
        "label": "Duplicates",
        "group": "Uniqueness",
        "description": "Share of repeated values in the array.",
    },
    "dispersion_ratio": {
        "label": "Dispersion",
        "group": "Spread",
        "description": "Standard deviation scaled by the full value range.",
    },
    "runs_ratio": {
        "label": "Runs",
        "group": "Structure",
        "description": "Monotonic run count divided by array length.",
    },
    "inversion_ratio": {
        "label": "Inversions",
        "group": "Disorder",
        "description": "Normalized inversion count capturing out-of-order pairs.",
    },
    "entropy_ratio": {
        "label": "Entropy",
        "group": "Randomness",
        "description": "32-bin histogram entropy normalized to [0, 1].",
    },
    "skewness_t": {
        "label": "Skewness",
        "group": "Shape",
        "description": "Signed log-transformed skewness of the value distribution.",
    },
    "kurtosis_excess_t": {
        "label": "Kurtosis",
        "group": "Shape",
        "description": "Signed log-transformed excess kurtosis.",
    },
    "longest_run_ratio": {
        "label": "Longest Run",
        "group": "Structure",
        "description": "Longest monotonic run divided by array length.",
    },
    "iqr_norm": {
        "label": "IQR",
        "group": "Spread",
        "description": "Interquartile range normalized by the full range.",
    },
    "mad_norm": {
        "label": "MAD",
        "group": "Spread",
        "description": "Median absolute deviation normalized by the full range.",
    },
    "top1_freq_ratio": {
        "label": "Top-1 Freq",
        "group": "Uniqueness",
        "description": "Relative frequency of the single most common value.",
    },
    "top5_freq_ratio": {
        "label": "Top-5 Freq",
        "group": "Uniqueness",
        "description": "Combined frequency of the five most common values.",
    },
    "outlier_ratio": {
        "label": "Outliers",
        "group": "Outliers",
        "description": "Share of values with |z| > 3.",
    },
    "mean_abs_diff_norm": {
        "label": "Mean |Diff|",
        "group": "Local Order",
        "description": "Average absolute adjacent difference normalized by range.",
    },
}


def parse_f1_filename(filename: str) -> dict[str, str]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 7:
        return {
            "year": "",
            "round": "",
            "session": "",
            "driver": "",
            "lap": "",
            "channel": "",
        }

    return {
        "year": parts[1],
        "round": parts[2],
        "session": parts[3],
        "driver": parts[4],
        "lap": parts[5],
        "channel": parts[6],
    }


def downsample(values: np.ndarray, target: int = 180) -> list[float]:
    if values.size == 0:
        return []
    if values.size <= target:
        sample = values.astype(np.float64)
    else:
        idx = np.linspace(0, values.size - 1, target).astype(int)
        sample = values[idx].astype(np.float64)

    vmin = float(sample.min())
    vmax = float(sample.max())
    if vmax - vmin < 1e-12:
        normed = np.full(sample.shape, 0.5, dtype=np.float64)
    else:
        normed = (sample - vmin) / (vmax - vmin)
    return [round(float(v), 6) for v in normed.tolist()]

def ordered_feature_records(
    row: pd.Series,
    contribs: np.ndarray,
    feature_ranges: dict[str, dict[str, float]],
) -> list[dict]:
    records = []
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        value = float(row[feature_name])
        per_class = {
            label: round(float(contribs[DISPLAY_INDEX[class_index], feature_index]), 6)
            for class_index, label in enumerate(DISPLAY_CLASSES)
        }
        abs_total = sum(abs(v) for v in per_class.values())
        stats = feature_ranges[feature_name]
        min_value = stats["min"]
        max_value = stats["max"]
        if max_value - min_value < 1e-12:
            normalized = 0.5
        else:
            normalized = (value - min_value) / (max_value - min_value)
        meta = FEATURE_META[feature_name]
        records.append(
            {
                "name": feature_name,
                "label": meta["label"],
                "group": meta["group"],
                "description": meta["description"],
                "value": round(value, 6),
                "normalized": round(float(np.clip(normalized, 0.0, 1.0)), 6),
                "min": round(min_value, 6),
                "max": round(max_value, 6),
                "contributions": per_class,
                "abs_contribution_total": round(abs_total, 6),
            }
        )
    records.sort(key=lambda item: item["abs_contribution_total"], reverse=True)
    return records


def make_example(
    row: pd.Series,
    probabilities: np.ndarray,
    margins: np.ndarray,
    contribs: np.ndarray,
    feature_ranges: dict[str, dict[str, float]],
) -> dict:
    filename = str(row["file"])
    parsed = parse_f1_filename(filename)
    raw_path = F1_RAW_DIR / filename
    raw_values = np.loadtxt(raw_path, dtype=np.float64)
    raw_values = raw_values[np.isfinite(raw_values)]

    ordered_probs = probabilities[DISPLAY_INDEX]
    ordered_margins = margins[DISPLAY_INDEX]
    ordered_contribs = contribs[DISPLAY_INDEX]
    bias_terms = ordered_contribs[:, -1]

    feature_records = ordered_feature_records(row, contribs, feature_ranges)
    classes = []
    for class_index, class_name in enumerate(DISPLAY_CLASSES):
        class_feature_total = float(ordered_contribs[class_index, :-1].sum())
        classes.append(
            {
                "name": class_name,
                "probability": round(float(ordered_probs[class_index]), 6),
                "margin": round(float(ordered_margins[class_index]), 6),
                "bias": round(float(bias_terms[class_index]), 6),
                "feature_sum": round(class_feature_total, 6),
            }
        )

    return {
        "id": class_name_from_example(filename, row["best_algorithm"]),
        "title": f"{str(row['best_algorithm']).capitalize()}-leaning F1 sample",
        "file": filename,
        "metadata": {
            **parsed,
            "n_elements": int(row["n_elements"]),
            "domain": str(row["domain"]),
            "ground_truth": str(row["best_algorithm"]),
            "predicted": DISPLAY_CLASSES[int(np.argmax(ordered_probs))],
            "confidence": round(float(np.max(ordered_probs)), 6),
        },
        "raw": {
            "normalized_series": downsample(raw_values),
            "min": round(float(raw_values.min()), 6) if raw_values.size else 0.0,
            "max": round(float(raw_values.max()), 6) if raw_values.size else 0.0,
            "mean": round(float(raw_values.mean()), 6) if raw_values.size else 0.0,
            "std": round(float(raw_values.std()), 6) if raw_values.size else 0.0,
        },
        "features": feature_records,
        "classes": classes,
        "timings": {
            "introsort": round(float(row["time_introsort"]), 9),
            "heapsort": round(float(row["time_heapsort"]), 9),
            "timsort": round(float(row["time_timsort"]), 9),
        },
    }


def class_name_from_example(filename: str, label: str) -> str:
    parsed = parse_f1_filename(filename)
    return f"{label}-{parsed['year']}-{parsed['round']}-{parsed['session']}-{parsed['channel']}".lower()


def main() -> None:
    df = pd.read_csv(F1_DATASET_PATH)
    feature_ranges = {
        feature_name: {
            "min": float(df[feature_name].min()),
            "max": float(df[feature_name].max()),
            "mean": float(df[feature_name].mean()),
        }
        for feature_name in FEATURE_NAMES
    }

    X = df[FEATURE_NAMES].values
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    booster = model.get_booster()

    pred_encoded = model.predict(X)
    pred_labels = LE.inverse_transform(pred_encoded)
    probabilities = model.predict_proba(X)

    scored = df.copy()
    scored["predicted"] = pred_labels
    scored["confidence"] = probabilities.max(axis=1)

    selected_indices = []
    for label in DISPLAY_CLASSES:
        subset = scored[
            (scored["best_algorithm"] == label) & (scored["predicted"] == label)
        ].sort_values("confidence", ascending=False)
        if subset.empty:
            raise RuntimeError(f"No correctly predicted F1 sample found for {label}")
        selected_indices.append(int(subset.index[0]))

    with EVAL_RESULTS_PATH.open("r", encoding="utf-8") as f:
        eval_results = json.load(f)

    selected_matrix = X[selected_indices]
    selected_dmatrix = xgb.DMatrix(selected_matrix, feature_names=FEATURE_NAMES)
    selected_margins = booster.predict(
        selected_dmatrix, output_margin=True, strict_shape=True
    )
    selected_contribs = booster.predict(
        selected_dmatrix, pred_contribs=True, strict_shape=True
    )

    examples = []
    for local_i, index in enumerate(selected_indices):
        examples.append(
            make_example(
                df.iloc[index],
                probabilities[index],
                selected_margins[local_i],
                selected_contribs[local_i],
                feature_ranges,
            )
        )

    feature_importance = eval_results.get("feature_importance", [])
    ordered_importance = [
        {
            "name": item["feature"],
            "label": FEATURE_META[item["feature"]]["label"],
            "importance": round(float(item["importance"]), 6),
        }
        for item in feature_importance
        if item["feature"] in FEATURE_META
    ]

    asset = {
        "title": "F1 Sort Explainer",
        "subtitle": "Existing XGBoost v5 model on independent F1 telemetry arrays",
        "model": {
            "name": "xgb_v5.json",
            "source": str(MODEL_PATH),
            "display_classes": DISPLAY_CLASSES,
            "model_class_order": MODEL_CLASSES,
            "dataset_accuracy": eval_results["results"]["test"]["accuracy"],
            "dataset_balanced_accuracy": eval_results["results"]["test"]["balanced_accuracy"],
        },
        "stages": [
            "F1 input array",
            "Engineered features",
            "Class scores",
            "Output probabilities",
        ],
        "feature_ranges": feature_ranges,
        "feature_importance": ordered_importance,
        "examples": examples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(asset, f, indent=2)

    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
