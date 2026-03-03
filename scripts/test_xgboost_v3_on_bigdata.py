#!/usr/bin/env python3
"""
Evaluate trained XGBoost v3 (log+pairwise) on downloaded .npy telemetry arrays.

Workflow per file:
1) Load array
2) Extract structural features
3) Benchmark introsort/heapsort/timsort timings
4) Build v3 feature vector (structural + log times + pairwise diffs)
5) Predict best algorithm and compare to measured best

Defaults are intentionally "heavier" to reduce timer noise:
- Skip very small arrays
- Use many timing repeats
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from feature_extraction import FEATURE_NAMES, extract_features


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "real_world_bigtest" / "raw"
DEFAULT_RESULTS_CSV = ROOT / "data" / "real_world_bigtest" / "results_xgboost_v3.csv"
DEFAULT_MODEL_PATH = ROOT / "models" / "xgboost_v3_logpairwise" / "xgb_classifier_v3_logpairwise.json"
DEFAULT_BENCHMARK_CONFIG = ROOT / "data" / "benchmark" / "benchmark_config.json"

ALGORITHMS = ["introsort", "heapsort", "timsort"]
EXTRA_FEATURES = [
    "log_time_introsort",
    "log_time_heapsort",
    "log_time_timsort",
    "diff_introsort_heapsort",
    "diff_introsort_timsort",
    "diff_heapsort_timsort",
]
ALL_FEATURES = FEATURE_NAMES + EXTRA_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test XGBoost v3 on downloaded .npy data.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory with .npy files.")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV, help="Output CSV path.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Trained model JSON path.")
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=DEFAULT_BENCHMARK_CONFIG,
        help="benchmark_config.json used to recover n_max.",
    )
    parser.add_argument(
        "--n-max",
        type=float,
        default=None,
        help="Override n_max for feature extraction. If omitted, read from benchmark config.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of files to process.")
    parser.add_argument(
        "--min-elements",
        type=int,
        default=10_000,
        help="Skip arrays smaller than this size (default: 10,000).",
    )
    parser.add_argument(
        "--repeats-small",
        type=int,
        default=51,
        help="Timing repeats for arrays <= large-threshold (default: 51).",
    )
    parser.add_argument(
        "--repeats-large",
        type=int,
        default=21,
        help="Timing repeats for arrays > large-threshold (default: 21).",
    )
    parser.add_argument(
        "--large-threshold",
        type=int,
        default=500_000,
        help="Array size threshold separating small/large repeat counts.",
    )
    return parser.parse_args()


def load_n_max(config_path: Path, override: float | None) -> float:
    if override is not None:
        return float(override)
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        if "n_max" in cfg:
            return float(cfg["n_max"])
    return 2_000_000.0


def sort_introsort(arr: np.ndarray) -> np.ndarray:
    return np.sort(arr, kind="quicksort")


def sort_heapsort(arr: np.ndarray) -> np.ndarray:
    return np.sort(arr, kind="heapsort")


def sort_timsort(arr: np.ndarray) -> np.ndarray:
    return np.sort(arr, kind="stable")


SORTERS = {
    "introsort": sort_introsort,
    "heapsort": sort_heapsort,
    "timsort": sort_timsort,
}


def time_algorithm(func, arr: np.ndarray, repeats: int) -> float:
    _ = func(arr.copy())  # warmup
    times: list[float] = []
    for _ in range(repeats):
        gc.disable()
        arr_copy = arr.copy()
        start = time.perf_counter()
        _ = func(arr_copy)
        elapsed = time.perf_counter() - start
        gc.enable()
        times.append(elapsed)
    return float(np.median(times))


def benchmark_three_algorithms(
    arr: np.ndarray,
    repeats_small: int,
    repeats_large: int,
    large_threshold: int,
) -> tuple[dict[str, float], int]:
    repeats = repeats_small if arr.size <= large_threshold else repeats_large
    timings: dict[str, float] = {}
    for name, func in SORTERS.items():
        timings[name] = time_algorithm(func, arr, repeats=repeats)
    return timings, repeats


def build_feature_row(
    arr: np.ndarray,
    sample_id: str,
    n_max: float,
    repeats_small: int,
    repeats_large: int,
    large_threshold: int,
) -> tuple[dict[str, float], dict[str, float], int]:
    feats = extract_features(arr, n_max=n_max, sample_id=sample_id)
    timings, repeats_used = benchmark_three_algorithms(
        arr,
        repeats_small=repeats_small,
        repeats_large=repeats_large,
        large_threshold=large_threshold,
    )

    feats["log_time_introsort"] = float(np.log1p(timings["introsort"]))
    feats["log_time_heapsort"] = float(np.log1p(timings["heapsort"]))
    feats["log_time_timsort"] = float(np.log1p(timings["timsort"]))
    feats["diff_introsort_heapsort"] = feats["log_time_introsort"] - feats["log_time_heapsort"]
    feats["diff_introsort_timsort"] = feats["log_time_introsort"] - feats["log_time_timsort"]
    feats["diff_heapsort_timsort"] = feats["log_time_heapsort"] - feats["log_time_timsort"]

    return feats, timings, repeats_used


def main() -> None:
    args = parse_args()

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw dir not found: {args.raw_dir}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    n_max = load_n_max(args.benchmark_config, args.n_max)

    model = xgb.XGBClassifier()
    model.load_model(str(args.model_path))
    label_encoder = LabelEncoder().fit(ALGORITHMS)

    npy_files = sorted(p for p in args.raw_dir.iterdir() if p.suffix == ".npy")
    if args.limit is not None:
        npy_files = npy_files[: args.limit]

    results: list[dict] = []
    failed = 0
    skipped_too_small = 0

    for file_path in tqdm(npy_files, desc="Evaluating"):
        try:
            arr = np.load(file_path)
            arr = np.asarray(arr).reshape(-1)
            if arr.size == 0:
                failed += 1
                continue
            if arr.size < args.min_elements:
                skipped_too_small += 1
                continue
            if not np.issubdtype(arr.dtype, np.number):
                failed += 1
                continue
            arr = arr.astype(np.float64, copy=False)

            feats, timings, repeats_used = build_feature_row(
                arr,
                sample_id=file_path.stem,
                n_max=n_max,
                repeats_small=args.repeats_small,
                repeats_large=args.repeats_large,
                large_threshold=args.large_threshold,
            )
            x_row = np.array([[feats[name] for name in ALL_FEATURES]], dtype=np.float64)

            y_pred_enc = model.predict(x_row)[0]
            prediction = str(label_encoder.inverse_transform([int(y_pred_enc)])[0])
            best_actual = min(timings, key=timings.get)

            results.append(
                {
                    "file": file_path.name,
                    "n_elements": int(arr.size),
                    "prediction": prediction,
                    "best_actual": best_actual,
                    "match": bool(prediction == best_actual),
                    "timing_repeats": int(repeats_used),
                    "time_introsort": float(timings["introsort"]),
                    "time_heapsort": float(timings["heapsort"]),
                    "time_timsort": float(timings["timsort"]),
                    "predicted_time": float(timings[prediction]),
                    "best_time": float(timings[best_actual]),
                    "regret_ratio_pct": float(
                        (timings[prediction] - timings[best_actual]) / max(timings[best_actual], 1e-12) * 100.0
                    ),
                }
            )
        except Exception:
            failed += 1
            continue

    out_df = pd.DataFrame(results)
    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.results_csv, index=False)

    total = len(npy_files)
    processed = len(out_df)
    eligible = total - skipped_too_small
    acc = float(out_df["match"].mean()) if processed > 0 else 0.0
    mean_regret = float(out_df["regret_ratio_pct"].mean()) if processed > 0 else 0.0

    print(f"Model: {args.model_path}")
    print(f"Input dir: {args.raw_dir}")
    print(f"n_max: {n_max}")
    print(
        f"Settings: min_elements={args.min_elements}, "
        f"repeats_small={args.repeats_small}, repeats_large={args.repeats_large}, "
        f"large_threshold={args.large_threshold}"
    )
    print(
        f"Processed: {processed}/{total} files "
        f"(eligible: {eligible}, skipped_small: {skipped_too_small}, failed: {failed})"
    )
    print(f"Accuracy vs measured best: {acc * 100:.2f}%")
    print(f"Mean regret: {mean_regret:.2f}%")
    print(f"Results saved to: {args.results_csv}")


if __name__ == "__main__":
    main()
