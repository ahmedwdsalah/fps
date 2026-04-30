#!/usr/bin/env python3
"""
Relabel F1 dataset with exact algorithm implementations (v2).

Default algorithms:
  - insertion_sort
  - quick_sort
  - merge_sort
  - tim_sort
  - heap_sort
  - shell_sort
  - comb_sort
  - parallel_merge_sort

Reads an existing training dataset CSV (features already computed), re-times
raw arrays with selected algorithms, and writes a new CSV with:
  - time_<algorithm_name> for each selected algorithm
  - best_algorithm_v2
  - winner_margin_v2

It is crash-safe and resumable by filename.
"""

from __future__ import annotations

import argparse
import difflib
import gc
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

INPUT_CSV = Path("/Volumes/k/thesis_data/f1_only_1m_packed/training_dataset.csv")
RAW_DIR = Path("/Volumes/k/thesis_data/f1_only_1m_packed/raw")
OUTPUT_CSV = Path("/Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2.csv")
PACKED_H5 = Path("/Volumes/k/thesis_data/f1_only_1m_packed/raw_arrays.h5")

DEFAULT_ALGORITHMS = [
    "insertion_sort",
    "quick_sort",
    "introsort",
    "merge_sort",
    "tim_sort",
    "heap_sort",
    "shell_sort",
    "comb_sort",
    "parallel_merge_sort",
]

ALGO_PROFILES = {
    # Fixed profile for your experiment using exact named algorithms.
    # Intentionally excludes tim_sort because it dominates this pure-Python set.
    "actual_5": [
        "quick_sort",
        "introsort",
        "merge_sort",
        "heap_sort",
        "shell_sort",
    ],
    "default": DEFAULT_ALGORITHMS,
}

ALGO_MAX_N_DEFAULTS = {
    "insertion_sort": 1024,
    "quick_sort": None,
    "introsort": None,
    "merge_sort": None,
    "tim_sort": None,
    "heap_sort": None,
    "shell_sort": None,
    "comb_sort": 4096,
    "parallel_merge_sort": None,
}

_WORKER_H5_FILE = None
_WORKER_H5_VALUES = None
_WORKER_H5_START = None
_WORKER_H5_LENGTH = None
_WORKER_H5_FILE_TO_ROW: dict[str, int] | None = None


def _worker_init(packed_h5_path: str) -> None:
    global _WORKER_H5_FILE, _WORKER_H5_VALUES, _WORKER_H5_START, _WORKER_H5_LENGTH, _WORKER_H5_FILE_TO_ROW
    _WORKER_H5_FILE_TO_ROW = None
    if not packed_h5_path:
        return

    h5_path = Path(packed_h5_path)
    if not h5_path.exists():
        return

    h5 = h5py.File(h5_path, "r")
    _WORKER_H5_FILE = h5
    _WORKER_H5_VALUES = h5["values"]
    _WORKER_H5_START = h5["start"]
    _WORKER_H5_LENGTH = h5["length"]

    names = h5["file"][:]
    _WORKER_H5_FILE_TO_ROW = {}
    for i, name in enumerate(names):
        key = name.decode("utf-8") if isinstance(name, bytes) else str(name)
        _WORKER_H5_FILE_TO_ROW[key] = i


def insertion_sort(data: list[float]) -> list[float]:
    arr = data[:]
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def quick_sort(data: list[float]) -> list[float]:
    arr = data[:]
    if len(arr) < 2:
        return arr

    stack: list[tuple[int, int]] = [(0, len(arr) - 1)]

    while stack:
        low, high = stack.pop()
        if low >= high:
            continue

        mid = (low + high) // 2
        pivot_candidates = [(arr[low], low), (arr[mid], mid), (arr[high], high)]
        pivot_candidates.sort(key=lambda x: x[0])
        pivot_index = pivot_candidates[1][1]
        arr[pivot_index], arr[high] = arr[high], arr[pivot_index]
        pivot = arr[high]

        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        i += 1
        arr[i], arr[high] = arr[high], arr[i]

        left_size = i - 1 - low
        right_size = high - (i + 1)
        if left_size > right_size:
            if low < i - 1:
                stack.append((low, i - 1))
            if i + 1 < high:
                stack.append((i + 1, high))
        else:
            if i + 1 < high:
                stack.append((i + 1, high))
            if low < i - 1:
                stack.append((low, i - 1))
    return arr


def introsort(data: list[float]) -> list[float]:
    arr = data[:]
    n = len(arr)
    if n < 2:
        return arr

    max_depth = int(math.log2(n)) * 2 if n > 0 else 0

    def insertion_sort_range(lo: int, hi: int) -> None:
        for i in range(lo + 1, hi + 1):
            key = arr[i]
            j = i - 1
            while j >= lo and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    def heap_sort_range(lo: int, hi: int) -> None:
        length = hi - lo + 1
        if length <= 1:
            return

        def sift_down(root: int, end: int) -> None:
            while True:
                child = 2 * root + 1
                if child > end:
                    break
                if child + 1 <= end and arr[lo + child] < arr[lo + child + 1]:
                    child += 1
                if arr[lo + root] < arr[lo + child]:
                    arr[lo + root], arr[lo + child] = arr[lo + child], arr[lo + root]
                    root = child
                else:
                    break

        for start in range((length - 2) // 2, -1, -1):
            sift_down(start, length - 1)
        for end in range(length - 1, 0, -1):
            arr[lo], arr[lo + end] = arr[lo + end], arr[lo]
            sift_down(0, end - 1)

    stack: list[tuple[int, int, int]] = [(0, n - 1, max_depth)]
    while stack:
        lo, hi, depth = stack.pop()
        size = hi - lo + 1
        if size <= 1:
            continue
        if size <= 16:
            insertion_sort_range(lo, hi)
            continue
        if depth == 0:
            heap_sort_range(lo, hi)
            continue

        mid = lo + ((hi - lo) // 2)
        trio = [(arr[lo], lo), (arr[mid], mid), (arr[hi], hi)]
        trio.sort(key=lambda x: x[0])
        pivot_idx = trio[1][1]
        arr[pivot_idx], arr[hi] = arr[hi], arr[pivot_idx]
        pivot = arr[hi]

        i = lo - 1
        for j in range(lo, hi):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        i += 1
        arr[i], arr[hi] = arr[hi], arr[i]

        left = (lo, i - 1, depth - 1)
        right = (i + 1, hi, depth - 1)
        if (left[1] - left[0]) > (right[1] - right[0]):
            if left[0] < left[1]:
                stack.append(left)
            if right[0] < right[1]:
                stack.append(right)
        else:
            if right[0] < right[1]:
                stack.append(right)
            if left[0] < left[1]:
                stack.append(left)
    return arr


def merge_sort(data: list[float]) -> list[float]:
    n = len(data)
    if n < 2:
        return data[:]

    arr = data[:]
    buf = [0.0] * n
    width = 1

    while width < n:
        for left in range(0, n, 2 * width):
            mid = min(left + width, n)
            right = min(left + 2 * width, n)

            i, j, k = left, mid, left
            while i < mid and j < right:
                if arr[i] <= arr[j]:
                    buf[k] = arr[i]
                    i += 1
                else:
                    buf[k] = arr[j]
                    j += 1
                k += 1
            while i < mid:
                buf[k] = arr[i]
                i += 1
                k += 1
            while j < right:
                buf[k] = arr[j]
                j += 1
                k += 1

            arr[left:right] = buf[left:right]
        width *= 2

    return arr


def tim_sort(data: list[float]) -> list[float]:
    # Python built-in sort uses Timsort.
    return sorted(data)


def heap_sort(data: list[float]) -> list[float]:
    # Pure Python heap sort (max-heap) for fairer comparison against
    # other Python-implemented algorithms in this script.
    arr = data[:]
    n = len(arr)

    def sift_down(start: int, end: int) -> None:
        root = start
        while True:
            child = 2 * root + 1
            if child > end:
                break
            if child + 1 <= end and arr[child] < arr[child + 1]:
                child += 1
            if arr[root] < arr[child]:
                arr[root], arr[child] = arr[child], arr[root]
                root = child
            else:
                break

    # Build max-heap
    for start in range((n - 2) // 2, -1, -1):
        sift_down(start, n - 1)

    # Heap sort
    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]
        sift_down(0, end - 1)

    return arr


def shell_sort(data: list[float]) -> list[float]:
    arr = data[:]
    n = len(arr)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr


def comb_sort(data: list[float]) -> list[float]:
    arr = data[:]
    n = len(arr)
    gap = n
    shrink = 1.3
    sorted_flag = False
    while not sorted_flag:
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted_flag = True
        i = 0
        while i + gap < n:
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                sorted_flag = False
            i += 1
    return arr


def _merge_sorted_lists(left: list[float], right: list[float]) -> list[float]:
    out = [0.0] * (len(left) + len(right))
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out[k] = left[i]
            i += 1
        else:
            out[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        out[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        out[k] = right[j]
        j += 1
        k += 1
    return out


def parallel_merge_sort(
    data: list[float],
    parallel_threshold: int,
) -> list[float]:
    n = len(data)
    if n < 2:
        return data[:]

    # For small arrays, normal merge sort is faster.
    if n < parallel_threshold:
        return merge_sort(data)

    # Chunk-based merge plan. The row-level pipeline itself is multi-process.
    # This keeps algorithm semantics explicit without nested process pools.
    chunk_count = 4
    chunk_size = (n + chunk_count - 1) // chunk_count
    chunks: list[list[float]] = []
    for i in range(0, n, chunk_size):
        chunks.append(merge_sort(data[i:i + chunk_size]))

    while len(chunks) > 1:
        merged: list[list[float]] = []
        for i in range(0, len(chunks), 2):
            if i + 1 < len(chunks):
                merged.append(_merge_sorted_lists(chunks[i], chunks[i + 1]))
            else:
                merged.append(chunks[i])
        chunks = merged
    return chunks[0]


def time_algorithm(
    arr: np.ndarray,
    algo: str,
    repeats: int,
    parallel_threshold: int,
) -> float:
    # Median-of-N to reduce timer noise.
    timings: list[float] = []
    for _ in range(repeats):
        data = arr.tolist()
        gc.disable()
        t0 = time.perf_counter()
        if algo == "insertion_sort":
            insertion_sort(data)
        elif algo == "quick_sort":
            quick_sort(data)
        elif algo == "introsort":
            introsort(data)
        elif algo == "merge_sort":
            merge_sort(data)
        elif algo == "tim_sort":
            tim_sort(data)
        elif algo == "heap_sort":
            heap_sort(data)
        elif algo == "shell_sort":
            shell_sort(data)
        elif algo == "comb_sort":
            comb_sort(data)
        elif algo == "parallel_merge_sort":
            parallel_merge_sort(data, parallel_threshold=parallel_threshold)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
        t1 = time.perf_counter()
        gc.enable()
        timings.append(t1 - t0)
    return float(np.median(np.array(timings, dtype=np.float64)))


def format_bar(pct: float, width: int = 40) -> str:
    filled = int(round((pct / 100.0) * width))
    filled = max(0, min(width, filled))
    return "#" * filled + "-" * (width - filled)


def print_distribution(counter: Counter, total: int, algorithms: list[str]) -> None:
    print("\n  Winner distribution:")
    for algo in algorithms:
        count = counter.get(algo, 0)
        pct = (100.0 * count / total) if total else 0.0
        print(f"    {algo:20s} {count:>8,}  ({pct:>5.1f}%)  [{format_bar(pct)}]")


def export_competitive_sets(
    out_df: pd.DataFrame,
    output_csv: Path,
    algorithms: list[str],
    max_margin: float | None,
    write_balanced: bool,
    balance_seed: int,
) -> Path | None:
    if max_margin is None:
        return None

    if "winner_margin_v2" not in out_df.columns or "best_algorithm_v2" not in out_df.columns:
        print("\n  Competitive export skipped (required columns missing).")
        return None

    hard_df = out_df[out_df["winner_margin_v2"] <= max_margin].copy()
    hard_path = output_csv.with_name(
        f"{output_csv.stem}_hard_m{str(max_margin).replace('.', 'p')}{output_csv.suffix}"
    )
    hard_df.to_csv(hard_path, index=False)

    print(f"\n  Hard-case export (winner_margin_v2 <= {max_margin:.3f}):")
    print(f"    Rows:   {len(hard_df):,}")
    print(f"    Path:   {hard_path}")
    if len(hard_df) > 0:
        vc = hard_df["best_algorithm_v2"].value_counts()
        for algo, cnt in vc.items():
            pct = 100.0 * cnt / len(hard_df)
            print(f"    {algo:20s} {cnt:>8,}  ({pct:>5.1f}%)  [{format_bar(pct)}]")

    if not write_balanced:
        return None

    if hard_df.empty:
        print("  Balanced hard-case export skipped (no hard-case rows).")
        return None

    vc = hard_df["best_algorithm_v2"].value_counts()
    if vc.empty or vc.min() <= 0:
        print("  Balanced hard-case export skipped (invalid class counts).")
        return None

    present = set(vc.index.tolist())
    expected = set(algorithms)
    missing = expected - present
    if missing:
        print(
            "  Balanced hard-case export skipped (missing winner classes)."
        )
        print(f"    Present winners: {sorted(present)}")
        print(f"    Missing winners: {sorted(missing)}")
        print(
            "    Reason: balancing cannot invent winners for algorithms that "
            "never win in the hard-case subset."
        )
        return None

    min_count = int(vc.min())
    parts = []
    for cls in vc.index:
        parts.append(
            hard_df[hard_df["best_algorithm_v2"] == cls].sample(
                n=min_count, random_state=balance_seed
            )
        )
    balanced_df = (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=balance_seed)
        .reset_index(drop=True)
    )
    balanced_path = output_csv.with_name(
        f"{output_csv.stem}_hard_m{str(max_margin).replace('.', 'p')}_balanced{output_csv.suffix}"
    )
    balanced_df.to_csv(balanced_path, index=False)

    print("\n  Balanced hard-case export:")
    print(f"    Rows:   {len(balanced_df):,}")
    print(f"    Path:   {balanced_path}")
    bvc = balanced_df["best_algorithm_v2"].value_counts()
    for algo, cnt in bvc.items():
        pct = 100.0 * cnt / len(balanced_df)
        print(f"    {algo:20s} {cnt:>8,}  ({pct:>5.1f}%)  [{format_bar(pct)}]")
    return balanced_path


def parse_algorithms(raw: str) -> list[str]:
    algorithms = [item.strip() for item in raw.split(",") if item.strip()]
    if not algorithms:
        raise ValueError("Algorithm list cannot be empty.")
    unknown = [a for a in algorithms if a not in ALGO_MAX_N_DEFAULTS]
    if unknown:
        known = list(ALGO_MAX_N_DEFAULTS.keys())
        suggestions = {}
        for name in unknown:
            close = difflib.get_close_matches(name, known, n=3, cutoff=0.55)
            if close:
                suggestions[name] = close
        lines = [f"Unknown algorithms: {unknown}"]
        lines.append(f"Valid names: {known}")
        if suggestions:
            for wrong, close in suggestions.items():
                lines.append(f"Did you mean for '{wrong}': {close}?")
        raise ValueError("\n".join(lines))
    return algorithms


def _load_array_for_file(raw_dir: Path, filename: str) -> np.ndarray | None:
    fpath = raw_dir / filename
    if fpath.exists():
        try:
            return np.loadtxt(fpath, dtype=np.float64)
        except Exception:
            return None

    if _WORKER_H5_FILE_TO_ROW is None or _WORKER_H5_VALUES is None:
        return None
    row = _WORKER_H5_FILE_TO_ROW.get(filename)
    if row is None:
        return None
    start = int(_WORKER_H5_START[row])
    length = int(_WORKER_H5_LENGTH[row])
    if length <= 0:
        return None
    return np.array(_WORKER_H5_VALUES[start : start + length], dtype=np.float64, copy=False)


def _process_one_row(
    payload: tuple[
        dict,
        str,
        int,
        int,
        dict[str, int | None],
        list[str],
        int,
    ]
) -> tuple[dict | None, str | None]:
    (
        row_dict,
        raw_dir_str,
        repeats_small,
        repeats_large,
        algo_max_n,
        algorithms,
        parallel_threshold,
    ) = payload
    raw_dir = Path(raw_dir_str)
    filename = str(row_dict["file"])
    arr = _load_array_for_file(raw_dir=raw_dir, filename=filename)
    if arr is None:
        return None, filename

    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 2:
        return None, filename

    repeats = repeats_small if n <= 10_000 else repeats_large
    times: dict[str, float] = {}
    for algo in algorithms:
        max_n = algo_max_n.get(algo)
        if max_n is not None and n > max_n:
            times[algo] = math.inf
            continue
        times[algo] = time_algorithm(
            arr=arr,
            algo=algo,
            repeats=repeats,
            parallel_threshold=parallel_threshold,
        )

    ranked = sorted(times.items(), key=lambda kv: kv[1])
    winner, winner_time = ranked[0]
    second_time = ranked[1][1]
    margin = 0.0 if not math.isfinite(second_time) else (second_time - winner_time) / (second_time + 1e-15)

    out_row = dict(row_dict)
    for algo in algorithms:
        out_row[f"time_{algo}"] = times[algo]
    out_row["best_algorithm_v2"] = winner
    out_row["winner_margin_v2"] = float(margin)
    return out_row, None


def relabel(
    input_csv: Path,
    raw_dir: Path,
    packed_h5: Path | None,
    output_csv: Path,
    repeats_small: int,
    repeats_large: int,
    algorithms: list[str],
    algo_max_n: dict[str, int | None],
    parallel_threshold: int,
    workers: int,
    flush_every: int,
    chunksize: int,
    overwrite_output: bool,
    max_margin: float | None,
    write_balanced: bool,
    require_balanced: bool,
    balance_seed: int,
    limit: int,
    dry_run: bool,
) -> None:
    df = pd.read_csv(input_csv)
    total_input = len(df)
    if limit > 0:
        df = df.head(limit)

    print("=" * 70)
    print("  RELABEL F1 DATASET WITH EXACT SORTING ALGORITHMS (V2)")
    print("=" * 70)
    print(f"  Input:   {input_csv}")
    print(f"  Raw dir: {raw_dir}")
    print(f"  Packed:  {packed_h5 if packed_h5 else 'disabled'}")
    print(f"  Output:  {output_csv}")
    print(f"  Rows in input: {total_input:,}")
    print(f"  Rows selected: {len(df):,}")
    print(f"  Workers: {workers}")
    print(f"  Algorithms: {', '.join(algorithms)}")

    if dry_run:
        print("\nDRY RUN complete.")
        return

    files = df["file"].astype(str) if "file" in df.columns else pd.Series(dtype=str)
    needs_packed = files.str.endswith(".h5arr").any()
    if needs_packed and not (packed_h5 and packed_h5.exists()):
        raise FileNotFoundError(
            f"Packed HDF5 is required for .h5arr rows but was not found: {packed_h5}"
        )

    base_columns = list(df.columns)
    extra_columns = [f"time_{algo}" for algo in algorithms] + [
        "best_algorithm_v2",
        "winner_margin_v2",
    ]
    output_columns = base_columns + [c for c in extra_columns if c not in base_columns]

    done = set()
    if overwrite_output and output_csv.exists():
        output_csv.unlink()
        print("  Overwrite enabled: existing output removed.")

    if output_csv.exists() and output_csv.stat().st_size > 0:
        existing_cols = pd.read_csv(output_csv, nrows=1).columns.tolist()
        missing = [c for c in output_columns if c not in existing_cols]
        if missing:
            raise ValueError(
                "Existing output file is missing required columns for current "
                f"algorithm set: {missing}. Use a new --output path."
            )
        old = pd.read_csv(output_csv, usecols=["file"])
        done = set(old["file"].astype(str).tolist())
        print(f"  Resume detected, already done: {len(done):,}")

    todo = df[~df["file"].astype(str).isin(done)]
    print(f"  To process now: {len(todo):,}")

    if not output_csv.exists() or output_csv.stat().st_size == 0:
        pd.DataFrame(columns=output_columns).to_csv(output_csv, index=False)

    processed = 0
    errors = 0
    t_start = time.time()
    winner_counter: Counter = Counter()
    buffer: list[dict] = []

    jobs = (
        (
            row.to_dict(),
            str(raw_dir),
            repeats_small,
            repeats_large,
            algo_max_n,
            algorithms,
            parallel_threshold,
        )
        for _, row in todo.iterrows()
    )

    packed_h5_arg = str(packed_h5) if packed_h5 and packed_h5.exists() else ""
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(packed_h5_arg,),
    ) as pool:
        for i, (out_row, failed_file) in enumerate(
            pool.map(_process_one_row, jobs, chunksize=chunksize),
            1,
        ):
            if failed_file is not None or out_row is None:
                errors += 1
            else:
                buffer.append(out_row)
                processed += 1
                winner_counter[out_row["best_algorithm_v2"]] += 1

            if len(buffer) >= flush_every:
                pd.DataFrame(buffer, columns=output_columns).to_csv(
                    output_csv, mode="a", header=False, index=False
                )
                buffer.clear()

            if i % 1000 == 0:
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 else 0.0
                remaining = len(todo) - i
                eta = remaining / rate if rate > 0 else 0.0
                print(
                    f"  [{i:>8,}/{len(todo):,}] rate={rate:>5.2f}/s "
                    f"ok={processed} errors={errors} ETA={eta/60:>6.1f} min"
                )

    if buffer:
        pd.DataFrame(buffer, columns=output_columns).to_csv(
            output_csv, mode="a", header=False, index=False
        )

    out = pd.read_csv(output_csv)
    if "best_algorithm_v2" in out.columns:
        vc = out["best_algorithm_v2"].value_counts()
        final_counter = Counter(vc.to_dict())
    else:
        final_counter = Counter()

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  RELABEL COMPLETE")
    print("=" * 70)
    print(f"  Processed this run: {processed:,}")
    print(f"  Errors this run:    {errors:,}")
    print(f"  Elapsed:            {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output rows total:  {len(out):,}")

    print_distribution(final_counter, len(out), algorithms)
    balanced_path = export_competitive_sets(
        out_df=out,
        output_csv=output_csv,
        algorithms=algorithms,
        max_margin=max_margin,
        write_balanced=write_balanced,
        balance_seed=balance_seed,
    )
    if require_balanced and balanced_path is None:
        raise RuntimeError(
            "Balanced dataset was not produced. "
            "Use an algorithm set/margin that yields all classes in hard cases."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=INPUT_CSV)
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--packed-h5", type=Path, default=PACKED_H5)
    p.add_argument("--output", type=Path, default=OUTPUT_CSV)
    p.add_argument(
        "--profile",
        type=str,
        choices=sorted(ALGO_PROFILES.keys()),
        default="actual_5",
        help="Predefined algorithm profile. Default is the fixed actual_5 setup.",
    )
    p.add_argument(
        "--algorithms",
        type=str,
        default=",".join(DEFAULT_ALGORITHMS),
        help="Comma-separated list of algorithms to benchmark (used only with --profile default).",
    )
    p.add_argument("--repeats-small", type=int, default=2)
    p.add_argument("--repeats-large", type=int, default=1)
    p.add_argument(
        "--max-n-overrides",
        type=str,
        default="",
        help=(
            "Comma-separated max-N overrides per algorithm, e.g. "
            "'insertion_sort:1500,comb_sort:3000'. Use 'none' for no cap."
        ),
    )
    p.add_argument("--parallel-threshold", type=int, default=8192)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--flush-every", type=int, default=500)
    p.add_argument("--chunksize", type=int, default=64)
    p.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete existing --output file and recompute all rows from scratch.",
    )
    p.add_argument(
        "--max-margin",
        type=float,
        default=0.05,
        help="If set, export hard-case subset where winner_margin_v2 <= max-margin.",
    )
    p.add_argument(
        "--write-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --max-margin is set, also export class-balanced hard-case file.",
    )
    p.add_argument(
        "--require-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail the run if balanced output cannot be generated.",
    )
    p.add_argument("--balance-seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.profile == "default":
        algorithms = parse_algorithms(args.algorithms)
    else:
        algorithms = list(ALGO_PROFILES[args.profile])
    algo_max_n = {k: ALGO_MAX_N_DEFAULTS[k] for k in algorithms}
    if args.max_n_overrides.strip():
        for chunk in args.max_n_overrides.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError(f"Invalid --max-n-overrides entry: '{chunk}'")
            name, val = chunk.split(":", 1)
            name = name.strip()
            val = val.strip().lower()
            if name not in algo_max_n:
                raise ValueError(
                    f"Override for unknown/not-selected algorithm '{name}'."
                )
            if val in ("none", "inf", "infinity"):
                algo_max_n[name] = None
            else:
                parsed = int(val)
                if parsed < 2:
                    raise ValueError(f"Invalid max-N for {name}: {parsed}")
                algo_max_n[name] = parsed

    relabel(
        input_csv=args.input,
        raw_dir=args.raw_dir,
        packed_h5=args.packed_h5,
        output_csv=args.output,
        repeats_small=args.repeats_small,
        repeats_large=args.repeats_large,
        algorithms=algorithms,
        algo_max_n=algo_max_n,
        parallel_threshold=args.parallel_threshold,
        workers=args.workers,
        flush_every=args.flush_every,
        chunksize=args.chunksize,
        overwrite_output=args.overwrite_output,
        max_margin=args.max_margin,
        write_balanced=args.write_balanced,
        require_balanced=args.require_balanced,
        balance_seed=args.balance_seed,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
