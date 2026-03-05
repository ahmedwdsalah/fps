#!/usr/bin/env python3
"""
Step 2: Build Training Dataset
================================
For each array in the real-world dataset:
  1. Load array from CSV
  2. Extract 16 structural features
  3. Time 3 sorting algorithms (introsort, heapsort, timsort)
  4. Record the winner (fastest algorithm)

Output: data/training_dataset.csv
  Columns: array_id, domain, file, 16 features, time_introsort,
           time_heapsort, time_timsort, best_algorithm

Crash-safe & resumable: writes every FLUSH_EVERY rows and skips
already-processed arrays on restart.

Usage:
    python3 scripts/build_training_dataset.py                # full run
    python3 scripts/build_training_dataset.py --limit 1000   # test on 1K
    python3 scripts/build_training_dataset.py --dry-run      # just show plan
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure scripts/ is on sys.path so imports work ───────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from feature_extraction import extract_features, FEATURE_NAMES

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = SCRIPT_DIR.parent
INDEX_CSV    = PROJECT_ROOT / "data" / "real_world_10k" / "index.csv"
RAW_DIR      = PROJECT_ROOT / "data" / "real_world_10k" / "raw"
OUTPUT_CSV   = PROJECT_ROOT / "data" / "training_dataset.csv"

# ── Timing config ─────────────────────────────────────────────────────────
TIMING_REPEATS_SMALL = 5   # n <= 10K
TIMING_REPEATS_LARGE = 3   # n > 10K
MAX_ARRAY_SIZE = 200_000   # skip arrays larger than this (too slow to time)
MIN_ARRAY_SIZE = 50        # skip tiny arrays

FLUSH_EVERY = 5000         # write to disk every N rows

# ── Domain detection ──────────────────────────────────────────────────────
DOMAIN_PREFIXES = [
    ("f1_",      "F1"),
    ("stock_",   "Stock"),
    ("weather_", "Weather"),
    ("crypto_",  "Crypto"),
    ("quake_",   "Earthquake"),
]

def get_domain(filename: str) -> str:
    for prefix, label in DOMAIN_PREFIXES:
        if filename.startswith(prefix):
            return label
    return "Unknown"


# ── Sorting functions ─────────────────────────────────────────────────────

def time_sort(arr: np.ndarray, kind: str, repeats: int) -> float:
    """Time a numpy sort. Returns best-of-N time in seconds."""
    best = float("inf")
    for _ in range(repeats):
        copy = arr.copy()
        gc.disable()
        t0 = time.perf_counter()
        np.sort(copy, kind=kind)
        t1 = time.perf_counter()
        gc.enable()
        best = min(best, t1 - t0)
    return best


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build training dataset")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Process only first N arrays (0=all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan, don't process")
    args = parser.parse_args()

    # Load index
    print("Loading index...")
    idx = pd.read_csv(INDEX_CSV, low_memory=False)
    total = len(idx)

    # Filter by size
    idx = idx[(idx["n_elements"] >= MIN_ARRAY_SIZE) &
              (idx["n_elements"] <= MAX_ARRAY_SIZE)]
    filtered = len(idx)
    print(f"  Total in index: {total:,}")
    print(f"  After size filter ({MIN_ARRAY_SIZE}-{MAX_ARRAY_SIZE:,}): {filtered:,}")

    if args.limit > 0:
        idx = idx.head(args.limit)
        print(f"  Limited to: {len(idx):,}")

    # Find n_max for length_norm feature
    n_max = float(idx["n_elements"].max())

    # Resume: load already-processed array IDs
    done_files = set()
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        done_files = set(existing["file"].tolist())
        print(f"  Already processed: {len(done_files):,} (will skip)")

    todo = idx[~idx["file"].isin(done_files)]
    print(f"  To process: {len(todo):,}")

    if args.dry_run:
        print("\nDRY RUN — showing domain breakdown:")
        for prefix, label in DOMAIN_PREFIXES:
            c = len(todo[todo["file"].str.startswith(prefix)])
            if c > 0:
                print(f"    {label:12s} {c:>10,}")
        return

    # Output columns
    columns = ["file", "domain", "n_elements"] + FEATURE_NAMES + [
        "time_introsort", "time_heapsort", "time_timsort", "best_algorithm"
    ]

    # Write header if file doesn't exist
    if not OUTPUT_CSV.exists():
        pd.DataFrame(columns=columns).to_csv(OUTPUT_CSV, index=False)

    # Process
    buffer = []
    processed = 0
    errors = 0
    t_start = time.time()

    for i, (_, row) in enumerate(todo.iterrows()):
        fpath = RAW_DIR / row["file"]

        if not fpath.exists():
            errors += 1
            continue

        try:
            arr = np.loadtxt(fpath)
        except Exception:
            errors += 1
            continue

        arr = arr[np.isfinite(arr)]
        n = arr.size
        if n < MIN_ARRAY_SIZE or n > MAX_ARRAY_SIZE:
            continue

        # 1. Extract features
        features = extract_features(arr, n_max, row["file"])

        # 2. Time 3 algorithms
        repeats = TIMING_REPEATS_SMALL if n <= 10_000 else TIMING_REPEATS_LARGE
        t_intro = time_sort(arr, "quicksort", repeats)
        t_heap  = time_sort(arr, "heapsort",  repeats)
        t_tim   = time_sort(arr, "stable",    repeats)

        # 3. Determine winner
        times = {"introsort": t_intro, "heapsort": t_heap, "timsort": t_tim}
        best = min(times, key=times.get)

        # Build row
        out_row = {
            "file": row["file"],
            "domain": get_domain(row["file"]),
            "n_elements": n,
        }
        out_row.update(features)
        out_row["time_introsort"] = t_intro
        out_row["time_heapsort"]  = t_heap
        out_row["time_timsort"]   = t_tim
        out_row["best_algorithm"] = best

        buffer.append(out_row)
        processed += 1

        # Flush buffer to disk periodically
        if len(buffer) >= FLUSH_EVERY:
            pd.DataFrame(buffer).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
            buffer.clear()

        # Progress
        if processed % 2000 == 0:
            elapsed = time.time() - t_start
            rate = processed / elapsed
            remaining = len(todo) - (i + 1)
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{processed:>9,}/{len(todo):,}]  "
                  f"rate={rate:.0f}/s  "
                  f"errors={errors}  "
                  f"ETA {eta/60:.1f} min")

    # Final flush
    if buffer:
        pd.DataFrame(buffer).to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING DATASET COMPLETE")
    print(f"{'='*60}")
    print(f"  Processed: {processed:,}")
    print(f"  Errors:    {errors:,}")
    print(f"  Time:      {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output:    {OUTPUT_CSV}")

    # Quick stats
    df = pd.read_csv(OUTPUT_CSV)
    print(f"\n  Rows: {len(df):,}")
    print(f"\n  Algorithm winners:")
    for algo, count in df["best_algorithm"].value_counts().items():
        pct = 100 * count / len(df)
        print(f"    {algo:12s} {count:>8,}  ({pct:.1f}%)")

    print(f"\n  Domain breakdown:")
    for domain, count in df["domain"].value_counts().items():
        print(f"    {domain:12s} {count:>8,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
