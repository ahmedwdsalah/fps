#!/usr/bin/env python3
"""
Step 2: Build Training Dataset
================================
For each array in the real-world dataset:
  1. Load array from CSV
  2. Extract 16 structural features
  3. Time 3 sorting algorithms (introsort, heapsort, timsort)
  4. Record the winner (fastest algorithm)

Output: data/training_dataset.csv by default
  Columns: domain, file, 16 features, time_introsort,
           time_heapsort, time_timsort, best_algorithm

Crash-safe & resumable: writes every FLUSH_EVERY rows and skips
already-processed arrays on restart.

Usage:
    python3 scripts/build_training_dataset.py                # full run
    python3 scripts/build_training_dataset.py --limit 1000   # test on 1K
    python3 scripts/build_training_dataset.py --dry-run      # just show plan
    python3 scripts/build_training_dataset.py --index /path/to/index.csv \
        --raw-dir /path/to/raw --output /path/to/out.csv
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import h5py
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


def build_training_dataset(
    index_csv: Path = INDEX_CSV,
    raw_dir: Path = RAW_DIR,
    output_csv: Path = OUTPUT_CSV,
    packed_h5: Path | None = None,
    limit: int = 0,
    dry_run: bool = False,
) -> dict:
    """Build a training dataset from an index + raw-array directory."""
    index_csv = Path(index_csv)
    raw_dir = Path(raw_dir)
    output_csv = Path(output_csv)

    # Resolve optional packed backend path.
    if packed_h5 is None:
        candidate1 = index_csv.parent / "raw_arrays.h5"
        candidate2 = raw_dir.parent / "raw_arrays.h5"
        if candidate1.exists():
            packed_h5 = candidate1
        elif candidate2.exists():
            packed_h5 = candidate2
    packed_h5 = Path(packed_h5) if packed_h5 else None

    # Load index
    print("Loading index...")
    idx = pd.read_csv(index_csv, low_memory=False)
    total = len(idx)

    # Filter by size
    idx = idx[(idx["n_elements"] >= MIN_ARRAY_SIZE) &
              (idx["n_elements"] <= MAX_ARRAY_SIZE)]
    filtered = len(idx)
    print(f"  Index: {index_csv}")
    print(f"  Raw dir: {raw_dir}")
    if packed_h5 is not None and packed_h5.exists():
        print(f"  Packed raw: {packed_h5}")
    print(f"  Output: {output_csv}")
    print(f"  Total in index: {total:,}")
    print(f"  After size filter ({MIN_ARRAY_SIZE}-{MAX_ARRAY_SIZE:,}): {filtered:,}")

    if limit > 0:
        idx = idx.head(limit)
        print(f"  Limited to: {len(idx):,}")

    if idx.empty:
        print("  No eligible arrays found.")
        return {
            "processed": 0,
            "errors": 0,
            "rows": 0,
            "output": str(output_csv),
        }

    # Find n_max for length_norm feature
    n_max = float(idx["n_elements"].max())

    # Resume: load already-processed array IDs
    done_files = set()
    if output_csv.exists() and output_csv.stat().st_size > 0:
        existing = pd.read_csv(output_csv)
        if "file" in existing.columns:
            done_files = set(existing["file"].tolist())
        print(f"  Already processed: {len(done_files):,} (will skip)")

    todo = idx[~idx["file"].isin(done_files)]
    print(f"  To process: {len(todo):,}")

    if dry_run:
        print("\nDRY RUN — showing domain breakdown:")
        for prefix, label in DOMAIN_PREFIXES:
            c = len(todo[todo["file"].astype(str).str.startswith(prefix)])
            if c > 0:
                print(f"    {label:12s} {c:>10,}")
        unknown = len(todo) - sum(
            len(todo[todo["file"].astype(str).str.startswith(prefix)])
            for prefix, _ in DOMAIN_PREFIXES
        )
        if unknown > 0:
            print(f"    {'Unknown':12s} {unknown:>10,}")
        return {
            "processed": 0,
            "errors": 0,
            "rows": len(todo),
            "output": str(output_csv),
        }

    # Optional HDF5 packed loader: file -> (start, length)
    h5 = None
    h5_values = None
    packed_lookup: dict[str, tuple[int, int]] = {}
    if packed_h5 is not None and packed_h5.exists():
        h5 = h5py.File(packed_h5, "r")
        if not all(k in h5 for k in ("values", "start", "length", "file")):
            h5.close()
            raise ValueError(f"Packed HDF5 missing required datasets: {packed_h5}")
        h5_values = h5["values"]
        starts = h5["start"][:]
        lengths = h5["length"][:]
        files = h5["file"][:]
        for i in range(len(files)):
            fname = files[i]
            if isinstance(fname, bytes):
                fname = fname.decode("utf-8")
            packed_lookup[str(fname)] = (int(starts[i]), int(lengths[i]))

    # Output columns
    columns = ["file", "domain", "n_elements"] + FEATURE_NAMES + [
        "time_introsort", "time_heapsort", "time_timsort", "best_algorithm"
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Write header if file doesn't exist
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        pd.DataFrame(columns=columns).to_csv(output_csv, index=False)

    # Process
    buffer = []
    processed = 0
    errors = 0
    t_start = time.time()

    for i, (_, row) in enumerate(todo.iterrows()):
        filename = str(row["file"])
        fpath = raw_dir / filename

        arr = None
        if fpath.exists():
            try:
                arr = np.loadtxt(fpath)
            except Exception:
                arr = None

        if arr is None and packed_lookup:
            loc = packed_lookup.get(filename)
            if loc is not None and h5_values is not None:
                start, length = loc
                arr = np.array(h5_values[start : start + length], dtype=np.float64)

        if arr is None:
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
            pd.DataFrame(buffer).to_csv(output_csv, mode="a", header=False, index=False)
            buffer.clear()

        # Progress
        if processed % 2000 == 0:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = len(todo) - (i + 1)
            eta = remaining / rate if rate > 0 else 0
            print(f"  [{processed:>9,}/{len(todo):,}]  "
                  f"rate={rate:.0f}/s  "
                  f"errors={errors}  "
                  f"ETA {eta/60:.1f} min")

    # Final flush
    if buffer:
        pd.DataFrame(buffer).to_csv(output_csv, mode="a", header=False, index=False)

    elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING DATASET COMPLETE")
    print(f"{'='*60}")

    if h5 is not None:
        h5.close()
    print(f"  Processed: {processed:,}")
    print(f"  Errors:    {errors:,}")
    print(f"  Time:      {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output:    {output_csv}")

    # Quick stats
    df = pd.read_csv(output_csv)
    print(f"\n  Rows: {len(df):,}")
    if not df.empty:
        print(f"\n  Algorithm winners:")
        for algo, count in df["best_algorithm"].value_counts().items():
            pct = 100 * count / len(df)
            print(f"    {algo:12s} {count:>8,}  ({pct:.1f}%)")

        print(f"\n  Domain breakdown:")
        for domain, count in df["domain"].value_counts().items():
            print(f"    {domain:12s} {count:>8,}")
    print(f"{'='*60}")

    return {
        "processed": processed,
        "errors": errors,
        "rows": len(df),
        "output": str(output_csv),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build training dataset")
    parser.add_argument("--index", type=Path, default=INDEX_CSV,
                        help=f"Metadata index CSV (default: {INDEX_CSV})")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help=f"Directory containing raw array CSV files (default: {RAW_DIR})")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV,
                        help=f"Output training CSV (default: {OUTPUT_CSV})")
    parser.add_argument("--packed-h5", type=Path, default=None,
                        help="Optional packed HDF5 raw array file (raw_arrays.h5).")
    parser.add_argument("--limit",   type=int, default=0,
                        help="Process only first N arrays (0=all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan, don't process")
    args = parser.parse_args()
    build_training_dataset(
        index_csv=args.index,
        raw_dir=args.raw_dir,
        output_csv=args.output,
        packed_h5=args.packed_h5,
        limit=args.limit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
