#!/usr/bin/env python3
"""
transform_f1_sample.py — Apply structural transforms to 100K sampled F1 arrays
================================================================================
Reads the existing 487K raw F1 arrays, samples 100K, applies 4 transforms
(REV, SHUF, QBIN50, PSORT10) to each → 400K new files.

Result: 100K raw + 400K transformed = 500K balanced F1 arrays.

  • Crash-safe: index CSV appended after EVERY array save.
  • Resumable: skips already-existing transform files on restart.
  • No API calls — purely local file operations.

Usage:
    python3 scripts/transform_f1_sample.py              # default 100K sample
    python3 scripts/transform_f1_sample.py --sample 50000
    python3 scripts/transform_f1_sample.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "real_world_10k" / "raw"
INDEX_CSV    = PROJECT_ROOT / "data" / "real_world_10k" / "index.csv"

MIN_ARRAY_LEN = 50

INDEX_COLUMNS = [
    "array_id", "file", "year", "event", "round", "session",
    "driver", "lap", "channel", "n_elements", "dtype", "size_bytes",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Structural transforms (identical to stock/crypto/earthquake)
# ═══════════════════════════════════════════════════════════════════════════

def apply_transforms(arr: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}

    # REV — reverse
    out["REV"] = arr[::-1].copy()

    # SHUF — random shuffle (destroys sorted structure → introsort territory)
    shuf = arr.copy()
    rng.shuffle(shuf)
    out["SHUF"] = shuf

    # QBIN50 — quantize to 50 bins (creates many duplicates → heapsort territory)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        edges = np.linspace(vmin, vmax, 51)
        idx = np.clip(np.digitize(arr, edges) - 1, 0, 49)
        centers = (edges[:-1] + edges[1:]) / 2.0
        out["QBIN50"] = centers[idx]

    # PSORT10 — sort then perturb 10% (nearly sorted → timsort sweet spot)
    sorted_arr = np.sort(arr.copy())
    n = len(sorted_arr)
    n_swap = max(2, int(n * 0.10))
    swap_idx = rng.choice(n, size=n_swap, replace=False)
    swap_vals = sorted_arr[swap_idx].copy()
    rng.shuffle(swap_vals)
    sorted_arr[swap_idx] = swap_vals
    out["PSORT10"] = sorted_arr

    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Index helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_existing_files() -> set[str]:
    if INDEX_CSV.exists():
        try:
            df = pd.read_csv(INDEX_CSV, low_memory=False)
            return set(df["file"].tolist())
        except Exception:
            return set()
    return set()


def append_index_row(row: dict) -> None:
    with open(INDEX_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Apply transforms to sampled F1 arrays"
    )
    parser.add_argument("--sample", type=int, default=100_000,
                        help="Number of raw F1 arrays to sample (default: 100000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without creating files")
    args = parser.parse_args()

    # Load index
    df = pd.read_csv(INDEX_CSV, low_memory=False)
    f1_raw = df[df["file"].str.startswith("f1_")]

    # Exclude any F1 files that are already transforms (from a previous run)
    transforms = ["_REV.", "_SHUF.", "_QBIN50.", "_PSORT10.",
                   "_REV_", "_SHUF_", "_QBIN50_", "_PSORT10_"]
    mask = ~f1_raw["file"].apply(lambda f: any(t in f for t in transforms))
    f1_raw = f1_raw[mask]

    print("=" * 70)
    print("  F1 Transform Generator")
    print("=" * 70)
    print(f"  Total raw F1 arrays:   {len(f1_raw):,}")
    print(f"  Sample size:           {min(args.sample, len(f1_raw)):,}")
    print(f"  Transforms per array:  4 (REV, SHUF, QBIN50, PSORT10)")
    print(f"  New files to create:   ~{min(args.sample, len(f1_raw)) * 4:,}")
    print(f"  Output:                {OUTPUT_DIR}")
    print("=" * 70)

    if args.dry_run:
        print("\nDRY RUN — no files created.")
        return

    # Sample
    sample_size = min(args.sample, len(f1_raw))
    sampled = f1_raw.sample(n=sample_size, random_state=42)

    # Load existing filenames to skip
    existing_files = load_existing_files()

    created = 0
    skipped = 0
    errors = 0
    total_bytes = 0
    t_start = time.time()

    for i, (_, row) in enumerate(sampled.iterrows()):
        fpath = OUTPUT_DIR / row["file"]

        if not fpath.exists():
            errors += 1
            continue

        try:
            arr = np.loadtxt(fpath)
        except Exception:
            errors += 1
            continue

        if arr.size < MIN_ARRAY_LEN:
            continue

        # Deterministic seed from filename
        seed = hash(row["file"]) & 0xFFFFFFFF
        transforms_dict = apply_transforms(arr, seed)

        for tname, tarr in transforms_dict.items():
            # Build transform filename: insert transform name before .csv
            base = row["file"]
            if base.endswith(".csv"):
                new_fname = base[:-4] + f"_{tname}.csv"
            else:
                new_fname = base + f"_{tname}.csv"

            if new_fname in existing_files:
                skipped += 1
                continue

            # Filter non-finite
            tarr = tarr[np.isfinite(tarr)]
            if tarr.size < MIN_ARRAY_LEN:
                continue

            # Save array
            np.savetxt(OUTPUT_DIR / new_fname, tarr, fmt="%.10g")
            nbytes = (OUTPUT_DIR / new_fname).stat().st_size
            total_bytes += nbytes

            # Append to index
            new_row = {
                "array_id": len(existing_files),
                "file": new_fname,
                "year": row.get("year", 0),
                "event": row.get("event", ""),
                "round": row.get("round", 0),
                "session": row.get("session", ""),
                "driver": row.get("driver", ""),
                "lap": row.get("lap", ""),
                "channel": str(row.get("channel", "")) + f"_{tname}",
                "n_elements": tarr.size,
                "dtype": str(tarr.dtype),
                "size_bytes": nbytes,
            }
            append_index_row(new_row)
            existing_files.add(new_fname)
            created += 1

        # Progress
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (sample_size - i - 1) / rate
            print(f"  [{i+1:>7,}/{sample_size:,}]  "
                  f"created={created:,}  skipped={skipped:,}  "
                  f"{total_bytes/(1024**2):,.1f} MB  "
                  f"ETA {eta/60:.1f} min")

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  F1 TRANSFORMS COMPLETE")
    print(f"{'='*70}")
    print(f"  Source arrays:    {sample_size:,} (sampled from {len(f1_raw):,})")
    print(f"  Created:          {created:,} transform files")
    print(f"  Skipped (exist):  {skipped:,}")
    print(f"  Errors:           {errors:,}")
    print(f"  Size:             {total_bytes/(1024**2):,.1f} MB")
    print(f"  Time:             {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Final combined totals
    df2 = pd.read_csv(INDEX_CSV, low_memory=False)
    print(f"\n  UPDATED DATASET:")
    for prefix, label in [("f1_", "F1"), ("stock_", "Stock"),
                           ("weather_", "Weather"), ("crypto_", "Crypto"),
                           ("quake_", "Earthquake")]:
        c = len(df2[df2["file"].str.startswith(prefix)])
        if c > 0:
            print(f"    {label:12s} {c:>10,}")
    print(f"    {'TOTAL':12s} {len(df2):>10,}")
    print(f"    Total size:  {df2['size_bytes'].sum()/(1024**2):,.1f} MB")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
