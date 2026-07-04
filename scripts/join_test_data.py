#!/usr/bin/env python3
"""
join_test_data.py — Merge all raw CSV files for a domain into a single .npz.
Replaces 200K individual files with one compressed archive.

Usage:
  python scripts/join_test_data.py f1
  python scripts/join_test_data.py stock
"""

import sys, shutil
from pathlib import Path
import numpy as np
import pandas as pd

DOMAIN = sys.argv[1] if len(sys.argv) > 1 else None

if not DOMAIN:
    print("Usage: python scripts/join_test_data.py <domain>")
    print("  domains: f1, stock, crypto, earthquake, weather")
    sys.exit(1)

ROOT = Path("data/test_200k") / DOMAIN
RAW_DIR = ROOT / "raw"
INDEX_CSV = ROOT / "index.csv"
NPZ_PATH = ROOT / f"{DOMAIN}_200k.npz"

if not INDEX_CSV.exists():
    print(f"No index.csv found at {INDEX_CSV}. Run the fetch script first.")
    sys.exit(1)

print(f"Loading index: {INDEX_CSV}")
idx = pd.read_csv(INDEX_CSV, low_memory=False)
n = len(idx)
print(f"  {n:,} rows")

print(f"Loading {n:,} CSV files from {RAW_DIR} ...")
arrays = []
failed = 0
for i, fname in enumerate(idx["file"]):
    fpath = RAW_DIR / fname
    try:
        arr = np.loadtxt(fpath)
        arrays.append(arr)
    except Exception as e:
        failed += 1
        print(f"  WARN: {fname}: {e}")
    if (i + 1) % 50000 == 0:
        print(f"  {i+1:,}/{n:,} ...")

print(f"  Done. {len(arrays):,} arrays loaded ({failed} failed)")

# Join all arrays into one object array (arrays vary in length)
arr_obj = np.empty(len(arrays), dtype=object)
for i, a in enumerate(arrays):
    arr_obj[i] = a

print(f"Saving {NPZ_PATH} ...")
np.savez_compressed(NPZ_PATH, arrays=arr_obj, file_names=idx["file"].values[:len(arrays)])
size_mb = NPZ_PATH.stat().st_size / (1024**2)
print(f"  Saved: {size_mb:.0f} MB")

# Verify
print("Verifying ...")
loaded = np.load(NPZ_PATH, allow_pickle=True)
loaded_arrays = loaded["arrays"]
loaded_names = loaded["file_names"]
print(f"  Arrays: {len(loaded_arrays):,}")
print(f"  Names:  {len(loaded_names):,}")
print(f"  Sample: {loaded_names[0]}, n={len(loaded_arrays[0]):,}")

# Delete old raw/ directory
print(f"\nDeleting {RAW_DIR} ...")
shutil.rmtree(RAW_DIR)
print(f"  Deleted.")

# Keep index.csv for reference
idx.to_csv(INDEX_CSV, index=False)
print(f"  Kept {INDEX_CSV} ({len(idx):,} rows)")

print(f"\nDone. {NPZ_PATH.name} is the only data file.")
print(f"  Index: {INDEX_CSV.name}")
