#!/usr/bin/env python3
"""Quick check of data layout before Step 3."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_extraction import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent

print("=" * 60)
print("DATA LAYOUT CHECK FOR STEP 3")
print("=" * 60)

for f in ["all_samples", "train", "val", "test_A", "test_B"]:
    df = pd.read_parquet(ROOT / f"data/benchmark/{f}.parquet")
    print(f"\n--- {f} ---")
    print(f"  Shape: {df.shape}")
    if "n" in df.columns:
        print(f"  N range: {df['n'].min():,} - {df['n'].max():,}")
    if "distribution" in df.columns:
        print(f"  Distributions: {df['distribution'].value_counts().to_dict()}")
    if "structure" in df.columns:
        print(f"  Structures: {df['structure'].value_counts().to_dict()}")
    if "best_algorithm" in df.columns:
        print(f"  Winners: {df['best_algorithm'].value_counts().to_dict()}")

print("\n\n--- REAL-WORLD DATA FOR EVALUATION ---")
v4 = pd.read_parquet(ROOT / "data/real_world_v4/real_world_v4_combined.parquet")
real = v4[~v4["domain"].isin(["synthetic", "largescale"])]
print(f"  Truly-real arrays: {len(real)}")
print(f"  Domains: {real['domain'].value_counts().to_dict()}")
print(f"  Winners: {real['best_algorithm'].value_counts().to_dict()}")

train = pd.read_parquet(ROOT / "data/benchmark/train.parquet")
feat_in_train = [f for f in FEATURE_NAMES if f in train.columns]
feat_in_real = [f for f in FEATURE_NAMES if f in real.columns]
print(f"\n  Features in train: {len(feat_in_train)}/16")
print(f"  Features in real: {len(feat_in_real)}/16")
print(f"  Match: {feat_in_train == feat_in_real}")

# Time columns
tcols = ["time_introsort", "time_heapsort", "time_timsort"]
print(f"\n  Time cols in train: {all(c in train.columns for c in tcols)}")
print(f"  Time cols in real: {all(c in real.columns for c in tcols)}")

# Check XGBoost
try:
    import xgboost
    print(f"\n  XGBoost: {xgboost.__version__}")
except ImportError:
    print("\n  XGBoost: NOT INSTALLED")

# Check sklearn
try:
    import sklearn
    print(f"  sklearn: {sklearn.__version__}")
except ImportError:
    print("  sklearn: NOT INSTALLED")
