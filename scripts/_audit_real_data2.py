#!/usr/bin/env python3
"""Audit all real-world data in the project."""
import pandas as pd
import os

BASE = "/Users/ahmed/Desktop/thesis/My-Master-thesis"

files = {
    "real_world v1 (F1)": "data/real_world/f1_real_world_results.csv",
    "real_world v2 (F1 expanded)": "data/real_world_v2/f1_real_world_v2_results.csv",
    "real_world v3 (financial+seismic)": "data/real_world_v3/real_world_v3_results.csv",
    "real_world v4 combined": "data/real_world_v4/real_world_v4_combined.csv",
    "real_world v4 new data only": "data/real_world_v4/real_world_v4_new_data.csv",
    "real_world bigtest": "data/real_world_bigtest/results_xgboost_v3.csv",
}

print("=" * 70)
print("REAL-WORLD DATA AUDIT")
print("=" * 70)

for name, relpath in files.items():
    path = os.path.join(BASE, relpath)
    try:
        df = pd.read_csv(path)
        print(f"\n{name}")
        print(f"  File: {relpath}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns[:8])}")
        if "source" in df.columns:
            src = dict(df["source"].value_counts().head(10))
            print(f"  Sources: {df['source'].nunique()} unique -> {src}")
        if "domain" in df.columns:
            print(f"  Domains: {df['domain'].nunique()} unique -> {dict(df['domain'].value_counts())}")
        if "array_name" in df.columns:
            print(f"  Arrays: {df['array_name'].nunique()} unique names")
        if "best_algorithm" in df.columns:
            print(f"  Best algo dist: {dict(df['best_algorithm'].value_counts())}")
        if "fastest_algorithm" in df.columns:
            print(f"  Best algo dist: {dict(df['fastest_algorithm'].value_counts())}")
        feat_cols = [c for c in df.columns if c in [
            "length_norm", "adj_sorted_ratio", "runs_ratio", "inversion_ratio",
            "entropy_ratio", "duplicate_ratio", "dispersion_ratio"
        ]]
        print(f"  Has features: {len(feat_cols) > 0} ({len(feat_cols)} found)")
    except Exception as e:
        print(f"\n{name}: ERROR - {e}")

print("\n" + "=" * 70)
print("SYNTHETIC TRAINING DATA (what models were trained on)")
print("=" * 70)
for split in ["train", "val", "test"]:
    path = os.path.join(BASE, f"data/features/{split}_features.parquet")
    try:
        df = pd.read_parquet(path)
        print(f"  {split}: {len(df)} arrays, {len(df.columns)} columns")
    except Exception as e:
        print(f"  {split}: ERROR - {e}")

raw_dir = os.path.join(BASE, "data/real_world_bigtest/raw")
if os.path.isdir(raw_dir):
    raw_files = os.listdir(raw_dir)
    print(f"\n{'=' * 70}")
    print(f"BIGTEST RAW FILES: {len(raw_files)} files")
    print("=" * 70)
    for f in sorted(raw_files)[:20]:
        fpath = os.path.join(raw_dir, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {f} ({size_mb:.1f} MB)")
    if len(raw_files) > 20:
        print(f"  ... and {len(raw_files) - 20} more")
