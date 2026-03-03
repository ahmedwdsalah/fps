#!/usr/bin/env python3
"""Audit all available real data in the repository."""
import pandas as pd
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 1. V4 combined
v4 = pd.read_parquet(ROOT / 'data/real_world_v4/real_world_v4_combined.parquet')
real = v4[~v4['domain'].isin(['synthetic', 'largescale'])]
print('=== REAL-WORLD V4 COMBINED ===')
print(f'Total in v4: {len(v4)}')
print(f'Synthetic: {len(v4[v4["domain"]=="synthetic"])}')
print(f'Largescale: {len(v4[v4["domain"]=="largescale"])}')
print(f'Truly real: {len(real)}')
print()
print('By domain:')
print(real['domain'].value_counts().to_string())
print()
print('Best algorithm distribution (real only):')
print(real['best_algorithm'].value_counts().to_string())
print()
print('Size stats (real only):')
print(f'  Min n: {real["n"].min():,}')
print(f'  Max n: {real["n"].max():,}')
print(f'  Mean n: {real["n"].mean():,.0f}')

# 2. Bigtest
bigtest_dir = ROOT / 'data/real_world_bigtest/raw'
npy_files = list(bigtest_dir.glob('*.npy'))
print(f'\n=== BIGTEST RAW .NPY ===')
print(f'Total .npy files: {len(npy_files)}')

bt_csv = ROOT / 'data/real_world_bigtest/results_xgboost_v3.csv'
if bt_csv.exists():
    bt = pd.read_csv(bt_csv)
    print(f'Tested (>=10K): {len(bt)}')
    print(f'Size range: {bt["n_elements"].min():,} - {bt["n_elements"].max():,}')
    print(f'Best actual:')
    print(bt['best_actual'].value_counts().to_string())

# 3. Columns check
FEAT_NAMES = ['length_norm','adj_sorted_ratio','duplicate_ratio','dispersion_ratio',
              'runs_ratio','inversion_ratio','entropy_ratio','skewness_t',
              'kurtosis_excess_t','longest_run_ratio','iqr_norm','mad_norm',
              'top1_freq_ratio','top5_freq_ratio','outlier_ratio','mean_abs_diff_norm']

print(f'\n=== DATA QUALITY ===')
print(f'V4 columns: {list(v4.columns)}')
print(f'Has timing cols: {all(c in real.columns for c in ["time_introsort","time_heapsort","time_timsort"])}')
print(f'Has feature cols: {sum(1 for c in FEAT_NAMES if c in real.columns)}/16')

# 4. Check overlap between bigtest and v4
print(f'\n=== POTENTIAL TOTAL REAL DATA ===')
print(f'V4 real arrays: {len(real)}')
print(f'Bigtest arrays (tested): {len(bt) if bt_csv.exists() else 0}')

# Check if bigtest arrays are already in v4
if bt_csv.exists():
    bt_files = set(bt['file'].values)
    # These are separate - bigtest was tested separately
    print(f'Bigtest arrays NOT in v4 (separate download): ~{len(bt)}')
    print(f'POTENTIAL TOTAL: ~{len(real) + len(bt)} real arrays')
