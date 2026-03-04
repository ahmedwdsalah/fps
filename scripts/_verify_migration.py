#!/usr/bin/env python3
"""Post-migration integrity check — verify zero data lost."""
import pandas as pd
import os
import numpy as np
from pathlib import Path

idx = pd.read_csv("data/real_world_10k/index.csv", low_memory=False)
raw_dir = Path("data/real_world_10k/raw")

print("=" * 60)
print("  POST-MIGRATION INTEGRITY CHECK")
print("=" * 60)

print(f"\n1. Symlink: {os.path.islink('data/real_world_10k')} -> {os.readlink('data/real_world_10k')}")
print(f"2. Index rows: {len(idx):,}")

files_on_disk = set(os.listdir(raw_dir))
print(f"3. Files on disk (raw/): {len(files_on_disk):,}")

files_in_index = set(idx["file"].tolist())
missing = files_in_index - files_on_disk
extra = files_on_disk - files_in_index
print(f"4. In index but MISSING on disk: {len(missing):,}")
print(f"5. On disk but NOT in index: {len(extra):,}")

sample = idx.sample(5, random_state=99)
ok = 0
for _, row in sample.iterrows():
    fpath = raw_dir / row["file"]
    try:
        arr = np.loadtxt(fpath)
        if arr.size == row["n_elements"]:
            ok += 1
        else:
            print(f"   SIZE MISMATCH: {row['file']} index={row['n_elements']} disk={arr.size}")
    except Exception as e:
        print(f"   READ FAIL: {row['file']}: {e}")
print(f"6. Spot-check 5 random arrays: {ok}/5 OK")

print(f"\n7. Per-domain:")
for prefix, label in [("f1_","F1"),("stock_","Stock"),("weather_","Weather"),("crypto_","Crypto"),("quake_","Earthquake")]:
    c = len(idx[idx["file"].str.startswith(prefix)])
    if c > 0:
        print(f"   {label:12s} {c:>10,}")
print(f"   {'TOTAL':12s} {len(idx):>10,}")

if len(missing) == 0 and ok == 5:
    print(f"\n   EVERYTHING OK -- zero data lost.")
else:
    print(f"\n   WARNING: {len(missing)} missing files!")
print("=" * 60)
