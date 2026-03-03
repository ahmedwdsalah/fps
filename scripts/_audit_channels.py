#!/usr/bin/env python3
"""Quick audit of data quality per channel in the 100K dataset."""
import pandas as pd
import numpy as np
import os

df = pd.read_csv("data/real_world_10k/index.csv")
raw_dir = "data/real_world_10k/raw"

print("=== DATA QUALITY PER CHANNEL (sampling 500 per channel) ===\n")

for ch in sorted(df["channel"].unique()):
    ch_df = df[df["channel"] == ch]
    sample = ch_df.sample(min(500, len(ch_df)), random_state=42)

    all_same = 0
    low_unique = 0
    med_unique = 0
    high_unique = 0

    for _, row in sample.iterrows():
        try:
            arr = np.loadtxt(os.path.join(raw_dir, row["file"]))
            n_unique = len(np.unique(arr))
            if n_unique == 1:
                all_same += 1
            elif n_unique < 10:
                low_unique += 1
            elif n_unique <= 100:
                med_unique += 1
            else:
                high_unique += 1
        except Exception:
            continue

    total = all_same + low_unique + med_unique + high_unique
    if total == 0:
        continue
    print(f"{ch:10s} ({len(ch_df):6,} total):")
    print(f"    All same value: {all_same:3d}/{total} ({all_same/total*100:5.1f}%)  <- USELESS")
    print(f"    < 10 unique:    {low_unique:3d}/{total} ({low_unique/total*100:5.1f}%)")
    print(f"    10-100 unique:  {med_unique:3d}/{total} ({med_unique/total*100:5.1f}%)")
    print(f"    > 100 unique:   {high_unique:3d}/{total} ({high_unique/total*100:5.1f}%)")
    print()

# Summary recommendation
print("=" * 60)
print("RECOMMENDATION:")
print("  Keep: Speed, RPM, Throttle (diverse numeric data)")
print("  Review: nGear (low unique but real sorting pattern)")
print("  Review: X, Y, Z (smooth curves)")
print("  Drop?: DRS (mostly all-zeros)")
