#!/usr/bin/env python3
"""Full dataset diversity & bias audit for training readiness."""

import pandas as pd
import numpy as np

idx = pd.read_csv("data/real_world_10k/index.csv", low_memory=False)

print("=" * 70)
print("  DATASET DIVERSITY & BIAS ANALYSIS")
print("=" * 70)

# ── 1. Domain distribution ───────────────────────────────────────────────
print("\n1. DOMAIN DISTRIBUTION")
print("-" * 50)
domain_map = [
    ("f1_", "F1"),
    ("stock_", "Stock"),
    ("weather_", "Weather"),
    ("crypto_", "Crypto"),
    ("quake_", "Earthquake"),
]
for prefix, label in domain_map:
    sub = idx[idx["file"].str.startswith(prefix)]
    pct = 100 * len(sub) / len(idx)
    print(f"  {label:12s}  {len(sub):>10,}  ({pct:5.1f}%)")
print(f"  {'TOTAL':12s}  {len(idx):>10,}")

# ── 2. Transform coverage ────────────────────────────────────────────────
print("\n2. TRANSFORM COVERAGE (anti-bias)")
print("-" * 50)
transforms = ["REV", "SHUF", "QBIN50", "PSORT10"]
for prefix, label in domain_map:
    sub = idx[idx["file"].str.startswith(prefix)]
    if len(sub) == 0:
        continue
    t_counts = {t: 0 for t in transforms}
    raw_count = 0
    for f in sub["file"]:
        found = False
        for t in transforms:
            if f"_{t}." in f or f"_{t}_" in f:
                t_counts[t] += 1
                found = True
                break
        if not found:
            raw_count += 1
    total = raw_count + sum(t_counts.values())
    print(f"  {label}:")
    print(f"    RAW:     {raw_count:>8,}  ({100*raw_count/max(total,1):5.1f}%)")
    for t, c in t_counts.items():
        print(f"    {t:8s} {c:>8,}  ({100*c/max(total,1):5.1f}%)")

# ── 3. Array size distribution ───────────────────────────────────────────
print("\n3. ARRAY SIZE DISTRIBUTION")
print("-" * 50)
sizes = idx["n_elements"]
buckets = [
    (50, 100, "50-100"),
    (100, 500, "100-500"),
    (500, 1000, "500-1K"),
    (1000, 5000, "1K-5K"),
    (5000, 10000, "5K-10K"),
    (10000, 50000, "10K-50K"),
    (50000, 200000, "50K-200K"),
]
for lo, hi, label in buckets:
    c = ((sizes >= lo) & (sizes < hi)).sum()
    pct = 100 * c / len(idx)
    bar = "#" * int(pct / 2)
    print(f"  {label:10s}  {c:>10,}  ({pct:5.1f}%)  {bar}")
print(f"  Median: {int(sizes.median()):,}  Mean: {int(sizes.mean()):,}  Std: {int(sizes.std()):,}")

# ── 4. Channel / variable diversity ──────────────────────────────────────
print("\n4. VARIABLE / CHANNEL DIVERSITY")
print("-" * 50)
for prefix, label in domain_map:
    sub = idx[idx["file"].str.startswith(prefix)]
    if len(sub) == 0:
        continue
    channels = sub["channel"].nunique()
    print(f"  {label:12s}  {channels:>4} unique channels")

# ── 5. Sample feature distribution (quick spot-check) ────────────────────
print("\n5. SORTING ALGORITHM WINNER SPOT-CHECK")
print("-" * 50)
print("  Sampling 1,000 arrays (200 per domain, max 10K elements) ...")

from pathlib import Path
import time as _time

RAW_DIR = Path("data/real_world_10k/raw")

results = {"heapsort": 0, "introsort": 0, "timsort": 0}
n_sampled = 0
MAX_SIZE = 10_000  # skip huge arrays for speed

for prefix, label in domain_map:
    sub = idx[idx["file"].str.startswith(prefix)]
    # keep only reasonably-sized arrays for speed
    sub = sub[sub["n_elements"] <= MAX_SIZE]
    if len(sub) == 0:
        continue
    sample = sub.sample(n=min(200, len(sub)), random_state=42)
    for _, row in sample.iterrows():
        fpath = RAW_DIR / row["file"]
        if not fpath.exists():
            continue
        try:
            arr = np.loadtxt(fpath)
            if arr.size < 50 or arr.size > MAX_SIZE:
                continue
        except Exception:
            continue

        times = {}
        for kind, algo_name in [("quicksort", "introsort"), ("heapsort", "heapsort"), ("stable", "timsort")]:
            best = float("inf")
            for _ in range(3):
                a = arr.copy()
                t0 = _time.perf_counter()
                np.sort(a, kind=kind)
                t1 = _time.perf_counter()
                best = min(best, t1 - t0)
            times[algo_name] = best

        winner = min(times, key=times.get)
        results[winner] += 1
        n_sampled += 1
    print(f"    {label}: done ({n_sampled} total so far)")

print(f"  Sampled: {n_sampled:,} arrays")
for algo, count in sorted(results.items()):
    pct = 100 * count / max(n_sampled, 1)
    bar = "#" * int(pct / 2)
    print(f"  {algo:12s}  {count:>6,}  ({pct:5.1f}%)  {bar}")

# ── 6. Verdict ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  VERDICT")
print("=" * 70)
total = len(idx)
algo_pcts = {algo: 100 * count / max(n_sampled, 1) for algo, count in results.items()}
min_pct = min(algo_pcts.values())
max_pct = max(algo_pcts.values())

issues = []
if total < 100_000:
    issues.append(f"SMALL: only {total:,} arrays (need 100K+)")
if min_pct < 10:
    issues.append(f"IMBALANCED: {min(algo_pcts, key=algo_pcts.get)} only {min_pct:.1f}%")
if max_pct > 70:
    issues.append(f"DOMINATED: {max(algo_pcts, key=algo_pcts.get)} at {max_pct:.1f}%")

n_domains = sum(1 for p, _ in domain_map if len(idx[idx["file"].str.startswith(p)]) > 0)
if n_domains < 3:
    issues.append(f"LOW DIVERSITY: only {n_domains} domains")

if not issues:
    print("  GOOD TO TRAIN!")
    print(f"  - {total:,} arrays across {n_domains} domains")
    print(f"  - Algorithm balance: {dict(sorted(algo_pcts.items()))}")
    print(f"  - Transforms ensure all 3 algorithms have winning regions")
else:
    print("  ISSUES FOUND:")
    for issue in issues:
        print(f"    - {issue}")
print("=" * 70)
