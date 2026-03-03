#!/usr/bin/env python3
"""Check F1 data potential for 10K dataset."""
import os
from collections import Counter

raw_dir = "data/real_world_bigtest/raw/"
files = [f for f in os.listdir(raw_dir) if f.endswith(".npy")]
print(f"Current F1 .npy files: {len(files)}")

years = Counter()
sessions = Counter()
channels = Counter()
for f in files:
    parts = f.split("_")
    if len(parts) >= 3:
        years[parts[1]] += 1
    if len(parts) >= 4:
        sessions[parts[3]] += 1
    ch = parts[-1].replace(".npy", "")
    channels[ch] += 1

print(f"By year: {dict(sorted(years.items()))}")
print(f"By session type: {dict(sessions)}")
print(f"By channel: {dict(channels)}")
print()

import fastf1
fastf1.Cache.enable_cache("data/f1_cache")

print("Available GP events per season:")
total_gps = 0
for yr in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
    try:
        s = fastf1.get_event_schedule(yr)
        races = len(s[s["EventFormat"] != "testing"])
        total_gps += races
        print(f"  {yr}: {races} GPs")
    except Exception as e:
        print(f"  {yr}: error - {e}")

print(f"\nTotal GPs across all seasons: {total_gps}")
print()

# Estimate potential arrays
# Per GP: ~7 sessions (FP1,FP2,FP3,Q,R + maybe Sprint)
# Per session: ~10 numeric channels
# Per session: full session + 5 drivers x full + 5 drivers x 3 laps = 1 + 5 + 15 = 21 telemetry blocks
# Per block: ~10 channels
# Conservative: per GP = 3 sessions x 10 channels x 6 blocks = 180 arrays
ARRAYS_PER_GP = 180
print(f"Conservative estimate per GP: ~{ARRAYS_PER_GP} arrays")
print(f"5 years x ~22 GPs = ~110 GPs")
print(f"Potential F1 arrays: ~{110 * ARRAYS_PER_GP:,}")
print()

# What we need from other sources to hit 10K
f1_est = 110 * ARRAYS_PER_GP
shortfall = max(0, 10_000 - f1_est)
print(f"F1 alone could give: ~{f1_est:,} arrays")
print(f"Shortfall to 10K: {shortfall}")
