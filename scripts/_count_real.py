#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv("data/real_world_v4/real_world_v4_combined.csv")
real_only = df[df["domain"] != "synthetic"]
print("v4 combined WITHOUT synthetic:", len(real_only), "arrays")
print("Domains:", dict(real_only["domain"].value_counts()))
print("Best algo:", dict(real_only["best_algorithm"].value_counts()))
print()

v2 = pd.read_csv("data/real_world_v2/f1_real_world_v2_results.csv")
v3 = pd.read_csv("data/real_world_v3/real_world_v3_results.csv")
v4new = pd.read_csv("data/real_world_v4/real_world_v4_new_data.csv")
bigtest = pd.read_csv("data/real_world_bigtest/results_xgboost_v3.csv")

print("UNIQUE REAL arrays:")
print(f"  v2 (F1 telemetry): {len(v2)}")
print(f"  v3 (financial+seismic): {len(v3)}")
print(f"  v4 new (weather+nasa+largescale+eq): {len(v4new)}")
print(f"  bigtest (F1 expanded raw): {len(bigtest)}")
total = len(v2) + len(v3) + len(v4new) + len(bigtest)
print(f"  TOTAL: ~{total}")
print()

print("Bigtest columns:", list(bigtest.columns))
has_feat = all(c in bigtest.columns for c in ["length_norm", "adj_sorted_ratio"])
print(f"Bigtest has features: {has_feat}")
print()

# Check overlap between v2 and v4 combined
v4_f1 = real_only[real_only["domain"] == "f1_telemetry"]
print(f"v4 combined F1 arrays: {len(v4_f1)}")
print(f"v2 F1 arrays: {len(v2)}")
print(f"These are the SAME arrays (v4 includes v2 + v3 + v4new)")
print()
real_with_features = len(real_only)
print(f"=== REAL ARRAYS WITH 16 FEATURES + TIMINGS: {real_with_features} ===")
print(f"=== BIGTEST (raw only, needs feature extraction): {len(bigtest)} ===")
print(f"=== BIGTEST raw .npy files: 175 ===")
