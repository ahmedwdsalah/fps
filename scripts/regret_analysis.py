#!/usr/bin/env python3
"""
Step 4: VBS / SBS / Model Regret Analysis
==========================================
Uses the training_dataset.csv (which has real timing for all 1.18M arrays)
and the XGBoost v5 model to compute:

  - VBS (Virtual Best Solver): always pick the true fastest → ceiling
  - SBS (Single Best Solver): always pick the one algorithm that minimizes
    total time across all instances → naive baseline
  - MODEL: XGBoost v5 predictions → our selector

Key metrics:
  - VBS-SBS Gap: how much room for improvement exists
  - Model Regret: how much slower model is vs VBS (lower = better)
  - Model Lift: how much faster model is vs SBS (higher = better)
  - Per-instance penalty distributions

Usage:
    python3 scripts/regret_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
DATA_CSV    = ROOT / "data" / "training_dataset.csv"
MODEL_PATH  = ROOT / "models" / "xgboost_v5" / "xgb_v5.json"
RESULTS_DIR = ROOT / "results" / "xgboost_v5"

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
TIME_COLS  = ["time_introsort", "time_heapsort", "time_timsort"]


def main():
    print("=" * 70)
    print("  STEP 4: VBS / SBS / Model Regret Analysis")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading data...")
    df = pd.read_csv(DATA_CSV)
    print(f"  Rows: {len(df):,}")

    times = df[TIME_COLS].values  # shape (N, 3)
    algo_names = np.array(ALGORITHMS)

    # ── VBS: per-instance best ────────────────────────────────────────────
    vbs_times = times.min(axis=1)
    vbs_total = vbs_times.sum()
    vbs_winner = algo_names[times.argmin(axis=1)]

    # ── SBS: single best algorithm (lowest total time) ────────────────────
    totals = {algo: times[:, i].sum() for i, algo in enumerate(ALGORITHMS)}
    sbs_algo = min(totals, key=totals.get)
    sbs_total = totals[sbs_algo]
    sbs_col_idx = ALGORITHMS.index(sbs_algo)
    sbs_times = times[:, sbs_col_idx]

    print(f"\n  Algorithm total times:")
    for algo, t in sorted(totals.items(), key=lambda x: x[1]):
        print(f"    {algo:12s}  {t:.3f}s")

    # ── VBS-SBS Gap ───────────────────────────────────────────────────────
    gap_pct = 100 * (sbs_total - vbs_total) / sbs_total
    print(f"\n{'='*60}")
    print(f"  VBS total:  {vbs_total:.3f}s (ceiling — perfect oracle)")
    print(f"  SBS total:  {sbs_total:.3f}s (always {sbs_algo})")
    print(f"  VBS-SBS Gap: {gap_pct:.2f}%")
    print(f"{'='*60}")

    if gap_pct < 1:
        print("\n  WARNING: Gap < 1% — algorithm selection barely matters.")
        print("  One algorithm dominates. Consider if thesis framing needs adjustment.")

    # ── Model predictions ─────────────────────────────────────────────────
    print("\nLoading XGBoost v5 model...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))

    X = df[FEATURE_NAMES].values
    le = LabelEncoder().fit(ALGORITHMS)
    y_pred_enc = model.predict(X)
    y_pred = le.inverse_transform(y_pred_enc)

    # Model time = time of the algorithm the model picks
    model_times = np.array([
        times[i, ALGORITHMS.index(y_pred[i])] for i in range(len(df))
    ])
    model_total = model_times.sum()

    # ── Model metrics ─────────────────────────────────────────────────────
    model_regret_vs_vbs = 100 * (model_total - vbs_total) / vbs_total
    model_lift_vs_sbs   = 100 * (sbs_total - model_total) / sbs_total

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  VBS (oracle):   {vbs_total:.3f}s")
    print(f"  MODEL (XGBv5):  {model_total:.3f}s")
    print(f"  SBS ({sbs_algo:>9s}): {sbs_total:.3f}s")
    print(f"")
    print(f"  VBS-SBS Gap:           {gap_pct:.2f}%  (room for improvement)")
    print(f"  Model regret vs VBS:   {model_regret_vs_vbs:.2f}%  (lower = better)")
    print(f"  Model lift vs SBS:     {model_lift_vs_sbs:.2f}%  (higher = better)")

    # How much of the gap does the model close?
    if sbs_total > vbs_total:
        gap_closed = 100 * (sbs_total - model_total) / (sbs_total - vbs_total)
        print(f"  Gap closed by model:   {gap_closed:.1f}%")
    print(f"{'='*60}")

    # ── Per-algorithm breakdown ───────────────────────────────────────────
    print(f"\n  Model picks distribution:")
    for algo in ALGORITHMS:
        n = (y_pred == algo).sum()
        pct = 100 * n / len(df)
        print(f"    {algo:12s} {n:>10,}  ({pct:.1f}%)")

    # ── Per-instance regret distribution ──────────────────────────────────
    per_instance_regret = model_times - vbs_times  # seconds wasted per array
    per_instance_sbs_penalty = sbs_times - vbs_times

    print(f"\n  Per-instance regret (model vs VBS):")
    print(f"    Mean:   {per_instance_regret.mean()*1e6:.1f} μs")
    print(f"    Median: {np.median(per_instance_regret)*1e6:.1f} μs")
    print(f"    P95:    {np.percentile(per_instance_regret, 95)*1e6:.1f} μs")
    print(f"    P99:    {np.percentile(per_instance_regret, 99)*1e6:.1f} μs")
    print(f"    Max:    {per_instance_regret.max()*1e6:.1f} μs")
    print(f"    Zero (perfect picks): {(per_instance_regret == 0).sum():,} "
          f"({100*(per_instance_regret == 0).mean():.1f}%)")

    print(f"\n  Per-instance penalty (SBS vs VBS):")
    print(f"    Mean:   {per_instance_sbs_penalty.mean()*1e6:.1f} μs")
    print(f"    Median: {np.median(per_instance_sbs_penalty)*1e6:.1f} μs")
    print(f"    P99:    {np.percentile(per_instance_sbs_penalty, 99)*1e6:.1f} μs")
    print(f"    Max:    {per_instance_sbs_penalty.max()*1e6:.1f} μs")

    # ── Breakdown by array size bucket ────────────────────────────────────
    print(f"\n  Regret by array size:")
    print(f"  {'Size bucket':>12s}  {'N':>8s}  {'VBS':>10s}  {'Model':>10s}  "
          f"{'SBS':>10s}  {'Gap%':>6s}  {'Lift%':>6s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*6}")

    buckets = [
        (50, 500, "<500"),
        (500, 2000, "500-2K"),
        (2000, 10000, "2K-10K"),
        (10000, 50000, "10K-50K"),
        (50000, 200001, "50K+"),
    ]
    sizes = df["n_elements"].values

    for lo, hi, label in buckets:
        mask = (sizes >= lo) & (sizes < hi)
        if mask.sum() == 0:
            continue
        v = vbs_times[mask].sum()
        m = model_times[mask].sum()
        s = sbs_times[mask].sum()
        g = 100 * (s - v) / s if s > 0 else 0
        l = 100 * (s - m) / s if s > 0 else 0
        print(f"  {label:>12s}  {mask.sum():>8,}  {v:>10.3f}  {m:>10.3f}  "
              f"{s:>10.3f}  {g:>5.1f}%  {l:>5.1f}%")

    # ── Save results ──────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = dict(
        total_instances=len(df),
        vbs_total_s=round(vbs_total, 6),
        sbs_total_s=round(sbs_total, 6),
        sbs_algorithm=sbs_algo,
        model_total_s=round(model_total, 6),
        vbs_sbs_gap_pct=round(gap_pct, 4),
        model_regret_vs_vbs_pct=round(model_regret_vs_vbs, 4),
        model_lift_vs_sbs_pct=round(model_lift_vs_sbs, 4),
        gap_closed_pct=round(gap_closed, 2) if sbs_total > vbs_total else None,
        per_instance_regret_us=dict(
            mean=round(per_instance_regret.mean() * 1e6, 2),
            median=round(float(np.median(per_instance_regret)) * 1e6, 2),
            p95=round(float(np.percentile(per_instance_regret, 95)) * 1e6, 2),
            p99=round(float(np.percentile(per_instance_regret, 99)) * 1e6, 2),
            max=round(float(per_instance_regret.max()) * 1e6, 2),
            zero_pct=round(100 * float((per_instance_regret == 0).mean()), 2),
        ),
        algorithm_totals_s={algo: round(t, 6) for algo, t in totals.items()},
        model_picks={algo: int((y_pred == algo).sum()) for algo in ALGORITHMS},
    )
    results_file = RESULTS_DIR / "regret_analysis.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved: {results_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
