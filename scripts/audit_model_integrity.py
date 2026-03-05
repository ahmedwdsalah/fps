#!/usr/bin/env python3
"""
MODEL INTEGRITY AUDIT — Is XGBoost v5 cheating?
=================================================
This script runs an HONEST evaluation to answer:

  1. Is the 93.1% gap-closed number inflated? (computed on training data?)
  2. How does the model compare to truly dumb baselines?
  3. Is there data leakage between train and test?
  4. What happens on ONLY the held-out test split?
  5. What about domain holdout (unseen domains)?

No tricks. No filtering. Just facts.

Usage:
    python3 scripts/audit_model_integrity.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
DATA_CSV    = ROOT / "data" / "training_dataset.csv"
MODEL_PATH  = ROOT / "models" / "xgboost_v5" / "xgb_v5.json"
RESULTS_DIR = ROOT / "results" / "audit"

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
TIME_COLS  = ["time_introsort", "time_heapsort", "time_timsort"]
SEED = 42


def regret_metrics(times: np.ndarray, picks: np.ndarray) -> dict:
    """Compute VBS/SBS/model regret for a set of instances.
    
    times: (N, 3) array of timing for each algorithm
    picks: (N,) array of algorithm names chosen by some strategy
    """
    n = len(times)
    algo_idx = {a: i for i, a in enumerate(ALGORITHMS)}
    
    # VBS = per-instance best
    vbs_times = times.min(axis=1)
    vbs_total = vbs_times.sum()
    
    # SBS = single algorithm with lowest total
    totals = {a: times[:, i].sum() for i, a in enumerate(ALGORITHMS)}
    sbs_algo = min(totals, key=totals.get)
    sbs_total = totals[sbs_algo]
    
    # Strategy time
    strategy_times = np.array([times[i, algo_idx[picks[i]]] for i in range(n)])
    strategy_total = strategy_times.sum()
    
    # Gap metrics
    gap = sbs_total - vbs_total
    if gap > 0:
        gap_closed = 100 * (sbs_total - strategy_total) / gap
    else:
        gap_closed = 0.0
    
    regret = strategy_times - vbs_times
    
    return dict(
        n_instances=n,
        vbs_total=round(vbs_total, 6),
        sbs_total=round(sbs_total, 6),
        sbs_algo=sbs_algo,
        strategy_total=round(strategy_total, 6),
        gap_pct=round(100 * gap / sbs_total, 2) if sbs_total > 0 else 0,
        gap_closed_pct=round(gap_closed, 2),
        mean_regret_us=round(regret.mean() * 1e6, 2),
        median_regret_us=round(float(np.median(regret)) * 1e6, 2),
        p95_regret_us=round(float(np.percentile(regret, 95)) * 1e6, 2),
        perfect_picks_pct=round(100 * (regret == 0).mean(), 2),
    )


def main():
    print("=" * 70)
    print("  MODEL INTEGRITY AUDIT — Is XGBoost v5 cheating?")
    print("=" * 70)
    t0 = time.time()
    
    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    df = pd.read_csv(DATA_CSV)
    print(f"  Total arrays: {len(df):,}")
    
    times_all = df[TIME_COLS].values
    
    # ── Load model ────────────────────────────────────────────────────────
    print("\n[2] Loading model...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    le = LabelEncoder().fit(ALGORITHMS)
    
    X_all = df[FEATURE_NAMES].values
    y_pred_all = le.inverse_transform(model.predict(X_all))
    y_true_all = df["best_algorithm"].values
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: Reproduce the EXACT same train/test split used during training
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 1: Reproduce exact train/test split")
    print("=" * 70)
    
    # Step 1: Apply the same noise filter
    time_cols_vals = df[TIME_COLS].values
    sorted_times = np.sort(time_cols_vals, axis=1)
    best_time = sorted_times[:, 0]
    second_time = sorted_times[:, 1]
    margin = (second_time - best_time) / (second_time + 1e-15)
    has_margin = margin >= 0.05
    is_large = df["n_elements"].values >= 2000
    keep = has_margin | is_large
    df_filtered = df[keep].reset_index(drop=True)
    print(f"  After noise filter: {len(df_filtered):,} (removed {(~keep).sum():,} noisy)")
    
    # Step 2: Apply the same undersampling
    counts = df_filtered["best_algorithm"].value_counts()
    min_count = counts.min()
    cap = int(min_count * 3.0)
    parts = []
    for cls in counts.index:
        subset = df_filtered[df_filtered["best_algorithm"] == cls]
        if len(subset) > cap:
            subset = subset.sample(n=cap, random_state=SEED)
        parts.append(subset)
    df_bal = pd.concat(parts, ignore_index=True)
    df_bal = df_bal.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"  After undersampling: {len(df_bal):,}")
    
    # Step 3: Apply the same 70/15/15 split
    X_bal = df_bal[FEATURE_NAMES].values
    y_bal = df_bal["best_algorithm"].values
    
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X_bal, y_bal, np.arange(len(df_bal)),
        test_size=0.30, stratify=y_bal, random_state=SEED
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp,
        test_size=0.50, stratify=y_temp, random_state=SEED
    )
    
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    
    # Now evaluate on ONLY the test split
    y_test_pred = le.inverse_transform(model.predict(X_test))
    test_acc = (y_test_pred == y_test).mean()
    
    # Get timing data for these test arrays
    test_files = df_bal.iloc[idx_test]["file"].values
    test_rows = df_bal.iloc[idx_test]
    times_test = test_rows[TIME_COLS].values
    
    print(f"\n  Held-out TEST SPLIT results:")
    print(f"    Accuracy:  {test_acc*100:.1f}%")
    
    test_regret = regret_metrics(times_test, y_test_pred)
    print(f"    Gap closed (test only): {test_regret['gap_closed_pct']:.1f}%")
    print(f"    Mean regret: {test_regret['mean_regret_us']:.1f} μs")
    print(f"    Perfect picks: {test_regret['perfect_picks_pct']:.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: Regret on TRAINING data vs TEST data (detect inflation)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 2: Training-set regret vs Test-set regret (leakage check)")
    print("=" * 70)
    
    y_train_pred = le.inverse_transform(model.predict(X_train))
    times_train = df_bal.iloc[idx_train][TIME_COLS].values
    
    train_regret = regret_metrics(times_train, y_train_pred)
    
    print(f"\n  {'Metric':<30s}  {'TRAIN':>10s}  {'TEST':>10s}  {'Δ':>8s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}")
    
    tr_gc = train_regret['gap_closed_pct']
    te_gc = test_regret['gap_closed_pct']
    print(f"  {'Gap closed':30s}  {tr_gc:>9.1f}%  {te_gc:>9.1f}%  {te_gc-tr_gc:>+7.1f}%")
    
    tr_mr = train_regret['mean_regret_us']
    te_mr = test_regret['mean_regret_us']
    print(f"  {'Mean regret (μs)':30s}  {tr_mr:>10.1f}  {te_mr:>10.1f}  {te_mr-tr_mr:>+8.1f}")
    
    tr_pp = train_regret['perfect_picks_pct']
    te_pp = test_regret['perfect_picks_pct']
    print(f"  {'Perfect picks':30s}  {tr_pp:>9.1f}%  {te_pp:>9.1f}%  {te_pp-tr_pp:>+7.1f}%")
    
    if tr_gc - te_gc > 10:
        print(f"\n  ⚠️  OVERFITTING: Train gap-closed is {tr_gc-te_gc:.1f}pp higher than test!")
    else:
        print(f"\n  ✓ Train-test gap is small ({tr_gc-te_gc:.1f}pp). No severe overfitting.")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: Compare to DUMB BASELINES
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 3: Dumb baselines comparison (on TEST split only)")
    print("=" * 70)
    
    baselines = {}
    
    # Baseline 1: Always pick each algorithm
    for algo in ALGORITHMS:
        picks = np.full(len(X_test), algo)
        acc = (picks == y_test).mean()
        reg = regret_metrics(times_test, picks)
        baselines[f"Always {algo}"] = dict(accuracy=acc, **reg)
    
    # Baseline 2: Random uniform
    rng = np.random.RandomState(SEED)
    random_picks = rng.choice(ALGORITHMS, size=len(X_test))
    random_acc = (random_picks == y_test).mean()
    baselines["Random uniform"] = dict(
        accuracy=random_acc, **regret_metrics(times_test, random_picks)
    )
    
    # Baseline 3: Majority class (from training set)
    from collections import Counter
    majority = Counter(y_train).most_common(1)[0][0]
    majority_picks = np.full(len(X_test), majority)
    majority_acc = (majority_picks == y_test).mean()
    baselines[f"Majority ({majority})"] = dict(
        accuracy=majority_acc, **regret_metrics(times_test, majority_picks)
    )
    
    # Baseline 4: Pick by array size heuristic (small=introsort, big=timsort)
    test_sizes = df_bal.iloc[idx_test]["n_elements"].values
    size_picks = np.where(test_sizes < 1000, "introsort",
                  np.where(test_sizes < 10000, "heapsort", "timsort"))
    size_acc = (size_picks == y_test).mean()
    baselines["Size heuristic"] = dict(
        accuracy=size_acc, **regret_metrics(times_test, size_picks)
    )
    
    # XGBoost v5
    baselines["XGBoost v5 (ours)"] = dict(
        accuracy=test_acc, **test_regret
    )
    
    print(f"\n  {'Strategy':<25s}  {'Acc':>7s}  {'Gap Closed':>11s}  {'Mean Reg(μs)':>13s}  {'Perfect':>8s}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*11}  {'-'*13}  {'-'*8}")
    
    for name in ["Random uniform", f"Majority ({majority})", "Always introsort",
                  "Always heapsort", "Always timsort", "Size heuristic",
                  "XGBoost v5 (ours)"]:
        b = baselines[name]
        flag = " ←" if name == "XGBoost v5 (ours)" else ""
        print(f"  {name:25s}  {b['accuracy']*100:>6.1f}%  "
              f"{b['gap_closed_pct']:>10.1f}%  "
              f"{b['mean_regret_us']:>13.1f}  "
              f"{b['perfect_picks_pct']:>7.1f}%{flag}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEST 4: Regret on DATA THE MODEL NEVER SAW (arrays not in balanced set)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 4: Regret on arrays the model NEVER saw")
    print("=" * 70)
    
    # Arrays that were removed by noise filter OR undersampling
    balanced_files = set(df_bal["file"].tolist())
    unseen_mask = ~df["file"].isin(balanced_files)
    df_unseen = df[unseen_mask].reset_index(drop=True)
    print(f"  Arrays NOT in balanced training set: {len(df_unseen):,}")
    print(f"  (These were removed by noise filter or undersampling)")
    
    if len(df_unseen) > 0:
        X_unseen = df_unseen[FEATURE_NAMES].values
        y_unseen_true = df_unseen["best_algorithm"].values
        y_unseen_pred = le.inverse_transform(model.predict(X_unseen))
        times_unseen = df_unseen[TIME_COLS].values
        
        unseen_acc = (y_unseen_pred == y_unseen_true).mean()
        unseen_regret = regret_metrics(times_unseen, y_unseen_pred)
        
        print(f"  Accuracy on unseen arrays: {unseen_acc*100:.1f}%")
        print(f"  Gap closed on unseen:      {unseen_regret['gap_closed_pct']:.1f}%")
        print(f"  Mean regret:               {unseen_regret['mean_regret_us']:.1f} μs")
        
        # Break down: noisy-filtered vs undersampled-away
        filtered_files = set(df_filtered["file"].tolist())
        noise_only = df[~df["file"].isin(filtered_files)].reset_index(drop=True)
        undersample_only = df_filtered[~df_filtered["file"].isin(balanced_files)].reset_index(drop=True)
        
        if len(noise_only) > 0:
            X_n = noise_only[FEATURE_NAMES].values
            y_n_pred = le.inverse_transform(model.predict(X_n))
            y_n_true = noise_only["best_algorithm"].values
            t_n = noise_only[TIME_COLS].values
            nr = regret_metrics(t_n, y_n_pred)
            print(f"\n  Noise-filtered arrays ({len(noise_only):,}):")
            print(f"    Accuracy:    {(y_n_pred == y_n_true).mean()*100:.1f}%")
            print(f"    Gap closed:  {nr['gap_closed_pct']:.1f}%")
        
        if len(undersample_only) > 0:
            X_u = undersample_only[FEATURE_NAMES].values
            y_u_pred = le.inverse_transform(model.predict(X_u))
            y_u_true = undersample_only["best_algorithm"].values
            t_u = undersample_only[TIME_COLS].values
            ur = regret_metrics(t_u, y_u_pred)
            print(f"\n  Undersampled-away arrays ({len(undersample_only):,}):")
            print(f"    Accuracy:     {(y_u_pred == y_u_true).mean()*100:.1f}%")
            print(f"    Gap closed:   {ur['gap_closed_pct']:.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEST 5: Per-domain results on TEST split (domain bias check)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 5: Per-domain accuracy on TEST split")
    print("=" * 70)
    
    test_domains = df_bal.iloc[idx_test]["domain"].values
    
    print(f"\n  {'Domain':<15s}  {'N':>7s}  {'Acc':>7s}  {'GapClosed':>10s}  {'MeanReg(μs)':>12s}")
    print(f"  {'-'*15}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*12}")
    
    for domain in sorted(set(test_domains)):
        dmask = test_domains == domain
        d_times = times_test[dmask]
        d_pred = y_test_pred[dmask]
        d_true = y_test[dmask]
        d_acc = (d_pred == d_true).mean()
        d_reg = regret_metrics(d_times, d_pred)
        print(f"  {domain:15s}  {dmask.sum():>7,}  {d_acc*100:>6.1f}%  "
              f"{d_reg['gap_closed_pct']:>9.1f}%  "
              f"{d_reg['mean_regret_us']:>12.1f}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEST 6: The ORIGINAL 93.1% claim — is it honest?
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TEST 6: The 93.1% gap-closed claim — honest check")
    print("=" * 70)
    
    # Original: computed on ALL 1.18M arrays
    all_regret = regret_metrics(times_all, y_pred_all)
    
    print(f"\n  Gap closed on ALL data (as reported):   {all_regret['gap_closed_pct']:.1f}%")
    print(f"  Gap closed on TEST SPLIT only:          {test_regret['gap_closed_pct']:.1f}%")
    if len(df_unseen) > 0:
        print(f"  Gap closed on UNSEEN arrays:            {unseen_regret['gap_closed_pct']:.1f}%")
    
    delta = all_regret['gap_closed_pct'] - test_regret['gap_closed_pct']
    if abs(delta) < 3:
        print(f"\n  ✓ Difference is only {abs(delta):.1f}pp. The 93.1% number is honest.")
    elif delta > 3:
        print(f"\n  ⚠️  Gap-closed is {delta:.1f}pp higher on all data than on test-only.")
        print(f"     The original claim is INFLATED by including training data.")
    else:
        print(f"\n  Interesting: test-only is actually BETTER by {abs(delta):.1f}pp.")
    
    # ═══════════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)
    
    issues = []
    
    # Check 1: Train-test gap
    if tr_gc - te_gc > 10:
        issues.append(f"Overfitting: train gap-closed {tr_gc:.1f}% vs test {te_gc:.1f}%")
    
    # Check 2: Better than all dumb baselines?
    best_dumb = max(
        v['gap_closed_pct'] for k, v in baselines.items() 
        if k != "XGBoost v5 (ours)"
    )
    model_gc = test_regret['gap_closed_pct']
    if model_gc <= best_dumb:
        issues.append(f"Model ({model_gc:.1f}%) doesn't beat best baseline ({best_dumb:.1f}%)")
    else:
        margin = model_gc - best_dumb
        print(f"\n  ✓ Beats best dumb baseline by {margin:.1f}pp")
    
    # Check 3: Inflation
    if delta > 5:
        issues.append(f"Reported 93.1% is inflated by {delta:.1f}pp vs test-only")
    else:
        print(f"  ✓ Reported gap-closed is within {abs(delta):.1f}pp of test-only")
    
    # Check 4: Model better than SBS?
    if test_regret['gap_closed_pct'] < 0:
        issues.append("Model is WORSE than just always picking the best single algorithm!")
    else:
        print(f"  ✓ Model closes {test_regret['gap_closed_pct']:.1f}% of gap vs SBS")
    
    if not issues:
        print(f"\n  VERDICT: Model is LEGITIMATE. Not cheating.")
        print(f"  The numbers are honest within the test methodology.")
    else:
        print(f"\n  VERDICT: ISSUES FOUND:")
        for issue in issues:
            print(f"    ✗ {issue}")
    
    # ── Save results ──────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        test_split_regret=test_regret,
        train_split_regret=train_regret,
        baselines={k: v for k, v in baselines.items()},
        all_data_regret=all_regret,
        unseen_regret=unseen_regret if len(df_unseen) > 0 else None,
        issues=issues,
        verdict="LEGITIMATE" if not issues else "ISSUES_FOUND",
    )
    out_file = RESULTS_DIR / "integrity_audit.json"
    out_file.write_text(json.dumps(output, indent=2, default=str))
    
    elapsed = time.time() - t0
    print(f"\n  Audit completed in {elapsed:.1f}s")
    print(f"  Results saved: {out_file}")


if __name__ == "__main__":
    main()
