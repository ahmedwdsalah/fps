#!/usr/bin/env python3
"""
Diagnose v6 results: Is the problem in the DATA or the MODEL?
==============================================================
Tests:
  1. Label noise analysis — how many arrays have <10% margin between
     introsort and heapsort? (if huge → data ceiling)
  2. Feature separability — can features actually distinguish classes?
     (KNN on raw features as upper-bound proxy)
  3. Confusion deep-dive — WHERE are the errors happening?
     (size brackets, domains, margin buckets)
  4. Bayes-optimal estimate — what's the theoretical accuracy ceiling
     given label noise?
  5. Model capacity check — is XGBoost underfitting or overfitting?
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES

DATA_CSV = ROOT / "data" / "diverse_training_data.csv"

def main():
    df = pd.read_csv(DATA_CSV)
    print(f"Dataset: {len(df):,} arrays, {df['source_id'].nunique():,} sources\n")

    time_cols = ["time_introsort", "time_heapsort", "time_timsort"]
    times = df[time_cols].values
    best = df["best_algorithm"].values

    # ─────────────────────────────────────────────────────────────────────
    # TEST 1: Label noise — how close are introsort vs heapsort?
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("TEST 1: LABEL NOISE — timing margins between algorithms")
    print("=" * 70)

    t_intro = df["time_introsort"].values
    t_heap  = df["time_heapsort"].values
    t_tim   = df["time_timsort"].values

    # For each array, margin = (2nd_best - best) / best
    best_time = times.min(axis=1)
    second_best = np.sort(times, axis=1)[:, 1]
    margin = (second_best - best_time) / np.where(best_time > 0, best_time, 1e-12)

    # For introsort vs heapsort specifically
    ih_margin = np.abs(t_intro - t_heap) / np.minimum(t_intro, t_heap)

    print(f"\n  Overall best-vs-2nd margin:")
    for pct in [5, 10, 15, 20, 30, 50]:
        n = (margin < pct/100).sum()
        print(f"    <{pct}% margin: {n:>7,} arrays ({100*n/len(df):.1f}%)")

    print(f"\n  Introsort vs Heapsort margin specifically:")
    for pct in [1, 2, 5, 10, 15, 20, 30]:
        n = (ih_margin < pct/100).sum()
        print(f"    <{pct}% margin: {n:>7,} arrays ({100*n/len(df):.1f}%)")

    # Among the introsort-labeled arrays, how many have heapsort nearly as fast?
    intro_mask = best == "introsort"
    heap_mask  = best == "heapsort"
    if intro_mask.sum() > 0:
        ih_margin_intro = (t_heap[intro_mask] - t_intro[intro_mask]) / t_intro[intro_mask]
        print(f"\n  Among {intro_mask.sum():,} introsort-labeled arrays:")
        print(f"    Mean margin over heapsort: {ih_margin_intro.mean()*100:.1f}%")
        print(f"    Median margin: {np.median(ih_margin_intro)*100:.1f}%")
        for pct in [5, 10, 20]:
            n = (ih_margin_intro < pct/100).sum()
            print(f"    Heapsort within {pct}%: {n:>6,} ({100*n/intro_mask.sum():.1f}%)")

    if heap_mask.sum() > 0:
        hi_margin_heap = (t_intro[heap_mask] - t_heap[heap_mask]) / t_heap[heap_mask]
        print(f"\n  Among {heap_mask.sum():,} heapsort-labeled arrays:")
        print(f"    Mean margin over introsort: {hi_margin_heap.mean()*100:.1f}%")
        print(f"    Median margin: {np.median(hi_margin_heap)*100:.1f}%")
        for pct in [5, 10, 20]:
            n = (hi_margin_heap < pct/100).sum()
            print(f"    Introsort within {pct}%: {n:>6,} ({100*n/heap_mask.sum():.1f}%)")

    # ─────────────────────────────────────────────────────────────────────
    # TEST 2: Theoretical accuracy ceiling ("if-confused-just-merge")
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 2: ACCURACY CEILING — what if we merge introsort+heapsort?")
    print("=" * 70)

    # If model only had to distinguish {introsort OR heapsort} vs {timsort}
    binary_best = np.where(best == "timsort", "timsort", "compare_sort")
    from collections import Counter
    c = Counter(binary_best)
    print(f"  Binary class distribution:")
    for k, v in sorted(c.items()):
        print(f"    {k:15s}: {v:>7,} ({100*v/len(df):.1f}%)")

    # Load v6 predictions and see binary accuracy
    pred_file = ROOT / "results" / "xgboost_v6" / "predictions_test.csv"
    if pred_file.exists():
        preds = pd.read_csv(pred_file)
        binary_true = np.where(preds["true"] == "timsort", "timsort", "compare")
        binary_pred = np.where(preds["pred"] == "timsort", "timsort", "compare")
        bin_acc = (binary_true == binary_pred).mean()
        print(f"\n  v6 binary accuracy (timsort vs rest): {bin_acc*100:.1f}%")
        print(f"  v6 3-class accuracy:                   {(preds['true']==preds['pred']).mean()*100:.1f}%")
        print(f"  → {(bin_acc - (preds['true']==preds['pred']).mean())*100:.1f}pp lost "
              f"specifically from intro↔heap confusion")

    # ─────────────────────────────────────────────────────────────────────
    # TEST 3: Feature separability — can features distinguish classes?
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 3: FEATURE SEPARABILITY — can features distinguish introsort vs heapsort?")
    print("=" * 70)

    X = df[FEATURE_NAMES].values
    y = df["best_algorithm"].values

    # Compare feature means per class
    print(f"\n  Feature means by class:")
    print(f"  {'Feature':>22s}  {'introsort':>10s}  {'heapsort':>10s}  {'timsort':>10s}  {'IH_diff':>8s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    ih_diffs = []
    for i, feat in enumerate(FEATURE_NAMES):
        m_i = X[y == "introsort", i].mean()
        m_h = X[y == "heapsort", i].mean()
        m_t = X[y == "timsort", i].mean()
        # Cohen's d between introsort and heapsort
        pooled_std = np.sqrt((X[y=="introsort", i].var() + X[y=="heapsort", i].var()) / 2)
        cohens_d = abs(m_i - m_h) / pooled_std if pooled_std > 0 else 0
        ih_diffs.append((feat, cohens_d))
        flag = "" if cohens_d < 0.2 else " *" if cohens_d < 0.5 else " **" if cohens_d < 0.8 else " ***"
        print(f"  {feat:>22s}  {m_i:>10.4f}  {m_h:>10.4f}  {m_t:>10.4f}  d={cohens_d:.3f}{flag}")

    print(f"\n  Cohen's d scale: <0.2 negligible, 0.2-0.5 small*, 0.5-0.8 medium**, >0.8 large***")

    ih_diffs.sort(key=lambda x: -x[1])
    print(f"\n  Top features separating introsort vs heapsort:")
    for feat, d in ih_diffs[:5]:
        print(f"    {feat:>22s}: d={d:.3f}")
    print(f"  Max Cohen's d: {ih_diffs[0][1]:.3f}")
    if ih_diffs[0][1] < 0.2:
        print(f"  → ALL features have negligible separation! This is a DATA problem.")
    elif ih_diffs[0][1] < 0.5:
        print(f"  → Weak separation. Likely a data problem, model can only do so much.")
    else:
        print(f"  → Some features have meaningful separation. Model improvement possible.")

    # ─────────────────────────────────────────────────────────────────────
    # TEST 4: Confusion by size bracket — is it worse for small arrays?
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 4: ERROR BREAKDOWN BY SIZE AND DOMAIN")
    print("=" * 70)

    if pred_file.exists():
        preds = pd.read_csv(pred_file)

        # Merge with test data to get size info
        print(f"\n  Accuracy by array size:")
        brackets = [(0, 500, "<500"), (500, 1000, "500-1K"),
                     (1000, 5000, "1K-5K"), (5000, 20000, "5K-20K"),
                     (20000, float("inf"), "20K+")]
        for lo, hi, label in brackets:
            mask = (preds["n_elements"] >= lo) & (preds["n_elements"] < hi)
            if mask.sum() == 0:
                continue
            acc = (preds.loc[mask, "true"] == preds.loc[mask, "pred"]).mean()
            n = mask.sum()

            # Per-class accuracy in this bracket
            intro_mask = mask & (preds["true"] == "introsort")
            heap_mask  = mask & (preds["true"] == "heapsort")
            tim_mask   = mask & (preds["true"] == "timsort")
            i_acc = (preds.loc[intro_mask, "pred"] == "introsort").mean() if intro_mask.sum() > 0 else float("nan")
            h_acc = (preds.loc[heap_mask, "pred"] == "heapsort").mean() if heap_mask.sum() > 0 else float("nan")
            t_acc = (preds.loc[tim_mask, "pred"] == "timsort").mean() if tim_mask.sum() > 0 else float("nan")
            print(f"    {label:>8s}: {acc*100:5.1f}% (n={n:>5,})  "
                  f"intro={i_acc*100:4.1f}%  heap={h_acc*100:4.1f}%  tim={t_acc*100:4.1f}%")

    # ─────────────────────────────────────────────────────────────────────
    # TEST 5: Margin-stratified accuracy — errors on noisy labels?
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 5: ACCURACY VS TIMING MARGIN — are errors on noisy labels?")
    print("=" * 70)

    if pred_file.exists() and "timing_margin" in df.columns:
        # Need to match test predictions with original data margins
        # Use the test set indices
        preds_test = pd.read_csv(pred_file)
        # timing_margin is in df but we need to align — use source_id+file to match
        # Actually, let's just recompute from the test data
        print(f"\n  (Using timing_margin from dataset)")

        # We need to reload the v6 split to get test margins
        # Instead, let's compute from predictions
        # Actually predictions_test.csv has source_id — merge with original
        merged = preds_test.merge(
            df[["source_id", "file", "timing_margin", "n_elements"]].drop_duplicates(),
            on=["source_id", "file", "n_elements"],
            how="left"
        )
        if "timing_margin" in merged.columns and merged["timing_margin"].notna().sum() > 0:
            for lo, hi, label in [(0.05, 0.10, "5-10%"), (0.10, 0.20, "10-20%"),
                                   (0.20, 0.50, "20-50%"), (0.50, 1.0, "50-100%"),
                                   (1.0, float("inf"), ">100%")]:
                mask = (merged["timing_margin"] >= lo) & (merged["timing_margin"] < hi)
                if mask.sum() == 0:
                    continue
                acc = merged.loc[mask, "correct"].mean()
                n = mask.sum()
                print(f"    margin {label:>8s}: {acc*100:5.1f}% acc  (n={n:>5,})")
        else:
            print("  Could not match timing_margin — computing from raw times")

    elif "timing_margin" not in df.columns:
        print("  timing_margin column not in dataset, computing from raw times...")
        best_t = times.min(axis=1)
        second_t = np.sort(times, axis=1)[:, 1]
        computed_margin = (second_t - best_t) / best_t
        df["timing_margin_computed"] = computed_margin

        # Distribution of margins
        print(f"\n  Timing margin distribution (all data):")
        for lo, hi, label in [(0.05, 0.10, "5-10%"), (0.10, 0.20, "10-20%"),
                               (0.20, 0.50, "20-50%"), (0.50, 1.0, "50-100%"),
                               (1.0, float("inf"), ">100%")]:
            mask = (computed_margin >= lo) & (computed_margin < hi)
            if mask.sum() == 0:
                continue
            n = mask.sum()
            print(f"    margin {label:>8s}: {n:>7,} arrays ({100*n/len(df):.1f}%)")

    # ─────────────────────────────────────────────────────────────────────
    # TEST 6: Quick KNN upper bound
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 6: KNN SANITY CHECK — feature-based upper bound estimate")
    print("=" * 70)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    groups = df["source_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_te = scaler.transform(X[te_idx])
    y_tr = y[tr_idx]
    y_te = y[te_idx]

    for k in [5, 15, 50]:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_tr, y_tr)
        knn_acc = knn.score(X_te, y_te)
        # Per-class
        knn_pred = knn.predict(X_te)
        from sklearn.metrics import balanced_accuracy_score
        knn_bal = balanced_accuracy_score(y_te, knn_pred)
        print(f"  KNN k={k:>2d}: acc={knn_acc*100:.1f}%  bal_acc={knn_bal*100:.1f}%")

    # ─────────────────────────────────────────────────────────────────────
    # TEST 7: What if we just always predict heapsort for the confused pair?
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TEST 7: BASELINES — trivial models")
    print("=" * 70)

    # SBS: always predict the most common class
    from collections import Counter
    mc = Counter(y_te).most_common(1)[0]
    sbs_acc = (y_te == mc[0]).mean()
    print(f"  Always-predict-{mc[0]}: {sbs_acc*100:.1f}%")

    # Random weighted
    rng = np.random.RandomState(42)
    classes, probs = zip(*[(k, v/len(y_te)) for k, v in Counter(y_te).items()])
    rand_pred = rng.choice(list(classes), size=len(y_te), p=list(probs))
    rand_acc = (y_te == rand_pred).mean()
    print(f"  Weighted random:       {rand_acc*100:.1f}%")

    # v6 for comparison
    print(f"  XGBoost v6:            71.2%")

    # ─────────────────────────────────────────────────────────────────────
    # VERDICT
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT")
    print("=" * 70)

    max_d = ih_diffs[0][1]
    n_small_margin = (ih_margin < 0.10).sum()
    pct_small = 100 * n_small_margin / len(df)

    print(f"\n  Introsort vs Heapsort margin <10%: {n_small_margin:,} ({pct_small:.1f}% of all data)")
    print(f"  Best feature Cohen's d (intro vs heap): {max_d:.3f}")

    if pct_small > 30 and max_d < 0.3:
        print(f"\n  DIAGNOSIS: Primarily a DATA problem.")
        print(f"  - {pct_small:.0f}% of arrays have <10% timing difference between intro & heap")
        print(f"  - Features can barely distinguish them (max Cohen's d = {max_d:.3f})")
        print(f"  - No model can reliably classify what's essentially a coin flip")
        print(f"\n  RECOMMENDATIONS:")
        print(f"  1. ACCEPT: 3-class may have ~70% ceiling. The 45% gap-closed is honest.")
        print(f"  2. RE-FRAME: Report binary (timsort vs comparison-sort) + 3-class separately")
        print(f"  3. RAISE THE TIE FILTER: Increase from 5% to 15-20% to remove noise")
        print(f"  4. ADD FEATURES: timing-relevant features (cache line alignment, branch patterns)")
    elif max_d < 0.3:
        print(f"\n  DIAGNOSIS: Mostly a DATA problem, with some model room.")
        print(f"  Features lack discriminative power for introsort vs heapsort.")
    else:
        print(f"\n  DIAGNOSIS: MODEL problem — features have signal, model isn't using it.")
        print(f"  Try: deeper trees, more trees, different architecture (e.g., LightGBM, neural net)")


if __name__ == "__main__":
    main()
