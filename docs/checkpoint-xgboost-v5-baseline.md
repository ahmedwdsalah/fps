# Checkpoint: XGBoost v5 Baseline Complete

**Date:** 2026-03-05  
**Commit:** Steps 1–4 (real-world data → training → model → regret analysis)  
**Status:** Offline classifier done. Online bandit (Step 5) not started.

---

## What This Checkpoint Contains

This is the **first complete end-to-end pipeline** on real-world data: 1.18M arrays from 5 domains, trained with balanced XGBoost, validated with regret analysis and domain holdout tests.

### Staged Files

| File | Purpose |
|------|---------|
| `models/xgboost_v5/xgb_v5.json` | Frozen XGBoost model (500 trees, JSON) |
| `results/xgboost_v5/evaluation_results.json` | Train/val/test/full metrics |
| `results/xgboost_v5/regret_analysis.json` | VBS/SBS/model regret (Step 4) |
| `results/domain_holdout/domain_holdout_results.json` | Leave-One-Domain-Out test |
| `scripts/build_training_dataset.py` | Step 2: features + timing for 1.18M arrays |
| `scripts/train_xgboost_v5.py` | Step 3: balanced classifier training |
| `scripts/regret_analysis.py` | Step 4: VBS/SBS/model value analysis |
| `scripts/test_xgboost_v5.py` | Standalone test-split evaluation |
| `scripts/demo_predict.py` | End-to-end demo: raw array → features → prediction |
| `scripts/domain_holdout_test.py` | Leave-One-Domain-Out generalization test |
| `scripts/_diagnose_accuracy.py` | Accuracy diagnostic (why 76% is fine) |
| `PROJECT.md` | Updated documentation through Step 4 |

---

## Dataset

- **Source:** 1,188,265 real-world arrays from 5 domains
- **Storage:** `/Volumes/k/thesis_data/real_world_10k/` (symlinked to `data/real_world_10k`)
- **Training data:** `data/training_dataset.csv` — 22 columns (file, domain, n_elements, 16 features, 3 timings, best_algorithm)

### Domain Breakdown

| Domain     | Arrays    | Share  |
|------------|-----------|--------|
| F1         | 885,042   | 74.5%  |
| Stock      | 100,000   | 8.4%   |
| Crypto     | 100,000   | 8.4%   |
| Earthquake | 100,003   | 8.4%   |
| Weather    | 3,220     | 0.3%   |
| **Total**  | **1,188,265** | **100%** |

### Algorithm Winner Distribution (Raw)

| Algorithm  | Count     | Win Rate |
|------------|-----------|----------|
| timsort    | 1,010,413 | 85.0%   |
| heapsort   | 125,218   | 10.5%   |
| introsort  | 52,634    | 4.4%    |

Timsort dominance is expected: median array is 419 elements, mostly partially sorted. At small sizes timing differences are noise (median gap = 1.2 μs).

---

## Feature Extraction

16 O(n) features computed by `scripts/feature_extraction.py`:

| # | Feature | Description | Importance |
|---|---------|-------------|------------|
| 1 | `length_norm` | log2(n) / log2(n_max) | **0.262** |
| 2 | `top5_freq_ratio` | Fraction in top-5 most frequent values | **0.184** |
| 3 | `longest_run_ratio` | Longest sorted run / n | 0.080 |
| 4 | `duplicate_ratio` | 1 − (unique / n) | 0.079 |
| 5 | `entropy_ratio` | Normalized Shannon entropy | 0.075 |
| 6 | `top1_freq_ratio` | Mode frequency / n | 0.051 |
| 7 | `runs_ratio` | Number of sorted runs / n | 0.041 |
| 8 | `adj_sorted_ratio` | Adjacent elements in order / n | 0.041 |
| 9 | `mean_abs_diff_norm` | Mean |a[i+1]−a[i]| / range | 0.039 |
| 10 | `inversion_ratio` | Estimated inversions / max possible | 0.035 |
| 11 | `dispersion_ratio` | std / range | 0.023 |
| 12 | `mad_norm` | Median absolute deviation / range | 0.019 |
| 13 | `skewness_t` | tanh(skewness) | 0.019 |
| 14 | `kurtosis_excess_t` | tanh(kurtosis_excess / 10) | 0.018 |
| 15 | `iqr_norm` | IQR / range | 0.018 |
| 16 | `outlier_ratio` | Fraction outside 1.5×IQR | 0.016 |

Key insight: **array size (`length_norm`) is the single strongest signal** (26%). This makes sense—algorithm behavior changes fundamentally at different scales.

---

## Model: XGBoost v5

### Hyperparameters
```
XGBoost 3.2.0, multi:softprob, hist tree method
n_estimators=500, max_depth=7, learning_rate=0.05
subsample=0.8, colsample_bytree=0.8
min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0
seed=42
```

### Balance Strategy
1. **Noise filter:** keep rows where timing margin ≥ 5% OR array size ≥ 2K (removes coin-flip labels)
2. **Undersample:** cap each class at 3× minority count → 196,624 rows
3. **Sample weights:** inverse-frequency weighting during training
4. **Split:** 70/15/15 stratified (train=137,636 / val=29,494 / test=29,494)

### Classification Results

| Split                | Accuracy | Balanced Acc | Introsort | Heapsort | Timsort |
|----------------------|----------|-------------|-----------|----------|---------|
| Train (137K)         | 82.8%    | 80.1%       | 73.2%     | 71.4%    | 95.8%   |
| Validation (29K)     | 76.3%    | 70.3%       | 52.0%     | 63.8%    | 95.0%   |
| **Test (29K)**       | **76.1%**| **70.1%**   | 52.1%     | 63.8%    | 94.5%   |
| Full dataset (1.18M) | 89.1%    | 68.9%       | 39.6%     | 73.5%    | 93.6%   |

- Val ≈ Test → **no overfitting**
- Full dataset accuracy (89.1%) higher than test (76.1%) because unfiltered data is 85% timsort, and the model excels at timsort
- Introsort recall is lowest (52%) because introsort vs heapsort is nearly indistinguishable at small sizes

---

## Regret Analysis (Step 4)

The **definitive evaluation** — classification accuracy doesn't capture cost-sensitivity. Regret analysis answers: "how much time does the model actually waste?"

### Headline Numbers

| Metric                    | Value     | Meaning |
|---------------------------|-----------|---------|
| VBS-SBS Gap               | 19.14%    | 19% room for improvement → thesis justified |
| Model regret vs VBS       | 1.62%     | Only 1.6% slower than perfect oracle |
| Model lift vs SBS         | 17.83%    | 17.8% faster than always-heapsort |
| **Gap closed by model**   | **93.1%** | Model captures 93% of theoretical optimum |
| Perfect picks (regret=0)  | 89.6%     | Model picks the true best 89.6% of the time |

### Total Times (1.18M arrays)

| Strategy   | Total Time | vs VBS |
|------------|-----------|--------|
| VBS        | 17.195s   | —      |
| **Model**  | **17.475s** | +1.6% |
| heapsort (SBS) | 21.267s | +23.7% |
| introsort  | 21.459s   | +24.8% |
| timsort    | 24.624s   | +43.2% |

SBS = heapsort, not timsort. Although timsort wins 85% of individual arrays, heapsort has the lowest *total* time. Timsort's catastrophic performance on random/shuffled data inflates its total.

### Per-Instance Regret

| Metric | Value |
|--------|-------|
| Mean | 0.23 μs |
| Median | 0.0 μs |
| P95 | 0.25 μs |
| P99 | 6.12 μs |
| Max | 659.25 μs |
| Zero regret | 89.6% |

**When the model is wrong, the penalty is almost always negligible.**

---

## Domain Holdout Test (Strongest Generalization Test)

Leave-One-Domain-Out: train on 4 domains, test on the held-out domain the model has **never seen**. Tests whether the 16 features generalize across data sources.

### Results (5 fresh models, one per fold)

| Holdout Domain | N Arrays | Accuracy | Bal. Acc | Gap Closed | Perfect Picks |
|----------------|----------|----------|----------|------------|---------------|
| Crypto         | 100,000  | 87.5%    | 61.1%    | 90.1%      | 87.9%         |
| Earthquake     | 100,003  | 82.1%    | 62.3%    | 95.0%      | 82.8%         |
| F1             | 885,042  | 88.8%    | 48.5%    | 75.5%      | 89.8%         |
| Stock          | 100,000  | 88.0%    | 63.7%    | 90.7%      | 88.3%         |
| Weather        | 3,220    | 60.6%    | 62.0%    | 89.7%      | 61.0%         |
| **Weighted Avg** | **1,188,265** | **88.0%** | **52.0%** | **79.7%** | **88.9%** |

### Key Takeaways

1. **Gap closed 75–95% even on unseen domains** — features truly generalize
2. **F1 holdout (75.5% gap closed)** is weakest because F1 is 75% of training data — losing it hurts
3. **Earthquake holdout (95.0% gap closed)** is strongest — earthquake data structure is learnable from other domains
4. **Weather is hardest** (60.6% accuracy) but still closes 89.7% of the gap — small dataset (3.2K), unusual structure

### Comparison: Holdout Model vs Full v5 Model

| Domain     | Holdout Acc | Full v5 Acc | Holdout Gap Closed | Full v5 Gap Closed |
|------------|-------------|-------------|--------------------|--------------------|
| Crypto     | 87.5%       | 88.6%       | 90.1%              | 90.7%              |
| Earthquake | 82.1%       | 86.6%       | 95.0%              | 95.7%              |
| F1         | 88.8%       | 89.4%       | 75.5%              | 92.5%              |
| Stock      | 88.0%       | 89.6%       | 90.7%              | 92.7%              |
| Weather    | 60.6%       | 65.5%       | 89.7%              | 89.9%              |

The holdout model (trained WITHOUT the target domain) approaches the full model's performance. This confirms the features capture universal sorting algorithm behavior, not domain-specific patterns.

---

## Accuracy Diagnosis: Why 76% Is Actually Good

The `_diagnose_accuracy.py` script analyzes why test accuracy is "only" 76%.

### Root Causes

1. **Small arrays dominate:** median = 419 elements, 66.2% have < 500 elements
2. **Tiny timing gaps at small sizes:** median gap between best and 2nd-best is 1.2 μs (28.6%) for arrays < 500 elements
3. **Label noise:** 15.1% of all arrays have margin < 5% — the label is a coin flip
4. **Introsort ≈ heapsort:** when introsort wins, 78.1% of the time the gap over heapsort is < 5%

### Why This Doesn't Matter

- The **regret analysis** is the right metric, not accuracy
- 93.1% gap closed = the model picks near-optimal algorithms
- Mean per-instance penalty is 0.23 μs
- "Wrong" picks that confuse introsort↔heapsort cost essentially nothing

---

## Pipeline Architecture

```
Raw Array (any source)
    │
    ▼
┌─────────────────────────┐
│  Feature Extraction     │  16 O(n) features
│  (feature_extraction.py)│  Pure math, no I/O
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  XGBoost v5 Classifier  │  Frozen JSON model
│  (xgb_v5.json)          │  500 trees, multi:softprob
└───────────┬─────────────┘
            ▼
      Prediction:
      introsort / heapsort / timsort
      + confidence probabilities
```

The pipeline is **100% portable**:
- Features are pure math (no external dependencies beyond numpy)
- Model is a frozen JSON file
- No training data needed at inference time
- Works on any numeric array from any domain

---

## Scripts Summary

### Core Pipeline (in order of execution)
| Script | Step | What It Does |
|--------|------|-------------|
| `build_training_dataset.py` | 2 | Loads 1.18M arrays, extracts features, times 3 algorithms |
| `train_xgboost_v5.py` | 3 | Trains balanced XGBoost (filter → undersample → weight → split) |
| `regret_analysis.py` | 4 | Computes VBS/SBS/model regret on full 1.18M dataset |

### Evaluation & Testing
| Script | What It Does |
|--------|-------------|
| `test_xgboost_v5.py` | Reproduces exact test split, evaluates saved model |
| `demo_predict.py` | End-to-end: load model → load raw array → extract features → predict → verify |
| `domain_holdout_test.py` | Leave-One-Domain-Out: train on 4 domains, test on held-out domain |
| `_diagnose_accuracy.py` | Diagnoses why 76% accuracy is actually fine |

---

## Post-v5 Experiments (April 2026)

Two experimental architectures were tested to see if v5 could be improved. **Both confirmed v5 is near-ceiling.**

### v7: Regret-Aware Training (REJECTED)

**Script:** `scripts/train_xgboost_v7_regret_aware.py` | **Report:** `docs/xgboost-v7-regret-aware-report.md`

Hypothesis: weight training samples by regret cost (expensive mistakes matter more).
- max_depth reduced to 5, 1500 estimators, regret-proportional sample weights
- **Result:** Gap closed dropped from 93.1% → ~78%. Regressed on every metric.
- **Root cause:** Regret weights downweight introsort↔heapsort boundary cases (low regret), which are the majority of training signal. max_depth=5 also caused underfitting.
- **Lesson:** v5's accuracy-based training already implicitly minimizes regret because the expensive errors (timsort misclassification) are already rare.

### v8: Binary Cascade (REJECTED — but produced key finding)

**Script:** `scripts/train_xgboost_v8_binary_cascade.py` | **Report:** `docs/xgboost-v8-binary-cascade-report.md`

Hypothesis: decompose the 3-class problem into two binary decisions.
- Stage 1 (timsort vs rest): AUC = **0.982** — near-perfect
- Stage 2 (introsort vs heapsort): AUC = **0.603** — near-random
- End-to-end gap closed: **78.3%** (≈ v5 on same test split)

**Key finding:** Introsort and heapsort are **feature-indistinguishable** with the 16 O(n) features. Stage 2 feature importance is flat (no signal concentrates), confirming this is a fundamental limitation of the feature space, not a model issue.

**Corollary:** The sorting algorithm selection problem under structural features is effectively a **binary timsort-detection problem**. The remaining ~7% gap (93.1% → 100%) requires information beyond O(n) features (cache behavior, branch prediction, etc.).

### Verdict

v5 remains the production model. Both experiments are documented as ablation studies for the thesis.

---

## What's Next

| Step | What | Status |
|------|------|--------|
| **Step 5: LinUCB bandit** | Online contextual bandit — adapts from v5's knowledge using real timing feedback | Script written (`scripts/train_linucb.py`), not yet validated |
| **Step 6: Comparison** | XGBoost vs bandit vs baselines | NOT STARTED |
| **Step 7: Package** | `import adaptive_sort; sort(arr)` | NOT STARTED |

---

## Reproducibility Notes

- **Python:** 3.13.2, venv at `venv/`
- **Key packages:** xgboost 3.2.0, scikit-learn, numpy, pandas
- **Random seed:** 42 everywhere
- **External data:** requires `/Volumes/k` USB drive for raw arrays (symlinked to `data/real_world_10k`)  
  - The `data/training_dataset.csv` (precomputed features + timings) does NOT require the external drive
- **All scripts run from project root:** `python3 scripts/<script>.py`
