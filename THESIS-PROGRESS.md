# Thesis Progress — Full Documentation

**Project:** Adaptive Sorting Algorithm Selection Using Offline + Online Machine Learning  
**Last Updated:** 2026-04-12  
**Author:** Ahmed  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Completed Work (Steps 1–3)](#3-completed-work)
4. [Model Evolution & Lessons Learned](#4-model-evolution--lessons-learned)
5. [Real-World Data Inventory](#5-real-world-data-inventory)
6. [Key Experimental Results](#6-key-experimental-results)
7. [Honest Assessment & Known Issues](#7-honest-assessment--known-issues)
8. [Remaining Work (Steps 4–8)](#8-remaining-work)
9. [Thesis Content Map](#9-thesis-content-map)
10. [File Reference](#10-file-reference)

---

## 1. Executive Summary

This thesis implements a **two-tier adaptive sorting algorithm selector**:
- **Tier 1 (Offline):** XGBoost model predicts the fastest sorting algorithm from 16 O(n) structural features extracted from the input array
- **Tier 2 (Online):** LinUCB contextual bandit adapts online using real timing feedback, improving accuracy on unseen data distributions

### Current Status
| Step | Description | Status |
|------|-------------|--------|
| 1 | Synthetic data generation + feature extraction (v2 features) | ✅ DONE |
| 1b | Pilot timing + VBS-SBS gap analysis | ✅ DONE |
| 2 | Full benchmark pipeline (720 samples × 3 algorithms) | ✅ DONE |
| 3 | XGBoost models (v1–v3: regression, classifier, log+pairwise) | ✅ DONE |
| 3+ | Real-world validation (v1–v4, 309 real-world arrays) | ✅ DONE |
| 3+ | Big data test (77 F1 telemetry .npy arrays) | ✅ DONE |
| 3++ | XGBoost v5: production model (1.18M arrays, 93% gap closed) | ✅ DONE |
| 3++ | XGBoost v6: honest source-aware checkpoint (71.2% test) | ✅ DONE |
| 3++ | XGBoost v7: regret-aware experiment (REJECTED) | ✅ DONE |
| 3++ | XGBoost v8: binary cascade experiment (REJECTED — key finding) | ✅ DONE |
| 4 | Baselines (random, always-SBS, decision tree, MLP) | ❌ NOT STARTED |
| 5 | LinUCB contextual bandit (MAIN CONTRIBUTION) | 🔄 Script written, not validated |
| 6 | Comparison: XGBoost vs bandit vs baselines | ❌ NOT STARTED |
| 7 | Extended real-world validation (scale to 1000+ arrays) | 🔄 PARTIALLY DONE |
| 8 | Package as Python library | ❌ NOT STARTED |

---

## 2. Architecture Overview

### Problem Formulation
- **Input:** 1D numeric array of arbitrary size
- **Output:** Which of 3 C-level sorting algorithms will sort it fastest
- **Constraint:** Feature extraction must be O(n) — cheaper than sorting itself

### Algorithm Portfolio (3 algorithms, all C-level via numpy)

| Algorithm | Implementation | Best When | Win Rate (synthetic) |
|-----------|---------------|-----------|---------------------|
| Introsort | `np.sort(kind='quicksort')` | Medium sizes, mixed structures | 33.3% |
| Heapsort | `np.sort(kind='heapsort')` | Random data, large arrays | 44.4% |
| Timsort | `np.sort(kind='stable')` | Pre-sorted/partially-sorted data | 22.2% |

**Dropped algorithms** (with empirical justification):
- Counting sort: 0 wins in 720 benchmarks (value range forces 30M-entry allocation)
- Radix sort: np.argsort per digit pass is O(n log n) per pass — 20× slower
- Python-loop sorts (insertion, shell, bubble): can't compete with C-level implementations

### 16 Structural Features (O(n), no sorting required)

| # | Feature | What It Captures |
|---|---------|-----------------|
| 1 | `length_norm` | Array size (normalized to n_max=2M) |
| 2 | `adj_sorted_ratio` | % of adjacent pairs in order |
| 3 | `duplicate_ratio` | Fraction of non-unique values |
| 4 | `dispersion_ratio` | Range / max possible range |
| 5 | `runs_ratio` | Number of sorted runs / n |
| 6 | `inversion_ratio` | Fraction of out-of-order pairs (merge-sort count) |
| 7 | `entropy_ratio` | Shannon entropy of binned histogram |
| 8 | `skewness_t` | Tanh-squished skewness |
| 9 | `kurtosis_excess_t` | Tanh-squished excess kurtosis |
| 10 | `longest_run_ratio` | Longest sorted run / n |
| 11 | `iqr_norm` | Interquartile range / range |
| 12 | `mad_norm` | Median absolute deviation / range |
| 13 | `top1_freq_ratio` | Most common value frequency / n |
| 14 | `top5_freq_ratio` | Top 5 values frequency / n |
| 15 | `outlier_ratio` | % values beyond 3 IQR from median |
| 16 | `mean_abs_diff_norm` | Mean |a[i]-a[i+1]| / range |

**Validation:** 214/214 tests passed. Zero NaN, zero Inf across all datasets. See `docs/feature-validation-report.md`.

### Two-Tier Selector Design

```
              ┌─ size < threshold? ─→ just use Timsort (no selection overhead)
[Raw Array] ──┤
              └─ size ≥ threshold? ─→ [Feature Extraction] → [Selector] → [Sort]
```

- **Tier 1 (Offline — XGBoost):** Trained once, frozen. Cold-start solution.
- **Tier 2 (Online — LinUCB Bandit):** Starts from XGBoost's knowledge, observes real timing feedback, adapts. **This is the thesis contribution.**

---

## 3. Completed Work

### Step 1: Synthetic Data Generation + Feature Extraction

**Script:** `scripts/generate_synthetic_dataset.py`, `scripts/extract_features.py`

- Generated 960 synthetic arrays across:
  - Sizes: 1K, 5K, 10K, 50K, 100K, 500K, 1M, 2M
  - Distributions: uniform, normal, lognormal, exponential
  - Structures: random, nearly_sorted, reverse_sorted, few_unique, many_duplicates, sorted_runs
- Extracted 16 structural features per array
- Stratified splits ensuring no leakage

**Output:** `data/synthetic/`, `data/features/`

### Step 1b: Pilot Timing + VBS-SBS Gap

**Scripts:** `scripts/pilot_timing.py`, `scripts/pilot_timing_v2.py`, `scripts/vbs_sbs_gap.py`

- Confirmed 3-way algorithm competition exists (not a single-winner landscape)
- VBS-SBS gap = **18.8%** on synthetic data → selector is worth building
- Timsort dominates sorted/reverse structures (10–17× faster than heapsort)
- Heapsort wins large random arrays
- Introsort competitive at medium sizes

### Step 2: Full Benchmark Pipeline

**Script:** `scripts/benchmark_algorithms.py`

- 720 samples × 4 algorithms (introsort, heapsort, timsort, counting_sort)
- Counting sort: 0 wins → dropped, leaving 3 algorithms
- Runtime: 312 seconds
- Created canonical dataset: `data/benchmark/all_samples.parquet`

**Data splits (by distribution for designed shift):**
| Split | N | Distributions | Purpose |
|-------|---|---------------|---------|
| Train | 216 | uniform + normal | Train XGBoost |
| Val | 72 | uniform + normal | Tune hyperparameters |
| Test A | 72 | uniform + normal | In-distribution evaluation |
| Test B | 360 | lognormal + exponential | **Distribution shift** — bandit evaluation |

### Step 3: XGBoost Models (3 Versions)

#### Version 1: Multi-Output Regressor

**Script:** `scripts/train_xgboost.py` (405 lines)  
**Architecture:** 3 independent XGBoost regressors, one per algorithm, predicting `time_algorithm`. Selection = `argmin(predicted_times)`.

**Results:**
| Split | N | Accuracy | Acceptable (≤5%) | Gap Recovery |
|-------|---|----------|-------------------|-------------|
| Train | 216 | 51.4% | 73.6% | 82.0% |
| Val | 72 | 43.1% | 69.4% | 63.8% |
| Test A | 72 | 44.4% | 68.1% | 77.0% |
| Test B | 360 | 52.5% | 85.0% | 58.6% |
| Real | 309 | 57.9% | 76.4% | −163.5% |

**Root Cause Analysis:** Regression predicts absolute time magnitude (dominated by array size), not relative ranking. Massive heapsort bias (predicted heapsort 73-81% of the time vs true 46-58%). R² was high (0.89-0.96) because size predicts time well, but the model couldn't distinguish *which* algorithm is fastest at a given size.

**Feature Importance (v1):** `length_norm` (22.4%), `mean_abs_diff_norm` (18.3%), `top1_freq_ratio` (17.8%) — dominated by scale features, not sortedness features.

**Key Lesson:** Regression ≠ ranking. Predicting absolute times is insufficient — the model needs to learn relative performance differences.

#### Version 2: Direct Classifier ⭐ (Current Best Deployable Model)

**Script:** `scripts/train_xgboost_classifier_v2.py` (144 lines)  
**Architecture:** Single XGBoost classifier, `multi:softprob`, directly predicts `best_algorithm` label from 16 features.

**Results:**
| Split | N | Accuracy |
|-------|---|----------|
| Train | 216 | 100% |
| Val | 72 | **66.7%** |
| Test A | 72 | **62.5%** |
| Test B | 360 | **61.7%** |
| Real | 309 | **60.2%** |

**Per-Algorithm Performance (Val):**
| Algorithm | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| Introsort | 0.44 | 0.32 | 0.37 |
| Heapsort | 0.61 | 0.72 | 0.66 |
| Timsort | 1.00 | 1.00 | 1.00 |

**Feature Importance (v2):** `runs_ratio` (17.3%), `longest_run_ratio` (12.2%), `mean_abs_diff_norm` (9.8%) — now sortedness features dominate, which is correct.

**Key Improvements Over v1:**
- +20pp accuracy across all splits
- No heapsort bias — model now differentiates all 3 algorithms
- Timsort perfectly classified (sorted data = high runs_ratio → timsort)
- Introsort still hard to distinguish from heapsort (recall 32%)

**Why v2 is the working model:** Uses only 16 structural features (no timing data). Can be deployed at inference time without pre-sorting. Honest accuracy.

#### Version 3: Log + Pairwise Classifier (Proof of Concept — NOT Deployable)

**Script:** `scripts/train_xgboost_v3_logpairwise.py` (159 lines)  
**Architecture:** Same as v2, but uses 22 features = 16 structural + 3 log(time) + 3 pairwise time differences.

**Results:**
| Split | N | Accuracy |
|-------|---|----------|
| Train | 216 | 100% |
| Val | 72 | **98.6%** |
| Test A | 72 | **100%** |
| Test B | 360 | **96.9%** |
| Real | 309 | **94.5%** |

**Feature Importance (v3):** `diff_heapsort_timsort` (24.1%), `diff_introsort_heapsort` (16.6%), `diff_introsort_timsort` (16.4%) — top 3 features are ALL timing-derived.

**⚠️ DATA LEAKAGE:** The 6 timing-derived features require sorting the array with all 3 algorithms first. This defeats the purpose of the selector — you can't predict the fastest sort if you've already sorted. The near-perfect accuracy is real within the training setup, but the model is unusable in production.

**Why v3 matters for the thesis:** It proves that *pairwise timing differences* contain the signal needed for perfect selection. This validates the LinUCB bandit approach — the bandit learns these relative timing signals **online through actual observations**, without pre-sorting everything. v3 is the *ceiling* the bandit aims toward; v2 is the *floor* it starts from.

### Real-World Validation (v1–v4)

#### v1: F1 Fastest-Lap Telemetry
- **Script:** `scripts/test_real_data.py`
- 35 arrays from FastF1, sizes ~700
- timsort won 77.1%, VBS-SBS gap 5.1%
- Very small arrays — timsort dominance expected

#### v2: F1 Full-Race Telemetry
- **Script:** `scripts/test_real_data_v2.py`
- 108 arrays, sizes 34K–1.13M
- heapsort 42.6%, introsort 37.0%, timsort 20.4%
- VBS-SBS gap 3.2% — genuine 3-way competition

#### v3: Financial + Seismic Data
- **Script:** `scripts/test_real_data_v3.py`
- APIs: yfinance (20 stocks + 9 crypto), USGS earthquakes
- 149 arrays, sizes 2K–309K
- heapsort 64.4%, VBS-SBS gap 1.6%
- Low gap because homogeneous domains → one algo dominates per domain

#### v4: Cross-Domain Combined
- **Script:** `scripts/test_real_data_v4.py`
- APIs: Open-Meteo weather (5 cities), NASA NEO (JPL), USGS extended
- 1,039 total arrays (720 synthetic + 309 real)
- Combined gap 17.5%, real-only gap 12.0%
- **Honest reading:** Every truly-real domain has gap under 3.1%
- Per-array margins large: 97.3% have >10% margin, 70.3% have >100%

#### Bigtest: F1 Large-Scale .npy Arrays
- **Script:** `scripts/test_xgboost_v3_on_bigdata.py`
- **Script:** `scripts/fetch_f1_bigdata.py` (data downloader)
- 175 .npy files downloaded via FastF1 (2020–2024 seasons)
- 77 arrays tested (≥10K elements)
- Sizes: 11K–48K
- **v3 accuracy: 89.6%, mean regret: 6.4%**
- Prediction distribution: timsort 38, heapsort 23, introsort 16
- Actual distribution: timsort 46, heapsort 18, introsort 13

---

## 4. Model Evolution & Lessons Learned

### Lesson 1: Regression ≠ Ranking (v1 → v2)

The single most important lesson from Step 3. Training separate regressors to predict absolute times and then picking argmin does NOT produce good selections, even when R² is high. The model learned "bigger array = longer time" but couldn't distinguish "is heapsort 2% faster or 2% slower than introsort on this specific array."

**Thesis content:** This is a publishable negative result. Most algorithm selection papers use regression (AS benchmarks like ASlib default to it). Show it fails and explain why.

### Lesson 2: Classification Recovers Sortedness Signal (v2)

Switching to direct classification forced the model to learn decision boundaries between algorithms rather than absolute magnitudes. Feature importance shifted from scale features (`length_norm`) to sortedness features (`runs_ratio`, `longest_run_ratio`), which is what actually distinguishes algorithm performance.

**Thesis content:** Feature importance shift from v1 to v2 is a concrete, visualizable result. Make a figure.

### Lesson 3: Pairwise Timing Is the Missing Signal (v3)

v3 proves the information-theoretic point: if you could observe pairwise time differences, selection becomes trivial (95%+). But you can't observe them without running the algorithms. This is exactly the **explore-exploit tradeoff** the bandit is designed to solve.

**Thesis content:** v3 establishes the performance ceiling and motivates the bandit. The gap between v2 (62%) and v3 (95%) is the value the bandit can unlock.

### Lesson 4: VBS-SBS Gap Is Misleading at Domain Level

Within a single homogeneous domain, one algorithm tends to dominate, making the aggregate gap small (1-3%). But per-array margins remain large (>100% for 70% of arrays). The thesis should frame contribution around **per-array prediction accuracy**, not aggregate gap inflation.

**Thesis content:** Explain why per-array accuracy is the fair metric. Cite that SATzilla and similar work use per-instance accuracy, not just portfolio-level gap.

### Lesson 5: Synthetic Training Data Is a Weakness

Training on 216 synthetic arrays and testing on real data is a validity concern. The 309 real-world arrays already have features + timings. Future work: retrain on real data or a mix.

**UPDATE (2026-04-12):** This was fixed — v5 retrains on 1.18M real-world arrays. See section 4b below.

### Lesson 6: Regret-Aware Training Is Unnecessary (v7)

Weighting samples by their misclassification cost sounds theoretically appealing. But v7 showed it hurts: regret weights suppress boundary cases (introsort↔heapsort, low regret) where the model needs training signal most. v5's inverse-frequency weights already minimize regret implicitly because the expensive errors (timsort → wrong algorithm, costing 36-50 μs) are already rare at 94.5% recall. See `docs/xgboost-v7-regret-aware-report.md`.

### Lesson 7: Introsort ≈ Heapsort Under Structural Features (v8)

The most important structural finding: v8's binary cascade proved that introsort and heapsort are **feature-indistinguishable** with the 16 O(n) features (Stage 2 AUC = 0.603, barely above coin flip). This means:
- The problem is effectively **binary** (timsort vs not-timsort) under structural features
- v5's remaining 7% gap (93% → 100%) requires hardware-level information (cache, branch prediction)
- No model architecture changes can fix this — it's a feature-space limitation
- See `docs/xgboost-v8-binary-cascade-report.md`

---

## 4b. XGBoost v5–v8: Real-World Production & Ablation Studies

### v5: Production Model (Current Best)

Retrained on **1,188,265** real-world arrays from 5 domains.

**Balance strategy:** noise filter (margin ≥ 5% OR size ≥ 2K) → undersample (3× minority cap) → inverse-frequency weights.

| Metric | Value |
|--------|-------|
| Test accuracy | 76.1% |
| Gap closed (full 1.18M) | **93.14%** |
| Perfect picks | **89.64%** |
| Mean regret | 0.23 μs |
| P99 regret | 6.12 μs |

Per-class recall: introsort 52.1%, heapsort 63.8%, timsort **94.5%**.

**Why 76% accuracy = 93% gap closed:** Introsort↔heapsort confusion costs ~1.5 μs per error. Timsort misclassification costs 36-50 μs but happens rarely. The model nails the expensive decision.

### v6: Honest Source-Aware Checkpoint

GroupShuffleSplit by source_id → test accuracy 71.2%, gap closed 45.6%. Stricter generalization estimate.

### v7: Regret-Aware Training (REJECTED)

max_depth=5, 1500 estimators, regret-proportional weights. All metrics worse. Regret weights suppress boundary training signal; shallow depth underfits. See Lesson 6 above.

### v8: Binary Cascade (REJECTED — Key Finding)

Stage 1 (timsort vs rest): AUC=0.982. Stage 2 (introsort vs heapsort): AUC=0.603. Cascade gap closed ≈ v5. See Lesson 7 above.

### Verdict: v5 Is Near-Ceiling

Both v7 and v8 confirm no further gains possible with current features + algorithms. v5 remains frozen production model.

---

## 5. Real-World Data Inventory

### Currently Available (with features + timings)

| Source | API/Method | Arrays | Size Range | Domain | Status |
|--------|-----------|--------|------------|--------|--------|
| Stock market | yfinance | 124 | 2K–6K | Finance | In v4 parquet ✅ |
| F1 telemetry | FastF1 | 108 | 34K–1.13M | Motorsport | In v4 parquet ✅ |
| Weather | Open-Meteo | 28 | 10K–105K | Climate | In v4 parquet ✅ |
| Crypto | yfinance | 18 | ~5K | Finance | In v4 parquet ✅ |
| NASA NEO | JPL SBDB | 18 | ~18K | Astronomy | In v4 parquet ✅ |
| Earthquakes | USGS FDSNWS | 13 | 2K–309K | Seismology | In v4 parquet ✅ |
| **Total real** | | **309** | **2K–1.13M** | **7 domains** | |

### Additional (features not yet extracted)

| Source | Arrays | Size Range | Notes |
|--------|--------|------------|-------|
| F1 bigtest .npy | 77 tested (175 files) | 11K–48K | Has timings, features extractable |

### Available Data Sources for Expansion

All APIs are **free, no API key required**:
- **yfinance:** Can fetch 100+ stocks × 20 years × 6 columns = potentially 600+ arrays
- **FastF1:** 5 seasons × 20+ races × 7 sessions × 20 drivers × 5 channels = potentially 1000s of arrays
- **Open-Meteo:** 50+ cities × 4 variables × 75 years = potentially 200+ arrays
- **NASA/JPL:** Multiple asteroid families, more orbital parameters
- **USGS:** Different earthquake regions, time windows, depth/magnitude arrays

**Expansion target:** 1,000+ real-world arrays for credible training

---

## 6. Key Experimental Results

### VBS-SBS Gap by Dataset

| Dataset | Arrays | Gap | Interpretation |
|---------|--------|-----|----------------|
| Synthetic benchmark | 720 | 18.8% | Strong 3-way competition |
| F1 telemetry v2 | 108 | 3.2% | 3-way competition at scale |
| Financial + Seismic | 149 | 1.6% | Homogeneous → one algo dominates |
| Cross-domain combined | 1,039 | 17.5% | Inflated by synthetic majority |
| Real data only | 309 | ~3% | Honest aggregate gap |

### XGBoost v2 Classifier Accuracy by Domain (Real Data)

| Domain | Arrays | Best Algo | Model Accuracy |
|--------|--------|-----------|---------------|
| Stock | 124 | heapsort dominant | 63.7% |
| F1 telemetry | 108 | mixed 3-way | 40.7% |
| Weather | 28 | heapsort dominant | 64.3% |
| Crypto | 18 | heapsort dominant | 83.3% |
| NASA NEO | 18 | heapsort dominant | 88.9% |
| Earthquake | 13 | mixed | 53.8% |

### Feature Importance Comparison

| Rank | v1 (Regression) | v2 (Classifier) | v3 (Log+Pairwise) |
|------|----------------|-----------------|-------------------|
| 1 | length_norm (22%) | **runs_ratio (17%)** | diff_heap_tim (24%) |
| 2 | mean_abs_diff (18%) | **longest_run (12%)** | diff_intro_heap (17%) |
| 3 | top1_freq (18%) | mean_abs_diff (10%) | diff_intro_tim (16%) |
| 4 | top5_freq (13%) | length_norm (8%) | outlier_ratio (5%) |
| 5 | duplicate (10%) | adj_sorted (7%) | top5_freq (4%) |

**Key insight:** v1 learned size → time. v2 learned structure → algorithm. v3 learned timing → answer.

---

## 7. Honest Assessment & Known Issues

### Strengths
1. **Feature engineering is solid** — 16 O(n) features, validated, defensible
2. **Benchmark pipeline is reproducible** — seeded, documented, configurable
3. **Real-world validation is diverse** — 7 domains, 309 arrays, free APIs
4. **Negative results are documented** — v1 regression failure, VBS-SBS gap honesty
5. **Model evolution tells a story** — v1 → v2 → v3 is a natural thesis narrative

### Weaknesses
1. **Training set is small synthetic** — 216 samples, only uniform+normal distributions
2. **v2 accuracy is modest** — 62-67% on test, 60% on real data
3. **Introsort recall is poor** — model confuses introsort with heapsort (32% recall)
4. **VBS-SBS gap within single domains is small** — 1-3%, hard to claim large savings
5. **No baselines yet** — can't contextualise v2's 62% without random/always-SBS/DT/MLP comparisons
6. **Bandit not implemented** — the main thesis contribution doesn't exist yet

### Known Issues
- `inversion_ratio` is the most expensive feature — if importance stays low, should drop it
- `outlier_ratio` has very low variance in synthetic data
- v3 has data leakage (timing features) — not deployable, only proof-of-concept
- PROJECT.md build steps table was outdated (showed Step 3 as NOT STARTED)

### Planned Fixes
1. **Retrain on real data** — use 309+ real-world arrays instead of 216 synthetic
2. **Expand to 1000+ arrays** — pull more from existing free APIs
3. **Implement baselines** (Step 4) — random, always-SBS, decision tree, MLP
4. **Implement LinUCB** (Step 5) — the main thesis contribution
5. **Package as library** (Step 8) — `import adaptive_sort; sort(arr)`

---

## 8. Remaining Work

### Step 4: Baselines (Priority: HIGH)

Implement and evaluate:
- **Random selector** — picks uniformly at random (expected 33.3%)
- **Always-SBS** — always picks heapsort (the SBS on synthetic data)
- **Decision tree** — simple interpretable baseline (sklearn)
- **MLP** — 2-layer neural network baseline (sklearn or PyTorch)

These establish context for v2's 62-67% accuracy.

### Step 5: LinUCB Contextual Bandit (Priority: CRITICAL — Main Thesis Contribution)

- Initialize from v2 classifier's knowledge
- Run on Test B (lognormal + exponential — unseen distributions)
- Observe real timing feedback after each selection
- Update model online
- **Key output:** Regret curve showing convergence from v2's 62% toward v3's 95%
- Also run on real-world arrays to show cross-domain adaptation

### Step 6: Comparison (Priority: HIGH)

- v2 XGBoost vs LinUCB bandit vs all baselines
- Metrics: accuracy, acceptable ≤5%, regret, VBS-SBS gap recovery
- Per-domain and per-structure breakdowns
- Statistical significance tests

### Step 7: Extended Real-World Validation (Priority: MEDIUM)

- Scale real data to 1000+ arrays from existing free APIs
- Retrain v2 on real data (not just synthetic)
- Re-evaluate all models on expanded dataset

### Step 8: Python Library (Priority: LOW — Nice-to-Have)

```python
from adaptive_sort import sort
sorted_arr = sort(my_array)  # automatically selects best algorithm
```

---

## 9. Thesis Content Map

### Chapter: Feature Engineering
- 16 features defined and justified (docs/feature-definitions.md)
- O(n) complexity analysis per feature
- Validation results: 214/214 tests, 0 NaN/Inf (docs/feature-validation-report.md)
- Defense of design choices (docs/feature-extraction-defense.md)

### Chapter: Algorithm Portfolio
- Why these 3 algorithms (C-level requirement)
- Why counting/radix/bucket were dropped (empirical evidence)
- VBS-SBS gap analysis showing selection is worthwhile (docs/vbs-sbs-gap-analysis.md)
- Win rate analysis by size/distribution/structure

### Chapter: Offline Model (XGBoost)
- **v1 regression → failure analysis** — publishable negative result
- **v2 classifier → honest baseline** — 62-67% accuracy
- **v3 log+pairwise → ceiling proof** — motivates the bandit
- Feature importance shift across versions (makes a great figure)
- Confusion matrices, per-structure breakdowns

### Chapter: Online Model (LinUCB Bandit) — TO BE IMPLEMENTED
- Theoretical motivation (v2 floor → v3 ceiling)
- Algorithm description
- Regret curves on Test B
- Comparison with offline model

### Chapter: Real-World Validation
- 7 domains, 309+ arrays
- Per-domain analysis (docs/real-world-f1-report.md through v4)
- Honest VBS-SBS gap discussion (docs/vbs-sbs-gap-analysis.md)
- Per-array margin analysis

### Chapter: Evaluation & Comparison
- All models vs all baselines
- Metrics suite: accuracy, regret, gap recovery, acceptable selections
- Statistical analysis

---

## 10. File Reference

### Scripts (execution order)
| Script | Step | Purpose |
|--------|------|---------|
| `scripts/generate_synthetic_dataset.py` | 1 | Generate synthetic arrays |
| `scripts/extract_features.py` | 1 | Extract 16 features |
| `scripts/feature_extraction.py` | 1 | Core feature extraction module (imported by all) |
| `scripts/assess_dataset_quality.py` | 1 | Validate data quality |
| `scripts/pilot_timing.py` | 1b | Initial timing pilot |
| `scripts/pilot_timing_v2.py` | 1b | Refined timing pilot |
| `scripts/vbs_sbs_gap.py` | 1b | VBS-SBS gap analysis |
| `scripts/benchmark_algorithms.py` | 2 | Full 720-sample timing pipeline |
| `scripts/validate_features.py` | 2 | Feature validation |
| `scripts/train_xgboost.py` | 3 | v1: regression model (405 lines) |
| `scripts/train_xgboost_classifier_v2.py` | 3 | v2: classifier model (144 lines) |
| `scripts/train_xgboost_v3_logpairwise.py` | 3 | v3: log+pairwise model (159 lines) |
| `scripts/test_real_data.py` | 3+ | v1 real-world test (F1 fastest lap) |
| `scripts/test_real_data_v2.py` | 3+ | v2 real-world test (F1 full race) |
| `scripts/test_real_data_v3.py` | 3+ | v3 real-world test (finance + seismic) |
| `scripts/test_real_data_v4.py` | 3+ | v4 cross-domain combined test |
| `scripts/fetch_f1_bigdata.py` | 3+ | Download F1 .npy arrays |
| `scripts/test_xgboost_v3_on_bigdata.py` | 3+ | Evaluate v3 on F1 big data |
| `scripts/_check_step3.py` | 3 | Data layout sanity check |
| `scripts/_audit_real_data.py` | — | Audit all available real data |

### Models
| Path | Version | Type |
|------|---------|------|
| `models/xgboost_v1/xgb_introsort.json` | v1 | Regressor (introsort) |
| `models/xgboost_v1/xgb_heapsort.json` | v1 | Regressor (heapsort) |
| `models/xgboost_v1/xgb_timsort.json` | v1 | Regressor (timsort) |
| `models/xgboost_v1/metadata.json` | v1 | Training metadata |
| `models/xgboost_classifier_v2/xgb_classifier_v2.json` | v2 | Classifier |
| `models/xgboost_v3_logpairwise/xgb_classifier_v3_logpairwise.json` | v3 | Classifier (leaky) |
| `models/xgboost_v5/xgb_v5.json` | v5 | **Production classifier** |
| `models/xgboost_v8/stage1_timsort_vs_rest.json` | v8 | Binary Stage 1 |
| `models/xgboost_v8/stage2_introsort_vs_heapsort.json` | v8 | Binary Stage 2 |

### Results
| Path | Content |
|------|---------|
| `results/xgboost_v1/evaluation_results.json` | v1 full evaluation (597 lines) |
| `results/xgboost_classifier_v2/evaluation_results.json` | v2 full evaluation (398 lines) |
| `results/xgboost_v3_logpairwise/evaluation_results.json` | v3 full evaluation (428 lines) |
| `results/xgboost_v5/evaluation_results.json` | v5 production evaluation |
| `results/xgboost_v5/regret_analysis.json` | v5 VBS/SBS/model regret (1.18M rows) |
| `results/xgboost_v7/evaluation_results.json` | v7 regret-aware (rejected) |
| `results/xgboost_v8/evaluation_results.json` | v8 binary cascade (rejected) |
| `results/xgboost_v8/figures/*.png` | v8 9 analysis figures |
| `results/*/predictions_*.csv` | Per-sample predictions per split |
| `data/real_world_bigtest/results_xgboost_v3.csv` | Bigtest 77-array results |

### Data
| Path | Content |
|------|---------|
| `data/benchmark/all_samples.parquet` | 720 samples with features + timings |
| `data/benchmark/train.parquet` | 216 samples (uniform + normal) |
| `data/benchmark/val.parquet` | 72 samples |
| `data/benchmark/test_A.parquet` | 72 samples (in-distribution) |
| `data/benchmark/test_B.parquet` | 360 samples (lognormal + exponential) |
| `data/real_world_v4/real_world_v4_combined.parquet` | 1,039 arrays (720 synthetic + 309 real) |
| `data/real_world_bigtest/raw/*.npy` | 175 F1 telemetry arrays |

### Documentation
| Path | Content |
|------|---------|
| `PROJECT.md` | Master project specification |
| `THESIS-PROGRESS.md` | This document |
| `docs/feature-definitions.md` | Complete 16-feature reference |
| `docs/feature-validation-report.md` | 214/214 feature tests |
| `docs/feature-extraction-defense.md` | Feature extraction audit |
| `docs/real-world-f1-report.md` | v1 F1 report |
| `docs/real-world-f1-report-v2.md` | v2 F1 report |
| `docs/real-world-v3-report.md` | v3 finance+seismic report |
| `docs/real-world-v4-report.md` | v4 combined report (honest version) |
| `docs/real-world-v4-report-original.md` | v4 original (over-optimistic) |
| `docs/vbs-sbs-gap-analysis.md` | VBS-SBS gap explanation |
| `docs/xgboost-v1-report.md` | v1 root cause analysis |
| `docs/checkpoint-xgboost-v5-baseline.md` | v5 checkpoint (updated with v7/v8) |
| `docs/xgboost-v7-regret-aware-report.md` | v7 negative result write-up |
| `docs/xgboost-v8-binary-cascade-report.md` | v8 cascade + key finding write-up |
