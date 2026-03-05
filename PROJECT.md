# Adaptive Sorting Algorithm Selection — Master Thesis

## What This Project Does

Given a 1D numeric array at runtime, predict which sorting algorithm will be fastest — without running all of them. Extract cheap statistical features from the array, feed them to a selector model, get the best algorithm, sort.

## Problem Formulation

- Input: feature vector x ∈ R^16 (array characteristics computed in O(n))
- Output: a* = argmin_a T_a(x) — the algorithm with the lowest execution time for this array
- This is a single-step, stateless decision. No sequential dependency.
- Model approach: REGRESSION (predict time per algorithm, pick min) — not classification
  - Avoids hard-label class imbalance when some algorithms rarely win
  - Handles near-ties gracefully (if two algorithms are within 5%, either is fine)
  - Enables direct regret measurement: T_predicted - T_oracle

## Architecture Decided

Two-layer system, not two competing models:

```
              ┌─ size < threshold? ─→ just use Timsort (no selection overhead)
[Raw Array] ──┤
              └─ size ≥ threshold? ─→ [Feature Extraction] → [Selector] → [Sort]
```

The size threshold is a RESULT, not a parameter — determined by measuring when feature extraction cost exceeds selection savings. Reported in thesis.

### Layer 1: Offline Selector (XGBoost multi-output regression)
- Trained once on labeled synthetic data (features → predicted time per algorithm)
- Picks algorithm with lowest predicted time
- Frozen at deployment — never updates
- Purpose: strong static baseline + cold-start solution

### Layer 2: Online Selector (Contextual Bandit — LinUCB)
- Starts from XGBoost's knowledge
- Every time it sorts a real array, it observes real timing feedback
- Updates its model from that feedback — learns while deployed
- Purpose: adapts to unseen workload distributions without retraining
- This is the main thesis contribution

### Why Not DQN / Double DQN
- DQN is for sequential decision problems (multi-step MDPs)
- This problem is one-step: observe features → pick algorithm → done
- In a one-step MDP: next_state = state, done = True, always
- Bellman equation collapses to Q(s,a) = r — all RL machinery is dead weight
- Contextual bandits are the correct RL formulation for one-step context-dependent decisions
- The old notebooks (model-1.ipynb, mdoel-2.ipynb) used DQN — moved to old_models/ folder

## Sorting Algorithm Portfolio (3 algorithms, ALL C-level)

CRITICAL RULE: all implementations must be at C-speed (numpy ops, no Python loops in sort code). Mixing Python-loop sorts with C-level sorts makes timing comparisons meaningless — the model would learn language overhead, not algorithm characteristics.

| Algorithm  | Implementation                                        | Win rate (Step 2) |
|-----------|-------------------------------------------------------|-------------------|
| Introsort  | np.sort(kind='quicksort') — quicksort+heapsort hybrid | 33.3%             |
| Heapsort   | np.sort(kind='heapsort')                              | 44.4%             |
| Timsort    | np.sort(kind='stable') — adaptive, exploits runs      | 22.2%             |

Dropped after empirical evaluation:
- Counting sort — 0 wins in 720 benchmarks. Value range 0–30M forces 30M-entry allocation; always slower than comparison sorts at n≤2M
- Radix sort (LSD) — np.argsort per digit pass is O(n log n) per pass; 20x slower than introsort. Not a true radix sort without C-level implementation
- Bucket sort — never implemented; same problem as radix (per-bucket np.sort adds overhead)
- Bubble sort — O(n²), never competitive
- Insertion sort — can't implement at C-level without Cython
- Shell sort — same Cython problem
- Pure merge sort — numpy's 'mergesort' is actually Timsort since numpy 1.17

## Feature Set (v2 — 16 features)

Computed in O(n), no sorting required:

v1 core (5): length_norm, adj_sorted_ratio, duplicate_ratio, dispersion_ratio, runs_ratio

v2 additions (11): inversion_ratio, entropy_ratio, skewness_t, kurtosis_excess_t, longest_run_ratio, iqr_norm, mad_norm, top1_freq_ratio, top5_freq_ratio, outlier_ratio, mean_abs_diff_norm

All features bounded and validated — no NaN, no Inf, no duplicate sample_ids.

NOTE: inversion_ratio is the most expensive feature (merge-sort-based count, Python recursion). If feature importance analysis (Step 3) shows it's not useful, drop it — cuts feature extraction cost ~10x.

NOTE: outlier_ratio has very low variance in current data (max 0.073, std 0.012). May be dead weight. Keep for now, evaluate after Step 3.

## Synthetic Dataset

- 960 samples total (480 train / 240 val / 240 test) — STARTING POINT, will scale up
- 10 sizes: 1K to 50K
- 4 distributions: uniform, normal, lognormal, exponential
- 6 structures: random, nearly_sorted, reverse_sorted, few_unique, many_duplicates, runs
- 4 repeats per combination
- Stratified splits — no leakage, balanced across all factors
- Data lives in data/synthetic/raw/*.npy, splits in data/synthetic/splits/

## Bandit Evaluation: Designed Distribution Shift

The bandit's value only shows when XGBoost fails on unseen distributions. This must be designed into the experiment from day one, not retrofitted.

| Data Split    | Distributions         | XGBoost trained on it? | Bandit adapts to it? | Purpose                    |
|--------------|----------------------|----------------------|--------------------|-----------------------------|
| Train         | uniform, normal       | Yes                  | No                 | Train offline model          |
| Test A (in-dist) | uniform, normal    | No (held out)        | No                 | Evaluate XGBoost on known   |
| Test B (shift) | lognormal, exponential | NO                  | Yes (online)       | Evaluate bandit on unseen   |

XGBoost performs well on Test A, poorly on Test B.
Bandit starts poor on Test B, improves with feedback.
The regret curve on Test B is the bandit's thesis contribution.

NOTE: this requires restructuring data splits by distribution type, not just by sample_id. Plan into timing pipeline.

## Build Steps (ordered)

| Step | What                                                        | Status      |
|------|-------------------------------------------------------------|-------------|
| 1    | Synthetic data generation + feature extraction (v2)         | DONE        |
| 1b   | Pilot timing — verify algorithm landscape + VBS-SBS gap     | DONE        |
| 2-old| Timing pipeline — 720 synthetic samples × 3 algorithms      | DONE (superseded) |
| 3-old| XGBoost models — v1 regressor, v2 classifier, v3 log+pairwise | DONE (superseded) |
| 3+   | Real-world validation — v1–v4 (309 arrays, 7 domains) + bigtest (77 arrays) | DONE |
| **1-new** | **Real-world data collection — 1.18M arrays, 5 domains**   | **DONE** |
| **2-new** | **Training dataset — features + timing for all 1.18M arrays** | **DONE** |
| **3-new** | **XGBoost v5 — balanced classifier on real-world data**     | **DONE** |
| **4**     | **VBS/SBS regret analysis — model value quantified**        | **DONE** |
| 5    | LinUCB contextual bandit — online loop, regret curve         | NOT STARTED |
| 6    | Comparison: XGBoost vs Bandit vs baselines                   | NOT STARTED |
| 7    | Package as Python library (import adaptive_sort; sort(arr))  | NOT STARTED |

## Thesis Contributions (5 defensible points)

1. Feature engineering — 16 cheap O(n) features that characterize sortability
2. Empirical benchmark — 3 C-level algorithms × 720 arrays, performance map showing when each wins
3. Offline selector — XGBoost regression, accuracy + regret analysis, feature importance
4. Online selector (NOVEL) — contextual bandit adapts to distribution shift without retraining
5. System — deployable Python library with two-tier threshold + adaptive selection

## Metrics

- Top-1 accuracy: did argmin of predicted times match argmin of actual times?
- Top-2 accuracy: was the oracle's pick in the model's top 2?
- Slowdown ratio: time_selected / time_oracle (1.0 = perfect)
- Regret: (time_selected - time_oracle) / time_oracle
- Regret curve (bandit): cumulative regret over time — shows convergence speed
- Feature extraction overhead: time to extract features vs time saved by selection

## Key Prior Work to Cite

- Rice (1976) — Algorithm Selection Problem (coined the concept)
- AutoFolio (Lindauer et al., 2015) — algorithm selection framework
- SATzilla (Xu et al., 2008-2012) — feature-based solver selection
- Learned Sort (Kristo et al., 2020) — ML-enhanced sorting
- Li & Mao (2009) — sorting selection via decision trees

## Stress-Tested Decisions (why we chose what we chose)

### Why regression not classification
- Classification: "which algorithm wins?" — class imbalance when some algorithms rarely win
- Regression: "how long does each take?" — pick min. Richer signal, no imbalance, near-ties handled.

### Why all C-level implementations
- Mixing Python-loop sorts with C sorts means the model learns language overhead, not algorithm performance
- numpy.sort(kind=...) gives us introsort, heapsort, Timsort at C level
- Radix and counting sort implemented via numpy vectorized ops (no Python loops)
- If we can't implement it at C-level without Cython, we don't include it

### Why two-tier threshold
- For small arrays, feature extraction costs more than just sorting
- Below threshold: skip selection, use Timsort
- Threshold value is measured and reported as a thesis result

### Why designed distribution shift
- If XGBoost gets 95% on i.i.d. test data, bandit has no room to improve
- Deliberately hold out distributions from training → bandit adapts to them online
- This is the experiment that proves the bandit's value

## Project Structure

```
scripts/
  generate_synthetic_dataset.py   — Step 1: synthetic data generation (DONE)
  extract_features.py             — Step 1: feature extraction (DONE)
  feature_extraction.py           — 16 features, single source of truth
  assess_dataset_quality.py       — Step 1: quality validation (DONE)
  pilot_timing.py                 — Step 1b: pilot timing v1 (DONE)
  pilot_timing_v2.py              — Step 1b: pilot timing v2 (DONE)
  vbs_sbs_gap.py                  — Step 1b: VBS vs SBS gap analysis (DONE)
  benchmark_algorithms.py         — Step 2-old: synthetic timing pipeline (DONE)
  fetch_f1_10k.py                 — Real-world: F1 telemetry fetcher (487K arrays)
  fetch_stock.py                  — Real-world: stock market fetcher (100K arrays)
  fetch_weather.py                — Real-world: weather data fetcher (3.2K arrays)
  fetch_crypto.py                 — Real-world: crypto market fetcher (100K arrays)
  fetch_earthquake.py             — Real-world: earthquake data fetcher (100K arrays)
  transform_f1_sample.py          — Structural transforms for F1 data
  build_training_dataset.py       — Step 2-new: features + timing for 1.18M arrays
  train_xgboost_v5.py             — Step 3-new: balanced classifier
  regret_analysis.py              — Step 4: VBS/SBS/model regret analysis
  _dataset_audit.py               — Dataset diversity & bias audit
  _verify_migration.py            — Post-migration integrity check

data/
  training_dataset.csv            — 1,188,265 rows × 22 columns (THE training data)
  real_world_10k/                 — SYMLINK → /Volumes/k/thesis_data/real_world_10k
    index.csv                     — 1,188,265 row index
    raw/                          — 1,188,265 CSV array files (~12 GB)
  synthetic/                      — Original 720 synthetic arrays (superseded)
  benchmark/                      — Original synthetic timing results (superseded)

models/
  xgboost_v5/
    xgb_v5.json                   — Current best model (trained on real-world data)
  xgboost_classifier_v2/          — v2 classifier (trained on synthetic, superseded)
  xgboost_v3_logpairwise/         — v3 log+pairwise (trained on synthetic, superseded)
  xgboost_v1/                     — v1 regressors (trained on synthetic, superseded)

results/
  xgboost_v5/
    evaluation_results.json       — v5 training metrics
    regret_analysis.json          — VBS/SBS/model regret (Step 4)
    predictions_test.csv          — Test split predictions
  xgboost_classifier_v2/          — v2 results (superseded)
  xgboost_v3_logpairwise/         — v3 results (superseded)
  xgboost_v1/                     — v1 results (superseded)

docs/
  feature-definitions.md          — Complete 16-feature reference
  feature-validation-report.md    — 214/214 feature tests passed
  feature-extraction-defense.md   — Feature extraction integrity audit
```

## Step 2 Results (Benchmark)

Run: 720 samples × 4 algorithms (introsort, heapsort, timsort, counting_sort)
Sizes: 10K, 50K, 100K, 500K, 1M, 2M | 4 distributions | 6 structures | 5 repeats
Runtime: 312s (5.2 min)

### Win Counts
| Algorithm      | Wins | Win Rate |
|---------------|------|----------|
| heapsort       | 320  | 44.4%    |
| introsort      | 240  | 33.3%    |
| timsort        | 160  | 22.2%    |
| counting_sort  | 0    | 0.0%     |

### VBS-SBS Gap
- VBS-SBS gap = 18.8% — "always heapsort" wastes 18.8% of total sorting time
- Consistent with pilot analysis (20.8% on 100K–10M only)
- Timsort dominates sorted/reverse_sorted (10–17× faster than heapsort)
- Introsort wins at medium sizes and some structures
- Heapsort wins random/large but loses catastrophically on pre-sorted data

### Key Findings
- counting_sort: DROPPED — 0 wins. Value range 0–30M makes bincount allocation too expensive at n≤2M
- 3 viable algorithms remain: introsort, heapsort, timsort
- Split balance good: win ratios consistent across train/val/test_A/test_B
- No NaN/Inf in features or timings — dataset is clean

### Dataset
- data/benchmark/all_samples.parquet — 720 rows, each with 16 features + 4 timing columns + metadata
- Splits by distribution: train (216, uniform+normal) / val (72) / test_A (72) / test_B (360, lognormal+exponential)

## Real-World Validation (v1–v4)

### v1: F1 Fastest Lap (35 arrays, n~700)
- timsort wins 77.1%, VBS-SBS gap 5.1%
- Small arrays — timsort dominance expected

### v2: F1 Full Race (108 arrays, 34K–1.13M)
- heapsort 42.6%, introsort 37.0%, timsort 20.4%, VBS-SBS gap 3.2%
- Genuine 3-way competition at scale

### v3: Financial + Seismic (149 arrays, 2K–309K)
- heapsort 64.4%, VBS-SBS gap 1.6%
- Low gap explained: homogeneous domains → one algo dominates within each

### v4: Cross-Domain Combined (1,039 arrays, 2K–2M)
- **Combines ALL previous data + new real-world data**
- New data: 5 cities weather (Open-Meteo), 23K NASA asteroids (JPL), 100K earthquakes (USGS), 10 large-scale generated arrays at 2M
- Headline: 17.5% combined gap, 12.0% "real-only" gap
- **HONEST READING: The 17.5% is dominated by 720 synthetic arrays (69% of data). The 12% includes 10 generated arrays. Every truly-real domain has gap under 3.1%.**
- Per-domain gaps: weather 0.4%, earthquake 0.4–0.8%, NASA 1.1%, stock 1.8%, crypto 2.5%, F1 3.1%
- Per-array margins remain large: 97.3% have >10%, 70.3% have >100%
- **Thesis framing: Contribution is per-array prediction accuracy + structural sensitivity, not aggregate gap inflation**
- Feature extraction validated: 0 NaN, 0 Inf on 62 new arrays

## Rules

- Always use venv: source venv/bin/activate
- Seed 42 for all randomness
- Parquet as canonical output format
- Features normalized on train split constants only
- All sort implementations must be C-level (numpy ops, no Python loops)
- No pretending simple things are novel — position contributions honestly
- Pilot before full pipeline — verify assumptions on small sample first

---

## Real-World Data Collection (Step 1-new) — DONE

### Why
v1–v3 models trained on only 720 synthetic arrays achieved ~60% on real data.
The synthetic distributions (uniform, normal, lognormal, exponential) don't capture real-world structure.
Decision: collect massive real-world data from diverse domains.

### Data Sources (5 domains)

| Domain      | Source                     | Raw Arrays | With Transforms | Method                                |
|-------------|----------------------------|------------|-----------------|---------------------------------------|
| F1          | OpenF1 API (telemetry)     | 487,073    | 885,042         | 2018–2024, all sessions/drivers/laps, 40 channels |
| Stock       | yfinance (market data)     | 20,000     | 100,000         | 209 tickers, 13 periods, 10 derived columns |
| Weather     | Open-Meteo API             | 644        | 3,220           | 7 cities, 10 variables, multiple years |
| Crypto      | yfinance (crypto data)     | 20,000     | 100,000         | 139 tickers, 13 periods, sub-array windowing |
| Earthquake  | USGS API (seismic data)    | 20,001     | 100,003         | 30 time windows, 12 regions, event chunking |
| **TOTAL**   |                            | **547,718**| **1,188,265**   |                                       |

### Structural Transforms (anti-bias)
Applied to sampled subsets to ensure all 3 algorithms have winning regions:
- **RAW**: original real-world data (timsort territory — partially sorted)
- **REV**: reversed array (timsort still wins — recognizes descending runs)
- **SHUF**: random shuffle (introsort territory — no exploitable structure)
- **QBIN50**: quantize to 50 bins (heapsort territory — massive duplicates)
- **PSORT10**: sort then perturb 10% (timsort sweet spot — nearly sorted)

### Storage
- Location: `/Volumes/k/thesis_data/real_world_10k/` (external USB drive, 119 GB)
- Symlink: `data/real_world_10k → /Volumes/k/thesis_data/real_world_10k`
- Total size: ~12 GB
- Format: individual CSV files (one float per line), tracked by `index.csv`

### Array Size Distribution
- Median: 419 elements, Mean: 1,735, Std: 5,542
- 66% are 100–500 elements (dominated by F1 telemetry)
- 14% are 1K–5K, 4.7% are 10K–50K, 0.1% are 50K+

### Scripts
- `scripts/fetch_f1_10k.py` — F1 telemetry fetcher
- `scripts/fetch_stock.py` — Stock market fetcher (209 tickers)
- `scripts/fetch_weather.py` — Weather data fetcher (7 cities)
- `scripts/fetch_crypto.py` — Cryptocurrency fetcher (139 tickers)
- `scripts/fetch_earthquake.py` — Earthquake data fetcher (30 windows × 12 regions)
- `scripts/transform_f1_sample.py` — F1 transform generator (resumable, crash-safe)

---

## Training Dataset Build (Step 2-new) — DONE

### Process
For each of 1,188,265 arrays:
1. Load array from CSV on external drive (via symlink)
2. Extract 16 structural features using `feature_extraction.py`
3. Time 3 sorting algorithms (best-of-5 for n≤10K, best-of-3 for n>10K)
4. Record winner as label

### Output
- File: `data/training_dataset.csv`
- Rows: 1,188,265
- Columns: `file`, `domain`, `n_elements`, 16 features, `time_introsort`, `time_heapsort`, `time_timsort`, `best_algorithm`
- Runtime: 65 minutes (303 arrays/sec), 0 errors

### Algorithm Winner Distribution
| Algorithm  | Count     | Win Rate |
|-----------|-----------|----------|
| timsort    | 1,010,413 | 85.0%    |
| heapsort   | 125,218   | 10.5%    |
| introsort  | 52,634    | 4.4%     |

Note: timsort dominance is expected — median array is 419 elements, mostly partially sorted real-world data. Timsort's adaptive merge sort excels here. Algorithm differences are meaningful only on larger arrays.

### Script
- `scripts/build_training_dataset.py` — resumable, flushes every 5K rows

---

## XGBoost v5 — Balanced Classifier (Step 3-new) — DONE

### Balance Strategy (3-pronged)
1. **Undersample majority**: cap each class at 3× minority count
   - timsort: 1.01M → 90K, heapsort: 76K (kept), introsort: 30K (kept)
   - Training set: ~196K rows (vs 1.18M raw)
2. **Sample weights**: inverse-frequency weighting during training
3. **Label filtering**: keep rows where margin ≥5% OR size ≥2K elements
   - Removes noisy "coin flip" labels from tiny arrays

### Model Configuration
```
XGBoost 3.2.0 (multi:softprob, hist tree method)
n_estimators=500, max_depth=7, lr=0.05
subsample=0.8, colsample_bytree=0.8
min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0
```

### Results

| Split               | Accuracy | Balanced Acc | Introsort Recall | Heapsort Recall | Timsort Recall |
|---------------------|----------|-------------|-----------------|----------------|---------------|
| Train (137K)        | 82.8%    | 80.1%       | 73.2%           | 71.4%          | 95.8%         |
| Validation (29K)    | 76.3%    | 70.3%       | 52.0%           | 63.8%          | 95.0%         |
| Test (29K)          | 76.1%    | 70.1%       | 52.1%           | 63.8%          | 94.5%         |
| Full dataset (1.18M)| 89.1%    | 68.9%       | 39.6%           | 73.5%          | 93.6%         |

### Top Features (by importance)
1. `length_norm` (0.262) — array size is the strongest signal
2. `top5_freq_ratio` (0.184) — duplicate concentration
3. `longest_run_ratio` (0.080) — sorted run structure
4. `duplicate_ratio` (0.080) — uniqueness
5. `entropy_ratio` (0.075) — randomness

### Artifacts
- Model: `models/xgboost_v5/xgb_v5.json`
- Results: `results/xgboost_v5/evaluation_results.json`
- Script: `scripts/train_xgboost_v5.py`

---

## VBS/SBS Regret Analysis (Step 4) — DONE

The definitive evaluation: how much time does the model actually save?

### Definitions
- **VBS (Virtual Best Solver)**: always picks the true fastest algorithm per instance — theoretical ceiling
- **SBS (Single Best Solver)**: always picks one fixed algorithm (the one with lowest total time across all instances) — naive baseline
- **Model**: XGBoost v5 predictions

### headline Results

| Metric                    | Value   | Meaning                                        |
|---------------------------|---------|------------------------------------------------|
| VBS-SBS Gap               | 19.14%  | 19% room for improvement → thesis is justified |
| Model regret vs VBS       | 1.62%   | Only 1.6% slower than perfect oracle           |
| Model lift vs SBS         | 17.83%  | 17.8% faster than always-heapsort              |
| **Gap closed by model**   | **93.1%** | **Model captures 93% of theoretical optimum** |
| Perfect picks (regret=0)  | 89.6%   | Model picks the true best 89.6% of the time    |

### Algorithm Total Times (1.18M arrays)
| Algorithm  | Total Time | Notes                        |
|-----------|-----------|------------------------------|
| VBS        | 17.195s   | Ceiling (perfect oracle)     |
| **Model**  | **17.475s** | **XGBoost v5**             |
| heapsort   | 21.267s   | SBS (best single algorithm)  |
| introsort  | 21.459s   |                              |
| timsort    | 24.624s   |                              |

Key insight: **SBS = heapsort, not timsort.** Although timsort wins 85% of individual arrays,
heapsort has the lowest *total* time because its worst cases are less severe. But timsort's
catastrophic worst case on random data inflates its total.

### Regret by Array Size

| Size Bucket | N Arrays   | VBS-SBS Gap | Model Lift vs SBS |
|-------------|-----------|-------------|-------------------|
| <500        | 787,054   | 33.2%       | 32.8%             |
| 500–2K      | 221,458   | 34.4%       | 34.0%             |
| 2K–10K      | 122,347   | 26.2%       | 24.2%             |
| 10K–50K     | 56,366    | 9.7%        | 8.2%              |
| 50K+        | 1,040     | 10.7%       | 9.7%              |

The model captures most of the gap across ALL size ranges.

### Per-Instance Regret Distribution
- Mean: 0.23 μs, Median: 0.0 μs
- P95: 0.25 μs, P99: 6.12 μs
- Max: 659.25 μs
- **89.6% of instances have zero regret**

### Model Pick Distribution
| Algorithm  | Times Picked | Share  |
|-----------|-------------|--------|
| timsort    | 960,606     | 80.8%  |
| heapsort   | 171,209     | 14.4%  |
| introsort  | 56,450      | 4.8%   |

### Artifacts
- Results: `results/xgboost_v5/regret_analysis.json`
- Script: `scripts/regret_analysis.py`

---

## Remaining Steps

| Step | What                                                        | Status      |
|------|-------------------------------------------------------------|-------------|
| 5    | LinUCB contextual bandit — online adaptation                | NOT STARTED |
| 6    | Comparison: XGBoost vs Bandit vs baselines                  | NOT STARTED |
| 7    | Package as deployable Python library                        | NOT STARTED |
