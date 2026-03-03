# Step 3: XGBoost v1 — Baseline Regressor Results

**Date:** $(date)  
**Script:** `scripts/train_xgboost.py`  
**Model version:** `xgboost_v1`

---

## Architecture

Three independent XGBoost regressors, one per algorithm:
- `xgb_introsort.json` — predicts `time_introsort`
- `xgb_heapsort.json` — predicts `time_heapsort`
- `xgb_timsort.json` — predicts `time_timsort`

**Selection rule:** `argmin(predicted_time_introsort, predicted_time_heapsort, predicted_time_timsort)`

**Features:** 16 structural features (see `docs/feature-definitions.md`)

**Hyperparameters (conservative baseline):**
| Parameter | Value |
|---|---|
| n_estimators | 300 |
| max_depth | 6 |
| learning_rate | 0.05 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |
| tree_method | hist |

---

## Regression Quality

| Algorithm | Train RMSE | Val RMSE | Val R² |
|---|---|---|---|
| introsort | 0.001819 | 0.002227 | 0.889 |
| heapsort | 0.001929 | 0.001793 | 0.920 |
| timsort | 0.003687 | 0.004143 | 0.957 |

**Interpretation:** R² values are high (0.89–0.96), meaning the model *predicts absolute times well*. But this is misleading — high R² on times is mostly driven by array size (larger arrays → longer times). What matters is whether the model can distinguish *which algorithm is fastest* for a given array, not whether it can predict the absolute time.

---

## Selection Accuracy Results

| Split | N | Accuracy | Acceptable (≤5% of oracle) | Gap Recovery |
|---|---|---|---|---|
| Train | 216 | **51.4%** | 73.6% | 82.0% |
| Val | 72 | **43.1%** | 69.4% | 63.8% |
| Test A | 72 | **44.4%** | 68.1% | 77.0% |
| Test B | 360 | **52.5%** | 85.0% | 58.6% |
| Real-world | 309 | **57.9%** | 76.4% | **−163.5%** |

### Baselines for context
- **Random guess:** 33.3% (3 algorithms)
- **Always-heapsort (SBS):** depends on split, ~44%–58%

---

## Honest Assessment

### What's working
1. **Better than random on all splits** (43–58% vs 33% random baseline)
2. **Acceptable selections are decent** (68–85%) — even when wrong, the model often picks the 2nd-best algorithm
3. **Test B (unseen distributions) doesn't collapse** (52.5%) — mild generalization
4. **Sorted arrays are near-perfect** (91–95%) — the model correctly learns timsort dominates sorted data

### What's NOT working

#### 1. Accuracy is low: 43–52% on test data
This is only ~10–20 percentage points above random. For a thesis contribution, we need this significantly higher.

#### 2. Massive heapsort bias
The model predicts heapsort far too often:

| Split | Predicted heapsort | True heapsort |
|---|---|---|
| Test B | 263/360 (73%) | 164/360 (46%) |
| Real | 249/309 (81%) | 180/309 (58%) |

The model is essentially learning "heapsort is usually fastest" and over-predicting it. This collapses the model towards single-best-solver behavior.

#### 3. Real-world gap recovery is NEGATIVE (−163.5%)
This means the model is **worse than always-heapsort** on real data. The SBS gap is tiny (1.4%), but the model's gap is 3.7% — it's *adding* regret, not removing it. The model's incorrect predictions on real data cost more than the gains from correct ones.

#### 4. Nearly-sorted and sorted_runs are terrible (8–33%)
These are precisely the structures where algorithm selection should matter most (timsort vs heapsort), and the model fails.

#### 5. Train accuracy is only 51.4%
The model can't even fit the training data well. This suggests the features don't carry enough discriminative signal for the regression approach, OR the time differences between algorithms are so small that regression noise swamps the ranking.

---

## Root Cause Analysis

### The fundamental problem: regression ≠ ranking
We're training regressors to predict absolute times, then using argmin for selection. But what we actually need is to predict the **ranking** (which is fastest?), not the absolute values.

Consider: if heapsort takes 0.050s and introsort takes 0.051s, the regression needs to be accurate to <0.001s to distinguish them. But if heapsort takes 0.050s and timsort takes 0.200s, a much coarser prediction works. The model wastes capacity predicting absolute magnitudes (driven by array size) rather than algorithm-relative differences.

### Secondary issues
1. **216 training samples** is very small for 300-tree XGBoost
2. **Only 2 distributions** in training (uniform, normal) — may need augmentation
3. **Feature cost vs signal**: Top features (length_norm, mean_abs_diff_norm, top1_freq_ratio) explain most variance, while structural features (adj_sorted_ratio, inversion_ratio) that should be discriminative have low importance

---

## Feature Importance

| Rank | Feature | Avg Gain |
|---|---|---|
| 1 | length_norm | 0.2236 |
| 2 | mean_abs_diff_norm | 0.1830 |
| 3 | top1_freq_ratio | 0.1784 |
| 4 | top5_freq_ratio | 0.1254 |
| 5 | duplicate_ratio | 0.0976 |
| 6 | entropy_ratio | 0.0530 |
| 7 | iqr_norm | 0.0379 |
| 8 | skewness_t | 0.0272 |
| 9 | longest_run_ratio | 0.0235 |
| 10 | runs_ratio | 0.0157 |
| 11 | adj_sorted_ratio | 0.0141 |
| 12 | kurtosis_excess_t | 0.0064 |
| 13 | inversion_ratio | 0.0057 |
| 14 | outlier_ratio | 0.0040 |
| 15 | mad_norm | 0.0032 |
| 16 | dispersion_ratio | 0.0014 |

**Key concern:** `length_norm` is #1 by a large margin. This means the model is primarily learning "big arrays → heapsort" rather than learning structural patterns. The sortedness features (adj_sorted_ratio: #11, inversion_ratio: #13) that should carry the most discriminative signal are nearly ignored.

---

## Per-Structure Breakdown (Test A)

| Structure | Accuracy | Note |
|---|---|---|
| sorted | 91.7% | ✅ Easy — timsort always wins |
| few_unique | 58.3% | Decent |
| random | 58.3% | Decent |
| nearly_sorted | 33.3% | ❌ Critical failure |
| sorted_runs | 16.7% | ❌ Critical failure |
| reverse_sorted | 8.3% | ❌ Critical failure |

The model completely fails on the structures where algorithm selection is most important and most definitively determined by structure.

---

## Confusion Matrix (Test B — largest test split)

```
                 introsort    heapsort     timsort
    introsort          18         102           6
     heapsort          19         131          14
      timsort           0          30          40
```

**Reading:** When the true best is introsort (126 arrays), the model predicts heapsort 102 times (81% misclassification). The model is a "heapsort-predictor with exceptions."

---

## What this means for the thesis

This v1 result is **expected and useful** as a baseline:
1. It proves naive XGBoost regression is insufficient → justifies more sophisticated approaches
2. It identifies the key failure mode (regression→ranking mismatch) → motivates pairwise/classification reformulation
3. The 73–85% "acceptable" rate shows the regret is small even when wrong → the problem has exploitable structure
4. The structural breakdown pinpoints exactly where improvement is needed

---

## Next Steps (v2 improvements to explore)

1. **Classification reformulation**: Train XGBoost classifier directly on `best_algorithm` label instead of regression
2. **Pairwise ranking**: Train on time-difference targets (heapsort_time − introsort_time) instead of absolute times
3. **Log-transform times**: Reduce scale-dependence, make small differences more learnable
4. **Feature engineering**: Add interaction features (e.g., sortedness × size)
5. **More training data**: Include test_B distributions in training fold
6. **Hyperparameter tuning**: Bayesian optimization on val accuracy
7. **Cost-sensitive objective**: Weight errors by actual regret (wrong on close races matters less than wrong on clear wins)

---

## Artifacts

| File | Description |
|---|---|
| `models/xgboost_v1/xgb_introsort.json` | XGBoost model for introsort |
| `models/xgboost_v1/xgb_heapsort.json` | XGBoost model for heapsort |
| `models/xgboost_v1/xgb_timsort.json` | XGBoost model for timsort |
| `models/xgboost_v1/metadata.json` | Training metadata & hyperparameters |
| `results/xgboost_v1/evaluation_results.json` | Full evaluation results (all metrics) |
| `results/xgboost_v1/predictions_train.csv` | Per-sample predictions on train |
| `results/xgboost_v1/predictions_val.csv` | Per-sample predictions on validation |
| `results/xgboost_v1/predictions_test_A.csv` | Per-sample predictions on test A |
| `results/xgboost_v1/predictions_test_B.csv` | Per-sample predictions on test B |
| `results/xgboost_v1/predictions_real.csv` | Per-sample predictions on real-world |
