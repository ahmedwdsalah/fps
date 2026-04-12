# XGBoost v7: Regret-Aware Training Experiment

**Date:** 2026-04-11  
**Status:** COMPLETED — experimental only, v5 remains production model  
**Script:** `scripts/train_xgboost_v7_regret_aware.py`  
**Results:** `results/xgboost_v7/evaluation_results.json`

---

## Motivation

v5 optimizes for classification accuracy (mlogloss), but the real goal is minimizing regret (time wasted by choosing the wrong algorithm). Hypothesis: weighting training samples by their regret cost — making expensive mistakes matter more — could improve gap closed even if accuracy drops.

## Changes from v5

| Parameter | v5 | v7 |
|-----------|-----|-----|
| max_depth | 7 | 5 |
| n_estimators | 500 | 1500 |
| learning_rate | 0.05 | 0.05 |
| Sample weights | Inverse-frequency | Regret-proportional |
| Evaluation | mlogloss | mlogloss + regret |

### Regret-Weighted Sample Weights

Instead of uniform inverse-frequency weights, v7 computes per-instance weights proportional to:
```
weight(i) = max(time_others(i)) - time_best(i) + epsilon
```

Arrays where choosing wrong is expensive get higher weight. Arrays where all algorithms are similar (low regret ceiling) get minimal weight.

## Results

### Accuracy

| Metric | v5 | v7 |
|--------|-----|-----|
| Test accuracy | 76.1% | ~72% |
| Balanced accuracy | 70.1% | ~65% |

### Regret Analysis

| Metric | v5 | v7 |
|--------|-----|-----|
| Gap closed | **93.14%** | ~78% |
| Perfect picks | **89.64%** | ~82% |
| Mean regret | **0.23 μs** | ~0.45 μs |

v7 regressed across **every metric**.

## Root Cause Analysis

### 1. max_depth=5 → Underfitting
Reducing depth from 7 to 5 limited the model's capacity. Evidence: early stopping never fired — the model trained all 1,496 of 1,500 rounds without the validation loss improving, indicating insufficient model complexity rather than overfitting.

### 2. Regret Weights Backfired
Regret-proportional weights **downweight boundary cases** — the exact instances where the model needs the most training signal. When introsort and heapsort are close (low regret), the weight goes near zero. But those are the majority of non-timsort instances, and the model loses its ability to distinguish them at all.

### 3. Cost Matrix Reveals the Problem
Computing the cost matrix showed:
```
introsort→heapsort penalty: ~1.5 μs (negligible)
heapsort→introsort penalty: ~1.5 μs (negligible)
timsort→heapsort penalty:   ~36 μs  (expensive)
timsort→introsort penalty:  ~50 μs  (very expensive)
```

The model should maximize timsort recall (which v5 already does at 94.5%). Regret weighting didn't help because the high-cost errors (timsort misclassification) were already rare in v5.

## Key Finding

**v5's accuracy-based training already implicitly minimizes regret** because:
1. Timsort (the high-stakes class) has the strongest feature signal → highest recall
2. Introsort↔heapsort confusion (the majority of errors) has negligible regret cost
3. Inverse-frequency weights provide sufficient balance without distorting learning

Explicitly optimizing for regret provides no benefit when the cost structure already aligns with feature separability.

## Conclusion

v7 is a **negative result**. Regret-aware sample weighting is theoretically appealing but empirically harmful for this specific problem because:
- The expensive errors are already rare
- The cheap errors dominate and become undertrained
- Shallower depth compounds the problem

**Decision:** v5 remains the production model. v7 demonstrates that accuracy optimization is a sufficient proxy for regret minimization in this domain.

---

*See also: `docs/checkpoint-xgboost-v5-baseline.md`, `docs/xgboost-v8-binary-cascade-report.md`*
