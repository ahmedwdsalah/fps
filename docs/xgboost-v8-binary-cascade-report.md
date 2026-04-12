# XGBoost v8: Binary Cascade Experiment

**Date:** 2026-04-11  
**Status:** COMPLETED — experimental only, v5 remains production model  
**Script:** `scripts/train_xgboost_v8_binary_cascade.py`  
**Results:** `results/xgboost_v8/evaluation_results.json`

---

## Motivation

v5 achieves 93.14% gap closed but introsort recall is only 52%. The confusion matrix shows heavy introsort↔heapsort misclassification. Hypothesis: a two-stage binary cascade could separate the easy decision (timsort vs rest) from the hard one (introsort vs heapsort).

## Architecture

```
Input Array → 16 Features
     │
     ▼
┌──────────────────────────────┐
│  Stage 1: timsort vs rest    │  Binary classifier
│  (AUC = 0.982)               │  1000 trees, max_depth=7
└──────────┬───────────────────┘
           │
    timsort ← YES    NO →
                      │
                      ▼
              ┌───────────────────────────┐
              │  Stage 2: intro vs heap   │  Binary classifier
              │  (AUC = 0.603)            │  500 trees, max_depth=6
              └───────────────────────────┘
                      │
              introsort ← or → heapsort
```

## Dataset

| Subset | Count |
|--------|-------|
| Total raw | 1,188,265 |
| After undersample | 335,754 |
| Train | 235,027 |
| Val | 50,363 |
| Test | 50,364 |
| Stage 2 train (non-timsort only) | 124,496 |

## Results

### Stage 1: timsort vs rest — Near-Perfect

| Metric | Value |
|--------|-------|
| AUC | **0.982** |
| Best iteration | 651 / 1000 |

Stage 1 confirms timsort is trivially separable. The 16 structural features (especially `entropy_ratio` at 0.317 importance) cleanly identify timsort-optimal arrays.

### Stage 2: introsort vs heapsort — Near-Random

| Metric | Value |
|--------|-------|
| AUC | **0.603** |
| Best iteration | 156 / 500 |

Stage 2 AUC of 0.603 is barely above 0.5 (coin flip). This is the **key finding**: introsort and heapsort are **feature-indistinguishable** with the current 16 O(n) features. No amount of model tuning can fix this — the signal doesn't exist in the feature space.

### Stage 2 Feature Importance (flat distribution)

| Feature | Importance |
|---------|------------|
| length_norm | 0.160 |
| duplicate_ratio | 0.109 |
| top5_freq_ratio | 0.078 |
| runs_ratio | 0.072 |
| dispersion_ratio | 0.059 |
| adj_sorted_ratio | 0.054 |
| ... | (all remaining between 0.041–0.054) |

No single feature dominates — the model is grasping at noise. Compare with Stage 1 where `entropy_ratio` alone carries 0.317. When a real signal exists, importance concentrates; when it doesn't, importance is flat.

### End-to-End Cascade Evaluation

| Metric | v5 (3-class) | v8 (cascade) | v8 (heap default) |
|--------|-------------|-------------|-------------------|
| Accuracy | 72.8% | 72.6% | 58.0% |
| Gap closed | 78.1% | 78.3% | **83.1%** |
| Perfect picks | 64.6% | 65.1% | **80.7%** |
| Mean regret (μs) | 0.469 | 0.465 | **0.364** |
| P95 regret (μs) | 1.875 | 1.833 | 1.125 |
| P99 regret (μs) | 8.166 | 7.958 | 7.416 |

**Note:** These metrics are on the v8 test split (50K), not the full 1.18M dataset. The v5 numbers here differ from the full-dataset regret analysis (93.14% gap closed) because of different evaluation sets.

### "Heap Default" Strategy

Defaulting all non-timsort predictions to heapsort (ignoring Stage 2 entirely) actually produces the best gap closed (83.1%) on this test split. This makes sense: Stage 2 is near-random, so defaulting to heapsort avoids 50% of its coin-flip errors.

However, **this defeats the thesis purpose**. The goal is adaptive per-array selection, not heuristic defaults. The "heap default" result is documented as evidence that introsort↔heapsort is the fundamental bottleneck, not as a recommended approach.

---

## Key Findings

### 1. The Real Decision Is Binary
Timsort detection is near-perfect (AUC=0.982). The 16 features excel at identifying presorted/partially-sorted arrays. This is the decision that matters — timsort misclassification costs 36-50 μs, while introsort↔heapsort confusion costs ~1.5 μs.

### 2. Introsort ≈ Heapsort Under Structural Features
AUC=0.603 proves these algorithms cannot be distinguished by O(n) structural features. They have fundamentally similar performance profiles on non-sorted data. Possible reasons:
- Both are comparison-based O(n log n) algorithms with similar constant factors
- Their performance differences depend on cache behavior and branch prediction — properties not captured by structural features
- The timing differences at small-to-medium sizes are within measurement noise

### 3. Cascade Adds Complexity Without Benefit
v8 cascade (78.3% gap closed on test) ≈ v5 3-class (78.1% gap closed on test). The architectural complexity of two models provides no measurable improvement. v5's single multi-class model is simpler and equally effective.

### 4. Feature Importance Concentration = Signal Strength
Stage 1: `entropy_ratio` dominates (0.317) → strong signal exists.
Stage 2: flat distribution (max 0.160) → no usable signal.
This pattern is a useful diagnostic: when importance concentrates, the model has learned something real; when it's flat, it's fitting noise.

---

## Figures

9 figures generated in `results/xgboost_v8/figures/`:

| Figure | Description |
|--------|-------------|
| `01a_stage1_training.png` | Stage 1 training loss curves |
| `01b_stage2_training.png` | Stage 2 training loss curves |
| `02_stage1_roc.png` | Stage 1 ROC curve (AUC=0.982) |
| `03_stage2_roc.png` | Stage 2 ROC curve (AUC=0.603) |
| `04_confusion_matrices.png` | Both stages confusion matrices |
| `05_feature_importance.png` | Side-by-side Stage 1 vs Stage 2 importance |
| `06_regret_comparison.png` | v5 vs v8 vs heap-default regret |
| `07_timsort_probability_dist.png` | Stage 1 probability distribution |
| `08_threshold_sweep.png` | Stage 1 threshold sweep analysis |

---

## Conclusion

v8 is a **negative result that strengthens the thesis**. It demonstrates:
1. The algorithm selection problem for sorting is fundamentally a **timsort-detection problem** under structural features
2. Introsort vs heapsort discrimination requires information beyond O(n) features (likely cache/hardware-level signals)
3. v5's simpler 3-class architecture is the right choice — no benefit from decomposition

**Decision:** v5 remains the production model. v8 is documented as an ablation study.

---

*See also: `docs/checkpoint-xgboost-v5-baseline.md`, `results/xgboost_v8/evaluation_results.json`*
