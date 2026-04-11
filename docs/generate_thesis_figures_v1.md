# Script: generate_thesis_figures_v1.py

## Purpose
Generate 11 publication-quality thesis figures from existing results (v5, v6, domain holdout, regret analysis).

Used for thesis chapters to visualize:
- Model evolution and accuracy
- Feature importance
- Regret analysis and gap closed
- Domain generalization
- Per-algorithm recall

## Inputs
- `results/xgboost_v5/evaluation_results.json` — v5 model metrics
- `results/xgboost_v5/predictions_test.csv` — v5 test predictions
- `results/xgboost_v5/regret_analysis.json` — VBS-SBS-model regret
- `results/xgboost_v6/evaluation_results.json` — v6 honest evaluation
- `results/domain_holdout/domain_holdout_results.json` — per-domain performance

## Outputs
- `results/thesis_figures_v1/01_model_history.png` — v1→v6 accuracy progression
- `results/thesis_figures_v1/02_v5_accuracy_metrics.png` — Train/val/test accuracy
- `results/thesis_figures_v1/03_v5_per_algorithm_recall.png` — Per-algorithm recall/precision
- `results/thesis_figures_v1/04_feature_importance.png` — Top 16 features
- `results/thesis_figures_v1/05_regret_analysis.png` — Regret distribution & metrics
- `results/thesis_figures_v1/06_vbs_sbs_model_comparison.png` — Speed comparison
- `results/thesis_figures_v1/07_domain_holdout.png` — Per-domain performance
- `results/thesis_figures_v1/08_confusion_matrix.png` — v5 test confusion matrix
- `results/thesis_figures_v1/09_predictions_distribution.png` — Model selector bias
- `results/thesis_figures_v1/10_v5_vs_v6_comparison.png` — Production vs honest evaluation
- `results/thesis_figures_v1/11_balanced_accuracy.png` — Balanced accuracy progression
- `results/thesis_figures_v1/generation_summary.json` — Metadata

## Key Findings

### 1. Model Evolution (Fig 1)
- v1 (regressors): 44.4% (failed)
- v2 (classifier): 62.5% (baseline)
- v3 (log+pairwise): 100% but leaky (timing features)
- v5 (production): **76.1%** ✓
- v6 (honest): 71.2% (source-aware splits)

### 2. Accuracy Metrics (Fig 2, 11)
- v5 train accuracy: 82.8% / balanced: 80.1%
- v5 val accuracy: 76.3% / balanced: 70.3%
- v5 test accuracy: **76.1%** / balanced: **70.1%**
- Good generalization (train/val/test are close)

### 3. Per-Algorithm Recall (Fig 3)
- **Timsort**: 94.5% recall, 96% precision (easiest class)
- **Heapsort**: 63.8% recall, 74.2% precision (medium)
- **Introsort**: 52.1% recall, 37.1% precision (hardest, low signal)

**Observation:** Introsort vs heapsort discrimination is weak. Both are fast on random/mixed data; features struggle to distinguish them.

### 4. Feature Importance (Fig 4)
Top 5 features:
1. `length_norm`: 0.262
2. `top5_freq_ratio`: 0.184
3. `longest_run_ratio`: 0.080
4. `duplicate_ratio`: 0.079
5. `entropy_ratio`: 0.075

**Note:** All 16 features validated and working (214/214 tests). Top 5 account for 68.9% of model decision weight.

### 5. Regret Analysis (Fig 5, 6)
- **VBS-SBS gap**: 19.14% (best vs naive single algorithm)
- **Model closes**: 93.14% of that gap
- **Model overhead vs VBS**: 1.62%
- **Perfect picks**: 89.64% (zero regret arrays)

Per-instance regret:
- Mean: 0.23 μs
- Median: 0.0 μs (89.64% are zero)
- P95: 0.25 μs (still excellent)
- P99: 6.12 μs
- Max: 659.25 μs (100× spike, outliers exist)

**Interpretation:** On 9 out of 10 arrays, the model picks perfectly. When wrong, cost is usually <1 μs. Long tail (max 659 μs) suggests systematic failures on specific array types.

### 6. VBS vs SBS vs Model (Fig 6)
On 1.18M arrays:
- **VBS (oracle)**: 17.195 s
- **Model v5**: 17.475 s (1.62% overhead)
- **SBS (Heapsort)**: 21.267 s

**Bottom line:** Our model is near-oracle performance, vastly better than naive choice.

### 7. Domain Holdout (Fig 7)
Generalization across domains (train/test from different sources):
- **Crypto**: 88.6% accuracy, 90.7% gap closed
- **Earthquake**: 86.6% accuracy, 79.9% gap closed
- **F1**: 91.1% accuracy, 92.3% gap closed
- **Stock**: 89.0% accuracy, 91.4% gap closed
- **Weather**: 60.6% accuracy, 89.7% gap closed ⚠️

**Issue:** Weather domain much weaker (only 0.3% of training data). Model still closes most gap even there.

### 8. Confusion Matrix (Fig 8)
- **Timsort**: 94.5% correctly identified (12,781 / 13,528)
- **Heapsort**: 63.8% correctly identified (7,307 / 11,457)
- **Introsort**: 52.1% correctly identified (2,350 / 4,509)

Main confusions:
- Introsort often mislabeled as heapsort
- Heapsort sometimes mislabeled as timsort

### 9. Predictions Distribution (Fig 9)
Model picks:
- Timsort: 80.8% (960,606 arrays)
- Heapsort: 14.4% (171,209 arrays)
- Introsort: 4.8% (56,450 arrays)

Matches raw label distribution closely. Model learned that timsort is default, only switches when features indicate otherwise.

### 10. v5 vs v6 (Fig 10)
- v5 (production, mixed data): 76.1%
- v6 (honest, source-aware): 71.2%

v6 is stricter (source leakage removed) but still strong. 71.2% from only structural features with honest evaluation is solid.

## Status
✅ **Complete** — All 11 figures generated successfully with 300 DPI, suitable for thesis publication.

## How to Use
```bash
cd /Users/ahmed/Desktop/thesis/My-Master-thesis
source venv/bin/activate
python scripts/generate_thesis_figures_v1.py
```

Output: `results/thesis_figures_v1/*.png`

## For Thesis Chapters
These figures are ready to insert:
- **Introduction/Overview**: Fig 1 (model evolution)
- **Methodology**: Fig 4 (features)
- **Results – Accuracy**: Fig 2, 11 (train/val/test)
- **Results – Generalization**: Fig 7 (domain holdout)
- **Results – Per-Algorithm**: Fig 3 (recall/precision)
- **Results – Regret**: Fig 5, 6 (gap closed, overhead)
- **Analysis – Model Internals**: Fig 8 (confusion), Fig 9 (selector bias)
- **Comparison**: Fig 10 (v5 vs v6 honest)

## Cross-links
- Obsidian: [[model-v5-production]], [[model-v6-honest]], [[result-domain-holdout]], [[result-regret-analysis]]
- Related: `scripts/test_xgboost_v5.py`, `scripts/domain_holdout_test.py`
- Results: `results/thesis_figures_v1/`, `results/xgboost_v5/`, `results/xgboost_v6/`, `results/domain_holdout/`
