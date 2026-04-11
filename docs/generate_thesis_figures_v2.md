# Script: generate_thesis_figures_v2.py

## Purpose
Generate 11 publication-quality thesis figures with improved labeling and academic language. v2 refinements on v1.

## Improvements from v1
✅ Enhanced x-axis labels (clearer, better formatted)
✅ Academic naming: v6 = "Source-Aware" (not "Honest")
✅ Formal language: "model", "framework", "selector" (not "our")
✅ Better font sizes and spacing throughout
✅ Improved readability on all figures

## Inputs
- `results/xgboost_v5/evaluation_results.json` — v5 metrics
- `results/xgboost_v5/predictions_test.csv` — v5 predictions
- `results/xgboost_v5/regret_analysis.json` — regret data
- `results/xgboost_v6/evaluation_results.json` — v6 metrics
- `results/domain_holdout/domain_holdout_results.json` — domain performance

## Outputs
- `results/thesis_figures_v2/01_model_history.png` — v1→v6 progression
- `results/thesis_figures_v2/02_v5_accuracy_metrics.png` — Train/Val/Test accuracy
- `results/thesis_figures_v2/03_v5_per_algorithm_recall.png` — Per-algorithm metrics
- `results/thesis_figures_v2/04_feature_importance.png` — 16-feature rankings
- `results/thesis_figures_v2/05_regret_analysis.png` — Regret distribution + metrics
- `results/thesis_figures_v2/06_vbs_sbs_model_comparison.png` — Strategy comparison
- `results/thesis_figures_v2/07_domain_holdout.png` — Cross-domain performance
- `results/thesis_figures_v2/08_confusion_matrix.png` — Test set confusion
- `results/thesis_figures_v2/09_predictions_distribution.png` — Selector bias
- `results/thesis_figures_v2/10_v5_vs_v6_comparison.png` — Production vs source-aware
- `results/thesis_figures_v2/11_balanced_accuracy.png` — Balanced accuracy progression

## Key Labeling Changes

### Academic Naming
- ❌ "Honest" → ✅ "Source-Aware" (v6)
- ❌ "our selection" → ✅ "the model", "the framework", "the selector"

### X-Axis Improvements
- Better formatted labels (e.g., "Training\nSet" instead of "train")
- Clearer algorithm names (Introsort, Heapsort, Timsort)
- Multi-line labels where needed for clarity

### Font & Spacing
- Better font sizes for readability
- Improved label positioning
- Enhanced grid and annotation clarity

## Status
✅ **Complete** — All 11 figures generated with improved academic language and labeling.

## How to Use
```bash
cd /Users/ahmed/Desktop/thesis/My-Master-thesis
source venv/bin/activate
python scripts/generate_thesis_figures_v2.py
```

Output: `results/thesis_figures_v2/*.png`

## For Thesis
Copy figures from `results/thesis_figures_v2/` directly into thesis chapters. All labels are academic and publication-ready.

## Cross-links
- v1 script: `scripts/generate_thesis_figures_v1.py` (baseline)
- Documentation: `docs/generate_thesis_figures_v1.md` (detailed findings)
- Obsidian: [[result-thesis-figures-v1]]
