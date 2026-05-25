# Results & Discussion Chapter — Writing Plan

**Status:** Planning phase (NOT started writing yet)  
**Last Updated:** 2026-05-23  
**Output Format:** Standalone .docx (Word) — Ahmed will integrate into full thesis later  
**Chapter Number:** TBD by Ahmed (likely Chapter 4)

---

## 1. Scope & Purpose

This chapter presents ALL experimental results and their interpretation. It covers:
- v5 production model results (classification + regret)
- Model evolution story (v1→v8) with lessons learnt
- F1 channel-flag multi-model experiment
- LinUCB contextual bandit (design rationale, framed as future direction)
- Domain holdout generalisation analysis
- Limitations and threats to validity

The methodology chapter (version-6) deliberately omitted the model evolution, F1 channel model, and LinUCB from Section 3. They belong here as experimental findings and extensions.

---

## 2. Proposed Chapter Structure

### 4.1 Overview of Experiments

Brief roadmap of what the chapter covers and why. Three experimental threads: (1) the production classifier and its iterative development, (2) a domain-routed extension using F1 metadata, and (3) a designed but unvalidated online adaptation layer.

### 4.2 Model Evolution: From Regression to Classification

The full v1→v8 progression. This is a results narrative, not methodology — each version is presented as a finding that motivated the next design decision.

- **4.2.1 v1: Multi-Output Regression** — 44.4% accuracy on 72 test arrays. Predicted absolute times; dominated by array size effect. Key lesson: regression learns magnitude, not ranking.
- **4.2.2 v2: Direct Classification on Synthetic Data** — 62.5% accuracy. Proved classification outperforms regression. Key lesson: synthetic distributions don't capture real-world structure.
- **4.2.3 v3: Timing-Based Features** — 100% accuracy (data leakage). Used pre-sorting execution times as features. Key lesson: strong signal exists in features, but timing features are circular and not deployable.
- **4.2.4 v5: Production Classifier** — 76.1% accuracy, 93.1% gap closed on 1.18M real arrays. The retained model. Forward reference to detailed results in Section 4.3.
- **4.2.5 v6: Source-Aware Evaluation** — 71.2% accuracy under GroupShuffleSplit. Stricter evaluation prevents source leakage. Gap between v5 and v6 quantifies optimistic bias from shared sources in train/test.
- **4.2.6 v7: Regret-Weighted Training** — 72.9% accuracy, gap closed dropped to 78%. Rejected. Key lesson: optimising directly for regret during training hurts classification accuracy without proportional regret improvement.
- **4.2.7 v8: Binary Cascade** — 72.6% accuracy. Stage 2 introsort-vs-heapsort AUC = 0.603. Key finding: these two algorithms are feature-indistinguishable with the available 16 features. The problem is effectively binary (timsort detection).

Table 4.1: Model evolution summary (version, approach, test accuracy, test size, key finding)

### 4.3 Production Model Results (v5)

Detailed presentation of the retained model's performance.

- **4.3.1 Classification Performance** — 76.1% accuracy, 70.1% balanced accuracy. Per-class breakdown: timsort recall 94.5%, heapsort 63.8%, introsort 52.1%. Confusion matrix analysis.
- **4.3.2 Regret Analysis** — 93.1% gap closed. 1.62% model regret vs VBS. Mean per-instance regret 0.23μs. 89.6% zero-regret predictions. P95=0.25μs, P99=6.12μs, max=659.25μs. Discussion of why 76% accuracy + 93% gap closed is not contradictory.
- **4.3.3 Feature Importance Analysis** — length_norm 26.2%, repetition group 28%, ordering group 21%. Why size dominates. Connection to algorithmic theory (cache effects, partition quality, run detection).
- **4.3.4 Error Analysis** — The introsort-heapsort confusion cell. Why these errors are low-cost. The v8 binary cascade finding as confirmation. Practical implications: the model's "mistakes" are concentrated where they matter least.

Figure 4.1: Confusion matrix (reuse F5 from methodology assets)
Figure 4.2: Feature importance bar chart (reuse F4)
Figure 4.3: Regret distribution (reuse F9)

### 4.4 Generalisation: Domain Holdout Results

- Leave-one-domain-out cross-validation results (Table 4.2)
- Gap closed exceeds 75% in all folds, 89% in four of five
- F1 holdout as weakest fold: largest domain, most structurally distinctive
- Weather holdout: lowest accuracy (60.6%) but high gap closed (89.7%) — errors fall on low-regret instances
- Comparison with full-dataset model per domain (existing_model_per_domain in results)
- Interpretation: structural features transfer across domains; the model learns physics of sorting, not domain idioms

Table 4.2: Domain holdout results (held-out domain, test size, accuracy, balanced accuracy, gap closed, zero-regret %)

### 4.5 F1 Channel-Flag Multi-Model

Secondary experiment: can domain metadata improve selection beyond structural features?

- **4.5.1 Motivation and Design** — Why F1 telemetry channels have distinct signal characteristics (Speed = smooth monotonic, RPM = rapid oscillations, DRS = binary). Mixture-of-experts architecture with hard routing by metadata flag.
- **4.5.2 Extended Algorithm Portfolio** — 5 algorithms (quick_sort, introsort, merge_sort, heap_sort, shell_sort). Why the F1 domain supports a larger portfolio than the main experiment.
- **4.5.3 Per-Channel Results** — Table 4.3 with 9 channels: rows, classes, test accuracy, algorithm set. DRS best (76.4%), RPM worst (44.0%). Discussion of why per-channel accuracy varies.
- **4.5.4 Router Evaluation** — 86.0% non-strict vs 48.9% strict accuracy. Channel-SBS 50.7%, global-SBS 20.0%. The gap between non-strict and strict: what it reveals about information leakage in the non-strict protocol. True performance lies between the two bounds.
- **4.5.5 Implications** — Domain metadata as a routing signal adds value. But the strict evaluation reveals that much of the non-strict gain is optimistic. The approach is promising but requires larger per-channel datasets to validate properly.

Table 4.3: Per-channel model performance
Figure 4.4: Channel-flag routing architecture (reuse F6 from methodology assets)

### 4.6 LinUCB Contextual Bandit

Designed architecture for online adaptation, presented as a research direction.

- **4.6.1 Design Rationale** — Why contextual bandits fit algorithm selection: stateless, single-step decision, one reward signal. Not full RL because no temporal dependencies.
- **4.6.2 Warm-Start from XGBoost** — Offline classifier probabilities initialise the bandit's prior beliefs. As runtime feedback accumulates, the upper confidence bound overrides the prior. Enables adaptation without catastrophic forgetting.
- **4.6.3 Exploration Parameter** — Alpha controls exploitation vs exploration. Higher alpha beneficial under distribution shift but incurs short-term cost.
- **4.6.4 Validation Status** — The bandit is implemented and integrated but not experimentally validated under controlled distribution shift. Convergence properties, regret bounds, and adaptation speed are identified as future work.

### 4.7 Discussion

Interpretive synthesis across all experiments.

- **4.7.1 Accuracy vs Regret: Why Both Matter** — 76% accuracy sounds mediocre in ML terms, but 93% gap closed and 89.6% zero-regret show the model is highly effective in practice. The disconnect arises because accuracy treats all errors equally while regret weights them by cost. Most "wrong" predictions fall in the introsort-heapsort region where runtime differences are negligible.
- **4.7.2 The Binary Nature of the Problem** — v8's AUC=0.603 finding. With the current 16 features, introsort and heapsort are indistinguishable. The effective task is timsort detection, which the model performs at 94.5% recall. Implications for future feature engineering: new features would need to capture cache behaviour or memory access patterns to separate introsort from heapsort.
- **4.7.3 Feature Importance and Algorithmic Theory** — Why length_norm dominates (cache hierarchy crossover points). Why repetition matters (partition quality for quicksort variants). Why ordering matters (timsort's run detection). Connection to the sorting algorithm literature.
- **4.7.4 Domain Independence** — Domain holdout results as evidence that the features capture universal sorting-relevant structure. The Weather fold anomaly: small dataset but maintained gap closed.
- **4.7.5 Metadata Routing vs Global Models** — F1 channel experiment shows domain metadata can help, but strict evaluation tempers optimism. When is metadata routing worth the complexity? When per-domain data is abundant and domains are structurally distinct.
- **4.7.6 Practical Deployment Considerations** — Feature extraction overhead vs sorting time. Break-even array size. When the selector adds value vs when a default algorithm suffices.

### 4.8 Limitations and Threats to Validity

- **Internal validity**: timing noise, near-tie label ambiguity, platform-specific results (single CPU, single OS)
- **External validity**: C-level algorithms only, numeric arrays only, five domains may not cover all real-world patterns
- **Construct validity**: gap closed assumes VBS is the correct upper bound; in practice, algorithm portfolio is fixed
- **Statistical validity**: no confidence intervals on regret metrics; single random seed

### 4.9 Summary

One paragraph tying results back to research objectives.

---

## 3. Figures & Tables Plan

### Figures to Include

| # | Figure | Source | New or Reuse |
|---|--------|--------|-------------|
| F1 | Confusion matrix (v5 test set) | methodology_assets/F5_confusion_matrix.png | Reuse |
| F2 | Feature importance bar chart | methodology_assets/F4_feature_importance.png | Reuse |
| F3 | Regret distribution | methodology_assets/F9_regret_distribution.png | Reuse |
| F4 | Channel-flag routing architecture | methodology_assets/F6_channel_routing.png | Reuse |
| F5 | Model evolution chart/timeline | methodology_assets/F3_model_evolution.png | Reuse |
| F6 | Domain holdout gap closed bar chart | NEW — generate | **New** |
| F7 | Accuracy vs regret scatter/comparison | NEW — generate | **New** |

### Tables to Include

| # | Table | Content |
|---|-------|---------|
| T1 | Model evolution summary | version, approach, test accuracy, test size, key finding (7 versions) |
| T2 | v5 per-class classification report | precision, recall, F1, support for each of 3 classes |
| T3 | v5 regret summary | VBS, SBS, model totals, gap closed, zero-regret %, mean/P95/P99 regret |
| T4 | Domain holdout results | held-out domain, test size, accuracy, balanced acc, gap closed, zero-regret % |
| T5 | Per-channel F1 model performance | channel, rows, classes, test accuracy, algorithm set |
| T6 | Router evaluation comparison | metric, non-strict, strict, channel-SBS, global-SBS |

### New Figures to Generate

| # | What | Data Source |
|---|------|-------------|
| F6 | Domain holdout gap closed bar chart | M6 in metrics_summary.json |
| F7 | Accuracy vs gap closed comparison | M1 + M6 data — shows that low accuracy can coexist with high gap closed |

---

## 4. Metrics to Extract (additional to existing metrics_summary.json)

| # | Metric | Source | Status |
|---|--------|--------|--------|
| M1-M10 | All existing | metrics_summary.json | Already extracted |
| M11 | v6 detailed results | results/xgboost_v6/evaluation_results.json | Need to read |
| M12 | v7 detailed results | results/xgboost_v7/evaluation_results.json | Need to read |
| M13 | v8 detailed results (AUC) | results/xgboost_v8/evaluation_results.json | Need to read |
| M14 | v1 detailed results | results/xgboost_v1/evaluation_results.json | Need to read |
| M15 | Optuna best params per channel | results/f1_9_channel_models_optuna/manifest.json | Need to extract |

---

## 5. Writing Style

Same style guide as PLAN.md Section 8. All rules from the methodology chapter apply:
- 30-75 word paragraphs, 2-4 sentences
- No em/en dashes, no bullets in body text
- British English, formal third person
- Justify then state pattern
- Tables introduced with sentence, followed by interpretive paragraph
- Cross-references by number (Section 4.3, Table 4.1, Figure 4.2)
- No meta-statements ("This section discusses...")

---

## 6. Key Narrative Threads

1. **Accuracy is misleading; regret tells the true story.** This is the central interpretive claim of the chapter.
2. **The problem is effectively binary.** Timsort detection is what the model actually does well; introsort vs heapsort is noise.
3. **Features capture universal sorting structure, not domain idioms.** Domain holdout proves this.
4. **Metadata routing helps but needs more data.** F1 channel experiment is promising but the strict evaluation is sobering.
5. **Online adaptation is the natural next step.** LinUCB bridges the gap between static training and deployment reality.

---

## 7. Preparation Order

1. **Extract additional metrics** (M11-M15) from version result files
2. **Generate new figures** (F6 domain holdout bar chart, F7 accuracy vs gap closed)
3. **Update metrics_summary.json** with discussion-chapter metrics
4. **Verify all numbers** against source files
5. **Write the chapter** in one focused pass
