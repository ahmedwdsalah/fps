# Methodology Chapter — Writing Plan

**Status:** Planning phase (NOT started writing yet)  
**Last Updated:** 2026-05-17  
**Output Format:** Standalone .docx (Word) for Chapter 3 only — Ahmed will integrate into full thesis later  

---

## 1. Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| What to do with existing Chapter 3 | **Rewrite from scratch** | Start fresh using full project knowledge |
| Primary model | **v5** (76.1% accuracy, 93.1% gap closed) | Production model; mention v6 as stricter validation variant |
| Model evolution story | **Include it** | Shows scientific rigor, justifies design decisions |
| Output format | **.docx (Word)** | CIU-compliant formatting |

---

## 2. CIU Formatting Requirements

- Font: Times New Roman, 12pt body, 14pt main headings
- Line spacing: 1.5 (single for footnotes, abstract, references)
- Margins: top 4cm, left 3.5cm, right 3cm, bottom 2.5cm
- Alignment: Justified (except headings)
- Headings: Arabic numerals, decimal system (3.1, 3.2, 3.2.1)
- Tables: title above, source below, left-aligned, 10pt content, bold numbering (e.g., **Table 3.1**)
- Figures: title below, source below, centered, bold numbering (e.g., **Figure 3.1**)
- Equations: centered, numbered right-aligned (3.1), (3.2), etc.
- Citations: APA style, in-text
- Word count target: 8,000–20,000 for full thesis (Basic Sciences Master's)

---

## 3. Project Understanding Summary

**Thesis:** Adaptive Sorting Algorithm Selection Using Machine Learning (Offline + Online)

**Core idea:** Given a 1D numeric array, extract 16 O(n) structural features → predict which of 3 C-level sorting algorithms (introsort, heapsort, timsort) will be fastest.

**Two-layer architecture:**
- Layer 1 (Offline): XGBoost classifier trained on 1.18M real-world arrays
- Layer 2 (Online): LinUCB contextual bandit for distribution shift adaptation

**Key results (v5):**
- 76.1% top-1 accuracy
- 93.1% VBS-SBS gap closed
- 1.62% regret vs oracle
- 89.6% zero-regret predictions

**Key finding (v8):** Introsort and heapsort are feature-indistinguishable (AUC 0.603) — problem is effectively binary (timsort detection).

**F1 Channel-Flag Multi-Model (missed in initial exploration):**
- A separate experiment using F1 telemetry data with an expanded 5-algorithm portfolio
- Uses the channel name (Speed, RPM, Throttle, Distance, nGear, DRS, X, Y, Z) as a **routing flag**
- Trains 9 independent XGBoost models — one per F1 channel — each with channel-specific algorithm classes
- Uses 17 features (the 16 structural features + n_elements)
- Dataset: 13,215 rows from F1 telemetry, balanced across 5 algorithms (quick_sort, introsort, merge_sort, heap_sort, shell_sort)
- Three training iterations: baseline → dynamic class sets (v2) → Optuna-tuned (20 trials/channel)
- Dynamic router accuracy: 86.0% (non-strict) / 55.4% (strict train/test split)
- Baselines: channel-SBS 50.7%, global-SBS 20.0%
- Models exported to ONNX for browser-based inference
- Interactive Svelte + WebSHAP demo with client-side SHAP explanations
- Key insight: domain metadata (channel flag) as a routing signal improves selection by specialising models to data characteristics of each signal type

---

## 4. Proposed Chapter 3 Structure

> **This section is a draft outline — to be discussed and refined before writing begins.**

### 3.1 Research Methodology Overview
- This is a **supervised classification** problem (not regression — explain why, with v1 failure as evidence)
- The ML pipeline: problem formulation → feature engineering → data collection → model training → evaluation
- Why **per-instance algorithm selection** (not portfolio-level or config-space search)
- Formal problem statement: given feature vector x ∈ R^16, predict a* = argmin_a T_a(x)
- Why XGBoost (gradient-boosted trees) over alternatives: random forest, SVM, neural networks, decision trees
  - Handles tabular data natively, feature interactions, built-in regularisation, interpretable
- Why classification over regression: regression learns T(x), but we need argmin — ranking ≠ magnitude prediction
- Why contextual bandits over full RL: stateless problem, one-step decision, no sequential dependency

### 3.2 System Architecture
- Two-layer design (offline + online)
- Inference flow diagram
- Graceful degradation (Layer 1 only when no feedback)

### 3.3 Algorithm Portfolio
- Why these 3 algorithms — theoretical performance characteristics that create a selection landscape
  - Introsort: O(n log n) worst-case hybrid (quicksort + heapsort fallback), good on random data
  - Heapsort: O(n log n) guaranteed, O(1) space, cache-unfriendly but consistent
  - Timsort: adaptive merge sort, O(n) best-case on sorted data, exploits natural runs
- Why C-level only: fair comparison requires same implementation language; Python-loop sorts measure language overhead not algorithm characteristics
- Empirical elimination process: counting sort (0/720 wins), radix sort (20× slower), bucket sort, bubble/insertion/shell (can't reach C-level without Cython)
- The selection landscape: no single algorithm dominates — timsort wins on structured data, introsort/heapsort compete on random data

### 3.4 Feature Engineering
- **Why feature engineering matters for algorithm selection**: features must be informative, cheap, and robust
- **O(n) constraint rationale**: feature extraction must be cheaper than sorting itself (O(n log n)), otherwise no point in selecting — just sort directly
- **Feature design principles**: each feature maps to a known algorithmic behaviour
  - Sortedness features (adj_sorted_ratio, inversion_ratio) → timsort's adaptive advantage
  - Uniqueness features (duplicate_ratio, top1/top5_freq) → partition quality for quicksort, equal-key runs for timsort
  - Spread/shape features (dispersion, entropy, skewness, kurtosis) → data distribution characteristics
  - Scale feature (length_norm) → algorithm crossover points at different sizes
- **Mathematical definitions** of each feature (table + per-group subsections with formulas)
- **Normalisation and bounding**: why most features are clipped to [0,1], signed-log transform for skewness/kurtosis
- **inversion_ratio exception**: O(n log n) merge-sort-based count, subsampled to 2000 elements for arrays >10K — explain the trade-off
- **Numerical stability**: epsilon guards (1e-12) for division by zero, edge cases (constant arrays, single-element arrays)
- **Feature validation**: 214/214 tests passed, zero NaN, zero Inf across 1.18M arrays
- **Single source of truth**: one Python function imported by every script — prevents feature drift

### 3.5 Data Collection
- **Why real-world data over synthetic**: v2 showed synthetic distributions (uniform, normal, lognormal) don't capture real structure — 60% accuracy on real data vs 62.5% on synthetic. Real data has natural correlations, trends, and patterns that synthetic generation misses.
- **Domain diversity rationale**: 5 domains chosen to cover different data characteristics
  - F1 telemetry (time-series, auto-correlated), stock/crypto (financial returns, heavy tails), earthquake (geophysical, spatial), weather (smooth, periodic)
- **1.18M arrays, domain breakdown** with table
- **Structural transforms** (within this section): RAW, REV, SHUF, QBIN50, PSORT10
  - Why: force diversity in sortedness/structure so model doesn't just learn "real data is always nearly sorted → always pick timsort"
  - Each transform targets a specific algorithm's strength/weakness
- **8 data quality controls**: table of problems identified and solutions
  - Duplicate arrays, NaN/Inf values, constant arrays, too-small arrays, domain imbalance, source leakage, timing noise, feature drift
- **source_id for leakage prevention**: why train/test must never share arrays from the same original time series — GroupShuffleSplit

### 3.5.1 Labelling Protocol
- **Timing methodology**: why best-of-K (reduces OS scheduling noise), K=5 for small arrays, K=3 for large
- **GC disabled**: why garbage collection creates timing artefacts
- **CPU isolation**: process pinning to avoid cache contention
- **Label assignment**: fastest algorithm = ground truth class label
- **Near-ties and label noise**: when two algorithms are within 5%, the label is inherently noisy — this is a fundamental challenge, not a bug
- **Class distribution**: natural balance achieved through domain diversity (18.2% introsort, 41.5% heapsort, 40.3% timsort)

### 3.6 Model Evolution (full subsections per version)
- 3.6.1 v1: Multi-output regression → failed (51%), learned size not algorithm preference
- 3.6.2 v2: Direct classifier on synthetic data → 62.5%, proved classification > regression
- 3.6.3 v3: Timing features added → 95% (data leakage), confirmed signal exists in features
- 3.6.4 v5: Production classifier on 1.18M real-world arrays → 76.1% accuracy, 93.1% gap closed
- 3.6.5 v6: Source-aware GroupShuffleSplit → 71.2%, stricter but more honest evaluation
- 3.6.6 v7: Regret-weighted training → rejected (gap closed dropped to 78%)
- 3.6.7 v8: Binary cascade → key finding: introsort ≈ heapsort (AUC 0.603), problem is binary

### 3.8 XGBoost Offline Classifier (v5)
- **Why XGBoost specifically** (over random forest, SVM, MLP, decision tree):
  - Gradient boosting: sequential ensemble that fits residuals — better than bagging (RF) for structured tabular data
  - Handles feature interactions naturally (e.g., size × sortedness)
  - Built-in L1/L2 regularisation prevents overfitting on noisy labels
  - Fast training via histogram-based splits (tree_method='hist')
  - Calibrated probability outputs via softmax (multi:softprob)
- **Class imbalance handling — 3-pronged strategy**:
  1. Noise filter: remove rows where best/2nd-best gap < 5% (unless n ≥ 2K) — these are inherently ambiguous labels
  2. Random undersampling: cap majority class at 3× minority count
  3. Inverse-frequency sample weights during training
  - Why this specific combination: noise filter addresses label quality, undersampling addresses quantity, weights address residual imbalance
- **Stratified split**: 70/15/15 (train/val/test) — why stratified (preserve class proportions in each split)
- **Hyperparameters with rationale** (table):
  - n_estimators=500, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
  - min_child_weight=5 (smoothing for noisy labels)
  - reg_alpha=0.1, reg_lambda=1.0 (L1+L2 regularisation)
  - Why these values: conservative depth to avoid memorising noise, slow learning rate for stability
- **Early stopping**: monitors validation mlogloss, prevents overfitting
- **Softmax objective**: multi:softprob produces probability estimates, not just hard predictions — enables confidence-based decisions

### 3.9 F1 Channel-Flag Multi-Model (Domain-Routed Selection) — Secondary experiment
- **ML technique: ensemble of specialised classifiers with metadata routing**
  - Instead of one global model, train K domain-specific models and route based on known metadata
  - Related to mixture-of-experts architecture but simpler: hard routing by flag, not learned gating
- **Why 5 algorithms here** (vs 3 in main): F1 data characteristics are different enough that quick_sort, merge_sort, and shell_sort become competitive — justify with empirical class distributions per channel
- Motivation: can domain metadata improve selection beyond structural features alone?
- Channel as routing flag: 9 F1 telemetry channels (Speed, RPM, Throttle, Distance, nGear, DRS, X, Y, Z)
- **Dynamic class sets**: some channels don't have enough samples for all 5 algorithms → only train on algorithms with ≥20 samples (avoids underfitting on rare classes)
- 17 features (16 structural + n_elements)
- **Optuna hyperparameter tuning**: Bayesian optimisation (TPE sampler), 20 trials per channel — why per-channel tuning matters (each channel has different data characteristics)
- Three iterations: baseline → dynamic v2 → Optuna-tuned
- Evaluation: routed accuracy vs channel-SBS vs global-SBS
- Results: 86.0% routed / 55.4% strict vs 50.7% channel-SBS vs 20.0% global-SBS
- **Deployment & Explainability (within this section):**
  - ONNX export pipeline (XGBoost → ONNX via onnxmltools, opset 15)
  - Interactive Svelte + WebSHAP demo with client-side SHAP explanations
  - ONNX Runtime Web for browser-based inference (no server needed)
  - Dual-mode UI: f1_routed (per-channel) vs v5_global (single model)
  - Figure: screenshot of the Svelte WebSHAP app

### 3.10 LinUCB Contextual Bandit — Framed as Future Work / Research Direction
- Why contextual bandits (not full RL) — theoretical justification
- Warm-start from XGBoost — design concept
- Exploration parameter α — design rationale
- Distribution shift evaluation design — proposed protocol
- **Note:** Present as a designed but not-yet-validated component; a future direction to pursue

### 3.11 Evaluation Protocol
- **Why standard accuracy is misleading for algorithm selection**: 76% accuracy sounds mediocre, but 89.6% of predictions have zero cost — the "errors" are mostly introsort↔heapsort confusion where both algorithms perform nearly identically
- **Classification metrics** (standard ML):
  - Top-1 accuracy, balanced accuracy, per-class precision/recall/F1
  - Confusion matrix analysis — where errors concentrate and why
- **Regret metrics** (algorithm selection specific — the real value measure):
  - VBS (Virtual Best Solver): oracle that always picks true fastest — theoretical lower bound
  - SBS (Single Best Solver): always use one algorithm — naive baseline upper bound
  - Model regret: (T_model − T_VBS) / T_VBS — how much slower than oracle
  - Gap closed: (T_SBS − T_model) / (T_SBS − T_VBS) — fraction of possible improvement captured
  - Zero-regret percentage: fraction of arrays with exactly correct prediction
  - Why regret matters more than accuracy: a "wrong" prediction that picks an algorithm 0.1% slower is fundamentally different from one that picks an algorithm 50% slower
- **Baseline comparisons**:
  - SBS (always heapsort — empirically best single algorithm)
  - Random selection (uniform random among 3 algorithms)
  - Why these baselines: SBS is the practical default, random is the information-free baseline
- **Domain holdout (leave-one-domain-out)**: train on 4 domains, test on unseen 5th — proves generalisation beyond memorising domain-specific patterns
- **Per-instance regret distribution**: not just mean regret but the full distribution — tail behaviour matters

### 3.12 Reproducibility
- Fixed seed, deterministic features
- Model serialisation (JSON, ONNX)
- Dataset CSV

### 3.13 Summary

---

## 5. Answered Questions

| Question | Answer |
|----------|--------|
| Chapter structure (3.1–3.12) | **Keep as is** — structure is good |
| Model evolution detail (3.6) | **Full subsections** — each version (v1, v2, v3, v5, v6, v7, v8) gets its own subsection |
| Structural transforms placement | **Under Data Collection** (3.4), not a separate subsection |
| F1 channel-flag multi-model prominence | **Secondary experiment** — important extension but secondary to main v5 pipeline |
| WebSHAP demo location | **In methodology (3.8)** — describe within the flag model section as deployment/explainability |
| LinUCB bandit depth | **Future work framing** — present as future goals / things to pursue, not a validated contribution |
| Figures/diagrams to include | **All**: architecture diagram, feature importance chart, model evolution table, flowchart, Svelte app screenshot |
| 5-algo vs 3-algo portfolio | **Justify both separately** — each portfolio justified in its own section (3.2 for main, 3.8 for flag model) |

## 6. Figures & Diagrams to Include

| Figure | Section | Description |
|--------|---------|-------------|
| **Figure 3.1** | 3.1 | System architecture diagram — two-layer design (offline XGBoost + online bandit), inference flow |
| **Figure 3.2** | 3.3 | Feature extraction flowchart — raw array → 16 features → feature vector |
| **Figure 3.3** | 3.6 | Model evolution timeline/table — v1→v8 progression with accuracy and key lessons |
| **Figure 3.4** | 3.7 | Feature importance bar chart — top features from v5 XGBoost |
| **Figure 3.5** | 3.8 | Channel-flag routing flowchart — array + channel flag → per-channel model → prediction |
| **Figure 3.6** | 3.8 | Screenshot of the Svelte WebSHAP app showing SHAP explanations |

---

## 7. Asset Preparation Checklist (MUST complete before writing)

> All figures as SVG/PNG generated by code. All metrics freshly extracted from models/results.

### 7.1 Figures to Generate

| # | Figure | Source | What to produce |
|---|--------|--------|-----------------|
| F1 | System architecture diagram | Design from PROJECT.md | SVG: two-layer design (raw array → feature extraction → XGBoost → prediction, with optional online bandit layer) |
| F2 | Feature extraction flowchart | feature_extraction.py | SVG: raw array → 7 feature groups → 16-dimensional vector |
| F3 | Model evolution timeline | THESIS-PROGRESS.md + results | SVG/PNG: v1→v8 progression with accuracy, key finding per version |
| F4 | Feature importance bar chart | v5 model | PNG: top-16 feature importances from XGBoost v5 |
| F5 | Confusion matrix (v5) | v5 predictions | PNG: 3×3 confusion matrix heatmap |
| F6 | Channel-flag routing flowchart | Flag model design | SVG: array + channel flag → router → per-channel model → prediction |
| F7 | Svelte WebSHAP app screenshot | Running demo | PNG: screenshot of the interactive demo |
| F8 | Data collection pipeline | Data scripts | SVG: 5 domains → fetch → transform → label → training_dataset.csv |
| F9 | Regret distribution | regret_analysis results | PNG: histogram of per-instance regret (showing 89.6% at zero) |

### 7.2 Metrics to Extract Fresh

| # | Metric | Source | What to extract |
|---|--------|--------|-----------------|
| M1 | v5 full evaluation | `results/xgboost_v5/evaluation_results.json` | Accuracy, balanced acc, per-class recall, confusion matrix |
| M2 | v5 regret analysis | `results/xgboost_v5/regret_analysis.json` | VBS, SBS, gap closed, zero-regret %, regret distribution |
| M3 | v5 feature importance | v5 model or results JSON | All 16 feature importances ranked |
| M4 | v5 hyperparameters | `scripts/train_xgboost_v5.py` | Exact params used |
| M5 | Dataset statistics | `data/training_dataset.csv` | Row count, class distribution, domain breakdown, array size stats |
| M6 | Domain holdout results | `results/domain_holdout/` or re-run | Per-domain accuracy and gap closed |
| M7 | v1–v8 accuracy summary | Various results files | Table of accuracy per version |
| M8 | Flag model per-channel results | `results/f1_9_channel_models_optuna/manifest.json` | Per-channel: rows, classes, test accuracy, balanced accuracy |
| M9 | Flag model router eval | `results/f1_9_channel_models_dynamic_v2/router_eval.json` | Routed vs channel-SBS vs global-SBS |
| M10 | Flag model strict eval | `results/f1_9_channel_models_dynamic_v2_strict/strict_router_eval.json` | Strict routed accuracy |

### 7.3 Scripts to Run

| # | Script / Action | Purpose |
|---|----------------|---------|
| S1 | Generate architecture diagram (new script) | Produce Figure F1 |
| S2 | Generate feature extraction flowchart (new script) | Produce Figure F2 |
| S3 | Generate model evolution visual (new script) | Produce Figure F3 |
| S4 | Extract v5 feature importance + generate bar chart | Produce Figure F4 |
| S5 | Generate v5 confusion matrix heatmap | Produce Figure F5 |
| S6 | Generate channel-flag routing flowchart (new script) | Produce Figure F6 |
| S7 | Take Svelte WebSHAP screenshot | Produce Figure F7 (may need manual or browser automation) |
| S8 | Generate data pipeline diagram (new script) | Produce Figure F8 |
| S9 | Generate regret distribution histogram | Produce Figure F9 |
| S10 | Collect all metrics into one summary JSON | Single source of truth for all numbers used in chapter |

### 7.4 Preparation Order

1. **Extract all metrics** (M1–M10) from existing results files → `results/methodology_assets/metrics_summary.json`
2. **Generate data-driven figures** (F4, F5, F9) — these need real numbers
3. **Design and generate diagrams** (F1, F2, F3, F6, F8) — these are conceptual
4. **Capture WebSHAP screenshot** (F7)
5. **Verify all assets** — review every figure and metric for accuracy
6. **Write the chapter** — all pieces ready, produce best output in one shot

---

## 8. Writing Style Guide (learned from Ahmed's Chapters 1 & 2)

> Every paragraph in the methodology MUST follow these patterns exactly. This is not a suggestion — it is a constraint.

### 8.1 Sentence & Paragraph Structure

- **Paragraph length**: 30–70 words typically. Never exceed 80 words. Never write one-sentence paragraphs unless it's a transitional sentence after a table/figure.
- **Sentences per paragraph**: 2–4 sentences. Most paragraphs have exactly 3 sentences.
- **Sentence length**: Mix short declarative sentences (10–15 words) with longer explanatory ones (25–35 words). Never exceed 40 words in a single sentence.
- **Opening sentences**: Start with a clear factual claim or context-setter, not a vague generality. Examples:
  - "Sorting is a foundational operation in computer science..."
  - "Modern computing workloads process arrays drawn from many domains..."
  - "Any comparison-based sorting algorithm must distinguish among the n! possible orderings..."
  - "The Algorithm Selection Problem (ASP) is the task of choosing..."
- **No filler openers**: Never start with "In this section, we will discuss..." or "It is important to note that..." — go straight to the substance.

### 8.2 Tone & Voice

- **Formal academic** but not stiff — reads like a confident researcher explaining to a knowledgeable reader.
- **Third person throughout**: "This thesis addresses..." not "We address..." or "I address..."
- **Active voice preferred** over passive when the subject is clear: "Introsort begins as quicksort and switches to heapsort" not "Quicksort is begun by introsort which is then switched to heapsort."
- **Assertive, not hedging**: "No single sorting algorithm is optimal across all inputs" — not "It might be the case that no single algorithm is always optimal."
- **Precise technical language**: Use exact terms (presortedness, distribution shift, label noise, regret, gap closed) — don't paraphrase them differently each time.
- **British English spelling**: behaviour, generalise, formalise, optimisation, minimise, defence, analysed.

### 8.3 Structural Patterns

- **Constraint-driven reasoning**: Problems are framed as constraints that conflict: "This requirement creates three interacting constraints. First... Second... Third..." — then explain the trade-off.
- **Justify then state**: Explain WHY before stating WHAT was done. The motivation always precedes the method.
- **"Synthesis" paragraphs**: End major subsections with a one-sentence synthesis that ties the subsection back to the thesis. Pattern: "Synthesis: [broader context]. This thesis [specific contribution]." Example: "Synthesis: most adaptive and hybrid sorting research focuses on improving the behaviour of a specific sorting algorithm. This thesis takes a different approach by using input features to select among several complete sorting algorithms before sorting begins."
- **Table integration**: Tables are introduced with a sentence ending in a table reference, and followed by a short interpretive paragraph. Pattern: "[Context sentence], as summarised in Table X.Y." → Table → "This mapping shows that [interpretation]."
- **Figure integration**: Figures referenced before they appear, explained after.
- **Cross-references**: Use explicit section references: "Section 2.13", "Table 2.1", "Equation 1". Never say "the table above" or "as shown below" without the numbered reference.

### 8.4 Formatting Rules (from CIU + Ahmed's existing chapters)

- **Headings**: ALL CAPS for chapter titles ("CHAPTER THREE"), Title Case for section headings ("3.1 System Architecture"), Title Case for subsections ("3.3.2 Sortedness Features")
- **Body text**: Times New Roman 12pt, 1.5 line spacing, justified alignment
- **Bullet points**: Used sparingly and only for research objectives. Body text is always in paragraph form — never use bullets to explain methodology.
- **Equations**: Inline for simple expressions (a* = argminₐ T(a, x)), displayed and numbered for key formulas
- **Mathematical notation**: Use Unicode symbols where possible (∈, →, ₐ, Ω), formal variable definitions before use ("let X denote...")
- **Tables**: Title above (bold Table X.Y:), 10pt content, source below if applicable
- **Figures**: Title below (bold Figure X.Y:), centered, source below if applicable
- **Citations**: APA in-text, minimal — only cite originator papers and key frameworks

### 8.5 Transition Patterns Between Paragraphs

- **However,** — to introduce a contrasting or complicating point
- **This requirement creates...** — to move from problem to constraints
- **From a technical perspective,** — to shift from conceptual to formal
- **The research problem is therefore...** — to synthesise into a clear statement
- **In the context of this thesis,** — to map a general concept to this specific work
- **Taken together,** — to summarise a group of items just discussed
- **[Topic] is relevant because...** — to justify inclusion of a concept

### 8.6 Things Ahmed NEVER Does

- Never uses "we" — always "this thesis", "the system", "the model"
- Never uses bullet points in explanatory text (only in research objectives list)
- Never writes one-line paragraphs for explanations
- Never starts a section with a meta-statement like "This section describes..."
- Never uses informal language ("basically", "a lot of", "kind of", "pretty much")
- Never uses emojis or exclamation marks
- Never repeats the same point in different words across consecutive paragraphs
- Never writes a paragraph that is purely definitional without connecting to the thesis purpose
- Never leaves a table or figure without an interpretive follow-up sentence
- **CRITICAL: Never uses em dashes (—) or en dashes (–) inside sentences.** This is the most common AI tell. Ahmed uses commas, semicolons, colons, and full stops to separate clauses. Never write "the model — which was trained on real data — achieved 76% accuracy." Write instead: "The model, which was trained on real data, achieved 76% accuracy." or split into two sentences. The only acceptable dash usage is in compound terms like "source-level" or "run-adaptive" (hyphens, not dashes).

### 8.7 Example Paragraph Anatomy (from Chapter 1, P38)

> "This requirement creates three interacting constraints. First, efficiency: feature extraction and prediction must be computationally lightweight. Second, latency: decisions must be produced fast enough to integrate into production pipelines. Third, resilience: the selector must remain effective when incoming data differs from the data used for training. These constraints often conflict in practice, since stronger predictive models tend to be more expensive to evaluate and may generalise poorly under distribution shift."

**Pattern**: Opening claim → Enumerated points (First... Second... Third...) → Closing sentence that creates tension/trade-off. 70 words. 5 sentences. Each enumerated point is a single clause, not a full paragraph.

### 8.8 Paragraph Length Distribution (Phase 2 Analysis)

- **Mean**: 39 words per paragraph (computed over 115 body paragraphs)
- **Median**: ~35 words
- **Range**: 1–88 words (the 1-word outlier is a table/figure label fragment, not a body paragraph)
- **Safe writing band**: 20–75 words. Keep most paragraphs in the 30–50 word range.
- **Hard ceiling**: Never exceed 90 words in a single paragraph. Ahmed's longest is 88 words.
- **Short paragraphs** (under 15 words): Only used for transitional sentences immediately after a table or figure, never for explanatory content.

### 8.9 First Word Patterns

Ahmed's paragraphs most frequently open with these words (descending frequency):

- **"This"** (11 times) — "This thesis...", "This requirement...", "This mapping..."
- **"The"** (10 times) — "The research problem...", "The features...", "The system..."
- **"Table"** (9 times) — Always introducing a table reference
- **"A"/"An"** (7 times) — "A sorting algorithm...", "An important distinction..."
- **"Sorting"/"Algorithm"** (5–6 times) — Topic-specific noun openers
- **"In"** (4 times) — "In this context,", "In the worst case,"
- **"For"** (3 times) — "For example,", "For a given input,"
- **"However,"** (3 times) — Always at paragraph start for contrast

**Rules for Chapter 3:**
- Favour "This", "The", or a topic-specific noun as the opening word.
- Never start consecutive paragraphs with the same word.
- Never start with "It is", "There are", "We", or "I".
- "However," and "In" are acceptable but should be used at most once per subsection.

### 8.10 Mathematical Notation Formatting

- **Variables and expressions** in body text are formatted as **bold+italic**: e.g., ***A = {a₁, a₂, ..., aₖ}***, ***f(x)***, ***S: f(x) → A***
- **Formal definitions** appear as indented paragraphs (List Paragraph style), not as numbered equations, when they are conceptual frameworks (e.g., Rice's ASP definition).
- **Numbered equations** are reserved for key computational formulas that are referenced later (e.g., regret formula, entropy formula). They are centred with right-aligned numbering: (3.1), (3.2), etc.
- **Greek letters and subscripts**: Use Unicode where possible (α, β, Ω, ₐ, ₓ). Spell out in running text when first introduced: "the exploration parameter alpha (α)".

### 8.11 Heading Hierarchy and Capitalisation

- **Heading 1** (Chapter title): ALL CAPS, centred, bold, 14pt. Example: "CHAPTER THREE" followed by "METHODOLOGY" on a new line.
- **Heading 2** (Major sections): Title Case with section number, bold, left-aligned, 12pt. Example: "3.1 System Architecture Overview"
- **Heading 3** (Subsections): Title Case with sub-number, bold, left-aligned, 12pt. Example: "3.3.2 Sortedness Features"
- **No Heading 4**: Ahmed never goes deeper than three levels. If further subdivision is needed, use a bold lead-in phrase within a paragraph: "**Feature normalisation.** The raw counts are..."

### 8.12 Table and Figure Caption Formatting

- **Table captions**: Above the table. Format is bold "Table X.Y:" followed by descriptive title in normal weight. Example: "**Table 3.1:** Structural Features Used for Array Characterisation". The caption uses the Word "Caption" style, centred.
- **Figure captions**: Below the figure. Same bold numbering pattern: "**Figure 3.1:** Overall System Architecture". Centred, Caption style.
- **Source lines**: Below figures/tables when content is adapted from another source: "Source: Adapted from [Author, Year]". Not needed for original content.
- **Table content**: 10pt Times New Roman, single-spaced within cells. Column headers bold. Left-aligned for text columns, right-aligned for numeric columns.

### 8.13 List Paragraph and Enumeration Rules

- **List Paragraph style** (indented, possibly numbered): Used ONLY for formal framework definitions (e.g., Rice's four spaces P, A, Y, S) and research objectives/questions. Never for methodology explanations.
- **Inline enumeration**: Ahmed uses "First, ... Second, ... Third, ..." within a paragraph rather than bullet points. This is the preferred method for listing methodology steps, constraints, or reasons.
- **Semicolon-separated series**: For longer items in an inline list, Ahmed uses semicolons: "the features capture three properties: sortedness, which measures how close the array is to sorted order; distribution shape, which captures skewness and kurtosis; and repetition structure, which quantifies duplicate density."
- **Parenthetical clarifications**: 28% of Ahmed's paragraphs contain parenthetical asides. These are used for units, acronyms, brief examples, or alternative terms: "(e.g., O(n log n))", "(ASP)", "(370 values per lap)".

### 8.14 Punctuation Patterns

- **Semicolons**: Rare (only 4% of paragraphs). Used to join closely related independent clauses, never for casual separation.
- **Colons**: Moderate (21% of paragraphs). Used to introduce a definition, list, or elaboration. Always preceded by a complete clause.
- **Em/en dashes**: Essentially absent (under 3% and those are formatting artefacts). NEVER use them.
- **Commas**: Primary clause separator. Ahmed uses Oxford commas in series.
- **Hyphens**: Only for compound modifiers ("real-world", "source-level", "run-adaptive", "time-stamped") and compound nouns ("trade-off", "best-in-class"). Never as sentence-level punctuation.

### 8.15 Additional Patterns Extracted from Version-6 Chapter 3

> These patterns were identified by analysing the final, Ahmed-edited version-6 of Chapter 3, which represents his most refined writing. They supplement the original patterns from Chapters 1 and 2.

**Paragraph length shift.** Version-6 Chapter 3 paragraphs are longer than Chapters 1-2. Mean is 70.6 words (vs 39 in Chapters 1-2), median is 71. Ahmed's methodology paragraphs pack more technical content per paragraph. The safe band for Chapter 3+ is **50-120 words**, with occasional shorter transitional paragraphs (under 30 words) after figures or tables.

**Sentence density.** Most paragraphs contain 3-5 sentences. The opening paragraph of each section tends to be the longest (100-140 words, 5+ sentences), establishing context before the subsections provide detail.

**"The" dominance.** 8 of 32 body paragraphs (25%) start with "The". This is Ahmed's default paragraph opener for Chapter 3. Pattern: "The [noun phrase] [verb] [technical detail]." Examples from version-6:
- "The runtime system processes each array in three stages..."
- "The portfolio comprises three comparison-based algorithms..."
- "The training corpus contains 1,188,265 numeric arrays..."
- "The production classifier is the final iteration of..."
- "The raw label distribution is severely skewed..."

**Semicolons are common in methodology (not rare).** Contrary to the Chapters 1-2 analysis (4%), version-6 Chapter 3 uses semicolons in 28% of paragraphs (9/32). They serve two purposes:
1. Joining closely related technical clauses: "the selector is most beneficial for medium-to-large arrays; on very short arrays the selection overhead can exceed any achievable runtime saving."
2. Separating items in inline enumerations: "RAW preserves the original ordering; REV fully reverses the array; SHUF applies a uniform random permutation; QBIN50 quantises..."

**Colons introduce technical specifications.** 31% of paragraphs use colons, always preceded by a complete clause. Pattern: "[Context clause]: [technical detail or enumeration]." Example: "The raw label distribution is severely skewed: timsort wins on 85.0% of arrays, heapsort on 10.5% and introsort on 4.4%."

**"because" as causal connector.** Ahmed frequently uses "because" mid-sentence to provide immediate justification rather than splitting into two sentences. Example: "Standard classification metrics are insufficient for algorithm selection because they treat all misclassifications equally."

**Paragraph ending patterns.** Body paragraphs end with one of three patterns:
1. Forward reference: "...as summarised in Table 3.1." / "Figure 3.2 summarises the data pipeline."
2. Consequence or implication: "The architecture is therefore deployed with a length threshold below which a default algorithm is used directly."
3. Thesis connection: "...so no single member of the portfolio dominates across realistic workloads and a feature-driven selector is potentially useful."

**Mixed voice in version-6.** Ahmed uses more passive constructions in Chapter 3 than in earlier chapters (25% of paragraphs). Passive is acceptable when the subject is a technical process: "Sixteen features are computed and organised into five groups." Active remains preferred when the subject is clear: "The system extracts sixteen features."

**Figure/table integration (version-6 refined pattern).** Ahmed now uses a tighter integration than in Chapters 1-2:
- Pre-reference: "...illustrated in Figure 3.1." or "...as summarised in Table 3.1."
- The figure/table appears immediately after.
- Post-interpretation starts with a specific finding, not a meta-statement: "The length normalisation feature contributes 26.2% of total gain..." (not "Figure 3.3 shows that...").

**Section opening pattern.** Sections open with a context-setting sentence that names the topic and connects it to the broader system, not a meta-statement. Correct: "Feature engineering transforms the raw array into a fixed-length vector that captures the structural properties relevant to sorting performance." Wrong: "This section describes the feature engineering process."

**Inline enumeration with "First... Second... Third..."** Used within a single paragraph for constraints or mechanisms. Each item is a clause, not a full sentence. Pattern preserved from Chapters 1-2 but used more densely in Chapter 3.

---

## 9. Resolved Questions

| Question | Answer |
|----------|--------|
| Citations approach | **Minimal** — only essentials (Rice 1976, XGBoost/Chen & Guestrin 2016, LinUCB/Li et al. 2010, algorithm originators: Musser 1997, Williams 1964, Peters 2002) |
| Chapter number | **Chapter 3** (Ch1 Intro → Ch2 Background → Ch3 Methodology) |
| Missing work? | **Yes — Ahmed has more to describe** (awaiting details) |

---

## 10. Files to Reference During Writing

| File | What it contains |
|------|-----------------|
| `scripts/feature_extraction.py` | Exact feature implementation (16 features) |
| `scripts/train_xgboost_v5.py` | v5 training pipeline, hyperparameters, balancing |
| `scripts/regret_analysis.py` | VBS/SBS regret computation |
| `scripts/train_linucb.py` | LinUCB bandit implementation |
| `docs/feature-definitions.md` | Full feature documentation |
| `docs/feature-validation-report.md` | 214 tests, edge cases |
| `docs/checkpoint-xgboost-v5-baseline.md` | v5 full summary |
| `docs/xgboost-v8-binary-cascade-report.md` | Binary cascade finding |
| `results/xgboost_v5/evaluation_results.json` | v5 metrics |
| `results/xgboost_v5/regret_analysis.json` | Regret numbers |
| `PROJECT.md` | Architecture decisions |
| `THESIS-PROGRESS.md` | Full execution log |
| `data-2026-04-13.md` | Data summary |
| `CIU Thesis Writing Guidelines.pdf` | University formatting rules |
| **F1 Channel-Flag Multi-Model files** | |
| `scripts/train_f1_9_channel_models.py` | Baseline 9-channel model training |
| `scripts/train_f1_9_channel_models_dynamic_v2.py` | Dynamic class set version |
| `scripts/tune_f1_9_channel_models_optuna.py` | Optuna hyperparameter tuning (20 trials/channel) |
| `scripts/train_eval_f1_dynamic_router_v2_strict.py` | Strict train/eval protocol |
| `scripts/eval_f1_dynamic_router_v2.py` | Router vs baselines evaluation |
| `scripts/predict_f1_by_channel_flag.py` | CLI inference by channel flag |
| `scripts/export_webshap_sorting_assets.py` | XGBoost → ONNX export + JSON metadata |
| `scripts/generate_f1_dynamic_figures.py` | Visualization figure generation |
| `models/f1_9_channel_models_optuna/` | 9 Optuna-tuned channel models (JSON) |
| `results/f1_9_channel_models_optuna/manifest.json` | Per-channel metrics, classes, best params |
| `results/f1_9_channel_models_dynamic_v2/manifest.json` | Dynamic v2 metrics |
| `results/f1_9_channel_models_dynamic_v2/router_eval.json` | Router accuracy: 86.0% |
| `results/f1_9_channel_models_dynamic_v2_strict/strict_router_eval.json` | Strict router: 48.9% |
| `docs/per_channel_training_notes.md` | Training notes, class counts, data sources |
| `webshap/examples/demo/` | Svelte + WebSHAP interactive demo |
| `webshap/examples/demo/public/models/sorting/*.onnx` | 9 ONNX channel models + v5_global.onnx |
| `webshap/examples/demo/public/data/sorting-f1.json` | Demo config: features, channels, examples, metrics |
