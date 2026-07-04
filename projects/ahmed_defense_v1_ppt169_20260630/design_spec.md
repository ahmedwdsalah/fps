# ahmed_defense_v1_ppt169_20260630 - Design Spec

## I. Project Information

| Item | Value |
| ---- | ----- |
| Project Name | Ahmed MSc Defense - Sorting Algorithm Selection |
| Canvas Format | PPT 16:9 (1280 x 720) |
| Page Count | 22 total defense slides. No backup deck, no appendix, no extra slides after Q&A. |
| Design Style | swiss-minimal |
| Target Audience | Mixed computer science and engineering MSc thesis defense committee at Cyprus International University |
| Use Case | Official MSc thesis defense presentation |
| Delivery Purpose | presentation |
| Content Strategy | Use the locked Spoken Main Slide Plan in `pptx.md`. Keep facts sourced from the submitted thesis and final JSON artifacts. Visible slide wording must be concise Ahmed-style defense claims, not internal planning language. |
| Created Date | 2026-06-30 |

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| Format | PPT 16:9 |
| Dimensions | 1280 x 720 |
| viewBox | `0 0 1280 720` |
| Margins | 56 left/right, 48 top, 40 bottom |
| Content Area | 1168 x 612 |

## III. Visual Theme

### Theme Style

- Mode: narrative
- Visual style: swiss-minimal
- Theme: light academic
- Tone: formal, calm, technical, evidence-led

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| Background | `#FFFFFF` | Slide background |
| Secondary bg | `#F6F8FB` | Quiet panels and chart surfaces |
| Primary | `#173B6D` | Titles, main rule lines, key icons |
| Accent | `#0F766E` | Runtime value and positive evidence |
| Secondary accent | `#D99A20` | Caution, limitation, method contrast |
| Body text | `#172033` | Main text |
| Secondary text | `#586174` | Notes, captions, axis labels |
| Tertiary text | `#7A8495` | Footer and low-emphasis labels |
| Border/divider | `#D8DEE8` | Rules and panel borders |
| Grid | `#E8EDF4` | Chart grid lines |
| Warning | `#B42318` | Strict-check caveats |

## IV. Typography System

### Font Plan

Typography direction: strict sans academic, readable from a projected defense room.

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| Title | Microsoft YaHei | Arial | sans-serif |
| Body | Microsoft YaHei | Arial | sans-serif |
| Emphasis | Microsoft YaHei | Arial | sans-serif |
| Code | - | Consolas, Courier New | monospace |

Per-role font stacks:

- Title: `Arial, 'Microsoft YaHei', sans-serif`
- Body: `Arial, 'Microsoft YaHei', sans-serif`
- Emphasis: `Arial, 'Microsoft YaHei', sans-serif`
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

- Body: 32
- Title: 52
- Subtitle: 38
- Lead: 34
- Subheading: 30
- Annotation: 22
- Chart annotation: 18
- Footnote: 16
- Cover title: 58
- Hero number: 72

## V. Layout Principles

### Page Structure

- Header: compact title area with a thin primary or accent rule; no heavy branding except title and closing.
- Content: one central evidence object per slide, supported by short spoken-message text.
- Footer: small page number and thesis label when useful; avoid clutter.

### Layout Pattern Library

- Breathing slides use one strong statement, one visual anchor, or one large number.
- Dense slides use clean two-column evidence layouts, figure plus short interpretation, or compact comparison rows.
- Charts from thesis assets are treated as evidence objects and framed lightly; do not redraw results unless the existing figure is unreadable.
- No decorative AI images. No ornamental gradients, floating blobs, or stock-style backgrounds.

### Spacing Specification

- Safe margin: 56 left/right, 48 top, 40 bottom.
- Content block gap: 28-40.
- Icon-text gap: 10-14.
- Card radius: 6-8 when cards are necessary.
- Use dividers and whitespace before extra boxes.

## VI. Icon Usage Specification

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| Task framing | `tabler-outline/target` | P02, P05 |
| Runtime result | `tabler-outline/gauge` | P14, P15 |
| Dataset | `tabler-outline/database` | P09 |
| System route | `tabler-outline/route` | P06, P07 |
| Rigor | `tabler-outline/shield-check` | P19 |
| Model | `tabler-outline/brain` | P12 |
| Features | `tabler-outline/filter` | P11 |
| Evolution | `tabler-outline/timeline` | P13 |
| Q&A | `tabler-outline/question-mark` | P22 |

## VII. Visualization Reference List

This deck uses provided thesis figures as chart evidence rather than chart-library templates.

| Page | Visualization | Source | Usage |
| ---- | ------------- | ------ | ----- |
| P06 | Feature extraction diagram | `images/F2_feature_extraction.png` | Show cheap structural signals before sorting |
| P07 | System architecture | `images/F1_system_architecture.png` | Show one-decision selector pipeline |
| P09 | Data pipeline | `images/F8_data_pipeline.png` | Explain domains, transforms, labels |
| P13 | Model evolution | `images/F3_model_evolution.png` | Compact research journey |
| P15 | Regret distribution | `images/F9_regret_distribution.png` | Explain why accuracy is incomplete |
| P16 | Confusion matrix | `images/F5_confusion_matrix.png` | Show timsort clarity and intro/heap boundary |
| P17 | Feature importance | `images/F4_feature_importance.png` | Show structural feature logic |
| P18 | Domain holdout | `images/F10_domain_holdout_gap_closed.png` | Show transfer where structure transfers |
| P19 | v5/v6 comparison | `images/10_v5_vs_v6_comparison.png` | Show strict checks and interpretation limits |

Runners-up considered:

- `kpi_cards`: rejected because the deck uses thesis result figures and large evidence moments, not generic KPI cards.
- `process_flow`: rejected because the existing architecture and pipeline figures already express the method.
- `grouped_bar_chart`: rejected because thesis chart images preserve the submitted result styling and source trace.

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Acquire Via | Status | Reference | text_policy | page_role |
| -------- | ---------- | ----- | ------- | ---- | -------------- | ----------- | ------ | --------- | ----------- | --------- |
| F1_system_architecture.png | 3030 x 1830 | 1.66 | System architecture evidence | Thesis figure | figure + interpretation | provided | ready | methodology asset | embedded | evidence |
| F2_feature_extraction.png | 3030 x 2130 | 1.42 | Feature extraction evidence | Thesis figure | figure + interpretation | provided | ready | methodology asset | embedded | evidence |
| F3_model_evolution.png | 3023 x 1539 | 1.96 | Model journey evidence | Thesis figure | wide figure | provided | ready | methodology asset | embedded | evidence |
| F4_feature_importance.png | 2400 x 1650 | 1.45 | Feature importance evidence | Thesis figure | chart + takeaway | provided | ready | methodology asset | embedded | evidence |
| F5_confusion_matrix.png | 1650 x 1350 | 1.22 | Confusion matrix evidence | Thesis figure | chart + takeaway | provided | ready | methodology asset | embedded | evidence |
| F8_data_pipeline.png | 3030 x 1680 | 1.80 | Data and labeling pipeline | Thesis figure | wide figure | provided | ready | methodology asset | embedded | evidence |
| F9_regret_distribution.png | 2700 x 1200 | 2.25 | Regret distribution evidence | Thesis figure | wide chart | provided | ready | methodology asset | embedded | evidence |
| F10_domain_holdout_gap_closed.png | 2370 x 1469 | 1.61 | Domain holdout evidence | Thesis figure | chart + takeaway | provided | ready | methodology asset | embedded | evidence |
| F11_accuracy_vs_gap_closed.png | 2370 x 1463 | 1.62 | Accuracy versus runtime value | Thesis figure | chart + takeaway | provided | ready | methodology asset | embedded | evidence |
| 06_vbs_sbs_model_comparison.png | 2963 x 2065 | 1.43 | VBS/SBS/model comparison | Thesis figure | chart + takeaway | provided | ready | results figure | embedded | evidence |
| 10_v5_vs_v6_comparison.png | 2963 x 1762 | 1.68 | Source-aware check evidence | Thesis figure | chart + takeaway | provided | ready | results figure | embedded | evidence |
| formula_regret.png | 514 x 109 | 4.72 | Regret formula | Latex formula | formula block | formula | rendered | formula manifest | none | equation |
| formula_gap_closed.png | 612 x 109 | 5.61 | Gap-closed formula | Latex formula | formula block | formula | rendered | formula manifest | none | equation |

## IX. Content Outline

P01. Machine Learning Based Algorithm Selection for Sorting Numeric Arrays
- Central visual/evidence: title composition with three-result strip.
- Purpose: establish thesis, candidate, supervisor, institution, date.
- Speaker intent: open calmly; say this defense is about choosing a sorting algorithm before sorting begins.
- Do not say: long abstract, full literature survey, backup framing.

P02. The problem is choosing the right algorithm for each input
- Central visual/evidence: simple decision route from array to algorithm.
- Purpose: make algorithm selection concrete.
- Speaker intent: sorting is familiar, but the choice is input-dependent.
- Do not say: generic complexity lecture.

P03. Same complexity does not mean same runtime
- Central visual/evidence: contrast between equal Big-O and different measured winners.
- Purpose: explain why this task matters.
- Speaker intent: asymptotic complexity does not remove practical runtime differences.
- Do not say: unrelated algorithm theory details.

P04. This thesis selects before sorting begins
- Central visual/evidence: array features -> selector -> algorithm.
- Purpose: introduce the central idea.
- Speaker intent: I do not sort multiple times and then choose; I use cheap structure first.
- Do not say: claim oracle perfection.

P05. The aim is a practical selector, not a perfect oracle
- Central visual/evidence: objective box with accuracy, regret, gap closed.
- Purpose: set evaluation discipline early.
- Speaker intent: the model can be useful even if some labels are missed, because not every mistake costs runtime.
- Do not say: accuracy alone is success.

P06. The array is converted into structural signals
- Central visual/evidence: `F2_feature_extraction.png`.
- Purpose: show how inputs become features.
- Speaker intent: the features describe size, repetition, ordering, and distribution.
- Do not say: every formula in detail.

P07. The system makes one decision before execution
- Central visual/evidence: `F1_system_architecture.png`.
- Purpose: explain the system as a practical pipeline.
- Speaker intent: feature extraction is followed by one XGBoost decision among three algorithms.
- Do not say: deployment claims beyond thesis scope.

P08. The algorithms are different enough to make selection useful
- Central visual/evidence: three algorithm comparison row.
- Purpose: justify portfolio choice.
- Speaker intent: introsort, heapsort, and timsort react differently to structure.
- Do not say: defend every possible sorting algorithm.

P09. Real arrays give the selector the structures synthetic data missed
- Central visual/evidence: `F8_data_pipeline.png` plus five-domain count.
- Purpose: explain 1.18M arrays and five domains.
- Speaker intent: the project moved from small synthetic experiments to real arrays and transformations.
- Do not say: all domain collection details.

P10. The labels come from measured runtime, not opinion
- Central visual/evidence: timing and labeling protocol diagram.
- Purpose: explain measured fastest-algorithm labels.
- Speaker intent: each training label comes from benchmarked runtime, then noisy cases are filtered.
- Do not say: hand-wave measurement noise.

P11. The features describe structure, not raw values
- Central visual/evidence: four feature groups with example features.
- Purpose: make the 16 features defendable.
- Speaker intent: the model sees structure that sorting algorithms care about, not the original array as a sequence.
- Do not say: every formula unless asked.

P12. The main model is the general v5 selector
- Central visual/evidence: model card with XGBoost, 16 features, 3 labels, split protocol.
- Purpose: separate v5 general model from F1-specific work.
- Speaker intent: v5 is the main thesis model; F1 channel routing is a separate specialization.
- Do not say: mix 3-label v5 with 5-label F1 routing.

P13. Each failed path clarified the final design
- Central visual/evidence: `F3_model_evolution.png`.
- Purpose: show journey without turning it into a diary.
- Speaker intent: regression, leakage-prone timing features, regret weighting, and cascade attempts all shaped the final interpretation.
- Do not say: over-focus on old models.

P14. The model is imperfect, but most mistakes are cheap
- Central visual/evidence: large 93.1%, 76.1%, 89.6%.
- Purpose: deliver the main result.
- Speaker intent: lead with runtime value; accuracy is part of the story, not the whole story.
- Do not say: perfect classifier.

P15. Accuracy alone does not explain the value
- Central visual/evidence: `F9_regret_distribution.png` and formulas.
- Purpose: define regret, SBS, VBS, and gap closed.
- Speaker intent: show why a 76.1% classifier can close most of the runtime gap.
- Do not say: too much math.

P16. Timsort is structurally visible; introsort vs heapsort is not
- Central visual/evidence: `F5_confusion_matrix.png`.
- Purpose: explain per-class behavior.
- Speaker intent: timsort recall is strong; the main confusion is between introsort and heapsort.
- Do not say: hide the weakness.

P17. The model mostly uses features that make sorting sense
- Central visual/evidence: `F4_feature_importance.png`.
- Purpose: interpret feature importance.
- Speaker intent: length, repetition, and runs dominate, matching the algorithm-selection logic.
- Do not say: causal proof from feature importance.

P18. The selector transfers when structure transfers
- Central visual/evidence: `F10_domain_holdout_gap_closed.png`.
- Purpose: show domain holdout evidence.
- Speaker intent: generalization is not uniform, but the runtime value survives across several domains.
- Do not say: universal domain generalization.

P19. Strict checks changed the interpretation, not the contribution
- Central visual/evidence: `10_v5_vs_v6_comparison.png` plus v7/v8 note.
- Purpose: handle rigor and negative results inside the 22-slide story.
- Speaker intent: stricter checks lower numbers and reveal boundaries, but they also make the conclusion more honest.
- Do not say: source-aware checks were ignored or that v7/v8 succeeded.

P20. The remaining errors show where the features stop seeing
- Central visual/evidence: boundary diagram for introsort/heapsort and F1 caveat.
- Purpose: state limitations without weakening the whole defense.
- Speaker intent: current features cannot fully separate some close algorithms; F1 routing is specialization, not the main result.
- Do not say: turn F1 into the main contribution.

P21. The contribution is a selector and an evaluation discipline
- Central visual/evidence: three contribution statements.
- Purpose: summarize contributions clearly.
- Speaker intent: contribution is not only model accuracy; it is dataset construction, structural features, and runtime-aware evaluation.
- Do not say: introduce new results.

P22. The next step is hardware-aware and adaptive selection
- Central visual/evidence: closing path to hardware-aware and adaptive work.
- Purpose: close and invite questions.
- Speaker intent: end with future work and readiness for questions.
- Do not say: validated LinUCB.

## X. Speaker Notes Strategy

- Notes are rehearsal-script style, written in natural Ahmed defense voice.
- Notes carry the details; slides stay sparse.
- For difficult questions, acknowledge the concern first, then answer with evidence.
- Use measured confidence: lead with runtime value, then clearly name limits.
- Avoid internal planning words in visible slide text and notes.

## XI. Verification Plan

- Main deck must remain exactly 22 slides.
- No standalone literature survey.
- No backup deck, appendix, or hidden second track.
- Every headline number traces to `methodology_assets/metrics_summary.json`.
- v5 general selector and F1-specific routing remain separated.
- Generated PPTX opens; all images, icons, and formulas render.
