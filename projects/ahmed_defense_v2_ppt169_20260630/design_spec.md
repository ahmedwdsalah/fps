# ahmed_defense_v2_ppt169_20260630 - Design Spec

## I. Project Information

| Item | Value |
| ---- | ----- |
| Project Name | Ahmed MSc Defense V2 - Sorting Algorithm Selection |
| Canvas Format | PPT 16:9 (1280 x 720) |
| Page Count | 22 slides total. No backup deck, no appendix, no extra slides after the closing/Q&A slide. |
| Design Style | Anthropic brand identity fused with academic_defense layout; visual style `swiss-minimal` |
| Target Audience | Mixed computer science and engineering MSc thesis defense committee at Cyprus International University |
| Use Case | Official MSc thesis defense presentation |
| Delivery Purpose | presentation |
| Content Strategy | Restructure freely inside the submitted thesis evidence. Use a script-first defense flow and keep every fact sourced. No standalone literature survey, no backup appendix, no machine planning words on slides. |
| Created Date | 2026-06-30 |

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| Format | PPT 16:9 |
| Dimensions | 1280 x 720 |
| viewBox | `0 0 1280 720` |
| Margins | Template safe area: x 40-1240, y 70-665 |
| Content Area | Header 0-70, key message 70-120, content 135-650, footer 665-720 |

## III. Visual Theme

### Theme Style

- Mode: narrative
- Visual style: swiss-minimal
- Theme: light academic
- Tone: formal, restrained, evidence-led, defense-ready

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| Background | `#FFFFFF` | Page field |
| Secondary bg | `#F8FAFC` | Template key-message bars, panels, figure surfaces |
| Primary | `#191919` | Headers, main text, strong rules |
| Accent | `#D97757` | Anthropic orange; section markers, key emphasis |
| Secondary accent | `#4A90D9` | Process arrows, secondary data highlights |
| Body text | `#191919` | Main body text |
| Secondary text | `#64748B` | Captions, source text, slide numbers |
| Tertiary text | `#94A3B8` | Low-emphasis footers |
| Border/divider | `#E2E8F0` | Figure frames and grid rules |
| Surface | `#F8FAFC` | Light panel fill |
| Grid | `#E8EEF5` | Chart grid lines |
| Success | `#10B981` | Positive evidence |
| Warning | `#EF4444` | Risks and limitations |

Rules:
- Do not use decorative short underlines under titles. Use the template header system, full-width bars, or no rule.
- Use Anthropic orange sparingly; it marks structure and emphasis, not decoration.
- Avoid generic card grids. Use the academic template frame plus selected diagram/chart structures.

## IV. Typography System

### Font Plan

Typography direction: Anthropic-style sans with safe fallbacks. Official Anthropic typefaces may require local install or PPTX embedding; Arial and Microsoft YaHei remain the safe fallback.

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| Title | Microsoft YaHei | Styrene A, Helvetica Neue, Arial | sans-serif |
| Body | Microsoft YaHei | Anthropic Sans, Helvetica Neue, Arial | sans-serif |
| Emphasis | Microsoft YaHei | Arial | sans-serif |
| Code | - | Consolas, Courier New | monospace |

Per-role font stacks:

- Title: `"Styrene A", "Helvetica Neue", Arial, "Microsoft YaHei", sans-serif`
- Body: `"Anthropic Sans", "Helvetica Neue", Arial, "Microsoft YaHei", sans-serif`
- Emphasis: `Arial, "Microsoft YaHei", sans-serif`
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

- Body: 32
- Title: 48
- Subtitle: 36
- Lead: 34
- Subheading: 30
- Annotation: 21
- Chart annotation: 18
- Footnote: 16
- Cover title: 58
- Hero number: 78

## V. Layout Principles

### Page Structure

- Header: template-derived academic header, adapted to Anthropic identity. Titles must wrap inside the safe area.
- Key message bar: one spoken claim or transition sentence. It must help Ahmed speak, not duplicate the title.
- Content area: one main evidence object, one diagram, or one structured comparison.
- Footer: source reference and page number only when useful.

### Layout Pattern Library

- Cover and closing inherit academic_defense anchor structure.
- Most content pages inherit `03_content.svg` as a page shell, then customize the content region.
- Process and methodology pages use chart/diagram templates: `pipeline_with_stages`, `vertical_pillars`, `layered_architecture`, `chevron_process`, `kpi_cards`, `pros_cons_chart`.
- Thesis figures are evidence, not decoration. They must be framed, sized to native ratio, and paired with a concise interpretation.
- No text may extend beyond safe area or container boundaries. Long titles are wrapped or shortened before SVG writing.

### Spacing Specification

- Safe margin: template x 40-1240, with 54 internal padding in dense slides.
- Content block gap: 24-36.
- Card gap: 20.
- Card padding: 20-26.
- Card border radius: 8 maximum.
- Header title max width: 1080; wrap before overflow.

## VI. Icon Usage Specification

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| Defense route | `tabler-outline/presentation` | P02 |
| Task/objective | `tabler-outline/target` | P03, P06 |
| Dataset | `tabler-outline/database` | P07 |
| Feature groups | `tabler-outline/filter` | P09, P10 |
| System selector | `tabler-outline/route` | P11 |
| Model | `tabler-outline/brain` | P12, P13 |
| Evaluation metrics | `tabler-outline/function` | P14 |
| Runtime result | `tabler-outline/gauge` | P15 |
| Result charts | `tabler-outline/chart-bar` | P16-P19 |
| Rigor and limits | `tabler-outline/shield-check`, `tabler-outline/alert-triangle` | P20-P21 |
| Questions | `tabler-outline/question-mark` | P22 |

## VII. Visualization Reference List

Catalog read: 71 templates

| Page | Template | Path | Summary-quote | Usage |
| ---- | -------- | ---- | ------------- | ----- |
| P02 | agenda_list | `templates/charts/agenda_list.svg` | "Pick for table of contents, meeting agendas, or presentation roadmap — numbered items + brief description + duration / owner per row. Skip for substantive content lists (use vertical_list) or single-page section dividers (use a cover layout)." | Defense route / spoken sections |
| P08 | pipeline_with_stages | `templates/charts/pipeline_with_stages.svg` | "Pick for 3-5 horizontal pipeline stages, each = title + 1-line description + output artifact, connected by arrows (data pipelines, ETL, build pipelines). Skip if any stage lacks an artifact (use process_flow or numbered_steps)." | Transform and labeling protocol |
| P09 | vertical_pillars | `templates/charts/vertical_pillars.svg` | "Pick for 1×3 / 1×4 / 1×5 vertical column layout where each pillar = one independent category with title + bullets — PEST (Political/Economic/Social/Technological), four-pillar strategy overview, side-by-side independent categories. Skip for 2×2 quadrant (use quadrant_text_bullets), pricing tiers (use comparison_columns), or 2×2 parallel aspects (use labeled_card)." | Feature groups |
| P11 | layered_architecture | `templates/charts/layered_architecture.svg` | "Pick for 3-4 horizontal architecture layers (presentation/service/data), 2-4 module cards per layer, each card = title + 1-line description (description required, even if source brief). Skip if no per-module descriptions (use icon_grid) or no horizontal layering (use module_composition)." | General selector system |
| P13 | chevron_process | `templates/charts/chevron_process.svg` | "Pick for 3-6 phase methodology with chunky arrow-chain progression and deliverables per phase. Skip for <=2 phases or non-linear flow (use process_flow), or chain ending in an aggregate outcome wedge (use chevron_chain_with_tail)." | Model evolution / design lessons |
| P15 | kpi_cards | `templates/charts/kpi_cards.svg` | "Pick for 4-8 standalone numeric metrics shown as overview cards (2x2 or 1x4) — exec summary opener, dashboard headline, quarterly recap, results-at-a-glance. Skip if metrics have target baselines (use bullet_chart) or single hero number (use gauge_chart)." | Main result trio |
| P20 | pros_cons_chart | `templates/charts/pros_cons_chart.svg` | "Pick for bilateral pros/cons list, 2-5 items per side. Skip for full feature comparison (use comparison_table) or numeric A/B mirror data (use butterfly_chart)." | Strict checks and rejected routes |

Runners-up considered:

- `process_flow` | rejected for P08/P11 because those slides need named artifacts and layered structure, not generic arrows.
- `comparison_table` | rejected for P20 because the point is evidence/interpretation, not a dense feature matrix.
- `timeline` | rejected for P13 because the journey should show method lessons, not dates.

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Acquire Via | Status | Reference | text_policy | page_role |
| -------- | ---------- | ----- | ------- | ---- | -------------- | ----------- | ------ | --------- | ----------- | --------- |
| F1_system_architecture.png | 3030 x 1830 | 1.66 | Selector architecture evidence | Thesis figure | architecture figure + interpretation | user | ready | methodology asset | embedded | evidence |
| F2_feature_extraction.png | 3030 x 2130 | 1.42 | Feature extraction detail | Thesis figure | figure detail | user | ready | methodology asset | embedded | evidence |
| F3_model_evolution.png | 3023 x 1539 | 1.96 | Model evolution evidence | Thesis figure | wide journey figure | user | ready | methodology asset | embedded | evidence |
| F4_feature_importance.png | 2400 x 1650 | 1.45 | Feature importance evidence | Thesis figure | chart + interpretation | user | ready | methodology asset | embedded | evidence |
| F5_confusion_matrix.png | 1650 x 1350 | 1.22 | Confusion matrix evidence | Thesis figure | chart + explanation | user | ready | methodology asset | embedded | evidence |
| F8_data_pipeline.png | 3030 x 1680 | 1.80 | Data pipeline evidence | Thesis figure | wide pipeline figure | user | ready | methodology asset | embedded | evidence |
| F9_regret_distribution.png | 2700 x 1200 | 2.25 | Regret distribution evidence | Thesis figure | wide chart | user | ready | methodology asset | embedded | evidence |
| F10_domain_holdout_gap_closed.png | 2370 x 1469 | 1.61 | Domain holdout evidence | Thesis figure | chart + takeaway | user | ready | methodology asset | embedded | evidence |
| F11_accuracy_vs_gap_closed.png | 2370 x 1463 | 1.62 | Accuracy versus gap closed evidence | Thesis figure | chart + takeaway | user | ready | methodology asset | embedded | evidence |
| 06_vbs_sbs_model_comparison.png | 2963 x 2065 | 1.43 | SBS/VBS/model comparison | Thesis figure | chart + metric definition | user | ready | thesis result figure | embedded | evidence |
| 10_v5_vs_v6_comparison.png | 2963 x 1762 | 1.68 | Source-aware check evidence | Thesis figure | comparison chart | user | ready | thesis result figure | embedded | evidence |
| formula_regret.png | 514 x 109 | 4.72 | Regret formula | Latex Formula | formula block | formula | rendered | formula manifest | none | equation |
| formula_gap_closed.png | 612 x 109 | 5.61 | Gap-closed formula | Latex Formula | formula block | formula | rendered | formula manifest | none | equation |

## IX. Content Outline

P01. Machine Learning Based Algorithm Selection for Sorting Numeric Arrays
- Central visual/evidence: template cover with Anthropic identity and thesis information.
- Purpose: establish the thesis and defense context.
- Speaker intent: introduce the problem as choosing before sorting begins.
- Do not say: abstract text, full literature survey, or backup framing.

P02. This defense follows the selector from task to evidence
- Central visual/evidence: agenda_list with five spoken parts.
- Purpose: make the talk easier to follow.
- Speaker intent: tell the committee how the defense will move: task, data/features, model, evaluation, limits.
- Do not say: a generic table of contents with empty section labels.

P03. The task is algorithm selection for one input array
- Central visual/evidence: array -> selector -> algorithm decision diagram.
- Purpose: define the task before any model/result.
- Speaker intent: this is not sorting itself; it is selecting the sorter.
- Do not say: complexity theory lecture.

P04. Sorting choice matters because equal complexity hides runtime differences
- Central visual/evidence: complexity/runtime contrast, with three algorithm names.
- Purpose: explain importance.
- Speaker intent: asymptotic class is not enough for this practical choice.
- Do not say: every algorithm textbook detail.

P05. Related work is mentioned inside the task, not separated
- Central visual/evidence: compact literature-in-context strip.
- Purpose: satisfy advisor instruction without a literature survey section.
- Speaker intent: previous work studied algorithm selection and ML-based performance prediction; this thesis applies that idea to numeric-array sorting with runtime regret evaluation.
- Do not say: long author-by-author survey.

P06. The objective is practical runtime selection, not oracle prediction
- Central visual/evidence: objective triangle: accuracy, regret, gap closed.
- Purpose: set claim discipline.
- Speaker intent: a selector can be useful even when not perfectly accurate.
- Do not say: perfect oracle or universal routing.

P07. The data comes from real numeric arrays across five domains
- Central visual/evidence: dataset scale and domain map using F8_data_pipeline or structured diagram.
- Purpose: explain dataset source and scale.
- Speaker intent: v5 is built on 1.18M arrays and five domains, not only small synthetic examples.
- Do not say: every data collection detail.

P08. The labels come from measured runtime after controlled transforms
- Central visual/evidence: pipeline_with_stages plus transform names.
- Purpose: explain how ground truth labels were created.
- Speaker intent: labels are measured winners, not manual opinions.
- Do not say: hand-wave timing noise.

P09. The extracted features are the bridge between arrays and algorithms
- Central visual/evidence: vertical_pillars for size, ordering, repetition, distribution.
- Purpose: properly explain feature logic.
- Speaker intent: each group exists because sorting algorithms respond to different structural properties.
- Do not say: just list feature names.

P10. The features are cheap summaries, not another sorting run
- Central visual/evidence: F2_feature_extraction detail and selected examples.
- Purpose: clarify extraction cost and examples.
- Speaker intent: the selector sees compact structure such as length, runs, duplicates, and entropy before sorting.
- Do not say: every formula unless asked.

P11. The system makes one general v5 decision before execution
- Central visual/evidence: layered_architecture plus F1_system_architecture.
- Purpose: connect data/features to the deployed selector idea.
- Speaker intent: feature extraction, XGBoost, one algorithm decision.
- Do not say: deployment beyond thesis.

P12. The portfolio is focused: introsort, heapsort, timsort
- Central visual/evidence: three algorithm comparison.
- Purpose: justify model labels.
- Speaker intent: the portfolio is small enough to defend and different enough to learn.
- Do not say: defend every possible sorting algorithm.

P13. The final model came from rejecting weaker paths
- Central visual/evidence: chevron_process plus F3_model_evolution.
- Purpose: explain research journey compactly.
- Speaker intent: regression, leakage-prone features, regret weighting, and cascade checks shaped the final v5 interpretation.
- Do not say: diary of every script.

P14. Evaluation needs SBS, VBS, regret, and gap closed
- Central visual/evidence: formulas and SBS/VBS model comparison.
- Purpose: prepare the committee for the main result.
- Speaker intent: accuracy alone does not measure runtime value.
- Do not say: dense math proof.

P15. The main result is runtime value first
- Central visual/evidence: kpi_cards for 93.1%, 76.1%, 89.6%, mean regret.
- Purpose: deliver headline result.
- Speaker intent: v5 closes 93.1% of the runtime gap with 76.1% test accuracy and 89.6% zero regret.
- Do not say: perfect classifier.

P16. Accuracy alone misses why most wrong labels are cheap
- Central visual/evidence: F9_regret_distribution and/or F11_accuracy_vs_gap_closed.
- Purpose: explain the result mechanism.
- Speaker intent: wrong class labels are not equally costly.
- Do not say: accuracy is irrelevant.

P17. The confusion matrix shows the hard boundary
- Central visual/evidence: F5_confusion_matrix.
- Purpose: show per-class behavior.
- Speaker intent: timsort is structurally visible; introsort vs heapsort is the weak boundary.
- Do not say: hide the weakness.

P18. Feature importance supports the structural interpretation
- Central visual/evidence: F4_feature_importance.
- Purpose: connect model behavior back to feature design.
- Speaker intent: length, repetition, and run-related features dominate in a way that makes sorting sense.
- Do not say: causal proof.

P19. Domain holdout shows transfer when structure transfers
- Central visual/evidence: F10_domain_holdout_gap_closed.
- Purpose: discuss generalization.
- Speaker intent: the result transfers unevenly, but runtime value survives across several holdouts.
- Do not say: universal generalization.

P20. Strict checks made the claim narrower and stronger
- Central visual/evidence: pros_cons_chart plus 10_v5_vs_v6_comparison and v7/v8 notes.
- Purpose: handle negative results and rigor.
- Speaker intent: source-aware v6 lowered metrics, v7 was rejected, v8 exposed the intro/heap boundary.
- Do not say: v7/v8 succeeded or source-aware checks were ignored.

P21. The limits are clear, including the F1 specialization boundary
- Central visual/evidence: limitation frame: intro/heap boundary, source split caveat, F1 routing separate.
- Purpose: state boundaries before Q&A.
- Speaker intent: v5 is the main general selector; F1 channel/flag models are a separate specialization, not the main contribution.
- Do not say: mix three-label v5 with five-label F1 routing.

P22. The contribution is a measured selector and an evaluation discipline
- Central visual/evidence: closing template with contributions and future work.
- Purpose: close and invite questions.
- Speaker intent: dataset/labels, structural selector, runtime-aware evaluation; future is hardware-aware and adaptive, LinUCB remains future work.
- Do not say: new results or validated bandit claims.

## X. Speaker Notes Strategy

- Notes are rehearsal-script style in natural Ahmed defense voice.
- Each note begins from what Ahmed should say, not from what the slide contains.
- Notes must include transitions between slides so the story does not jump.
- Difficult questions: acknowledge the concern first, then defend with evidence.
- Keep visible slide text sparse; put thesis detail in notes.

## XI. Technical Constraints

- SVG viewBox: `0 0 1280 720`.
- Use template page shells where mapped in `spec_lock.md page_layouts`.
- No decorative short underlines under titles.
- No text outside safe area or container boundaries.
- No standalone literature section.
- No backup deck or appendix.
- v5 general selector and F1-specific routing must remain separated.
- Every headline number must trace to `sources/metrics_summary.json`.
