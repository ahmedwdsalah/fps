# Micro-Asset Registry

## 01 CIU Cover Background

- File: `components/01_ciu_cover_background.svg`
- Use on: official title slide and possibly final Q&A slide.
- Purpose: make the defense look official and institutional before showing results.
- Must include:
  - CIU logo;
  - same global top and bottom bars used across all slides;
  - large subtle CIU watermark only;
  - exact submitted thesis title area;
  - wide centered formal title using the full page width.
- Must not include:
  - `CYPRUS INTERNATIONAL UNIVERSITY` text on the top bar;
  - left-aligned title block;
  - Nicosia/location text;
  - date/time text;
  - unnecessary degree sentence such as `A thesis for...`;
  - decorative array image or sorting texture on the title slide.
- Required wording:
  - `Presented by`
  - `AHMED SALAH ABDALHI MOHAMMED`
  - `Supervised by`
  - `Assoc. Prof. Dr. Emre Özbilge`
- Animation groups:
  - `ciu_background`
  - `ciu_watermark`
  - `title_block`
  - `identity_block`
  - `array_texture`

## 02 Algorithm Selection Problem Scene

- File: `components/02_algorithm_selection_scene.svg`
- Rejected draft: `components_v3/02_algorithm_selection_scene_v3.svg`
- Current candidate: `components_v3/02_algorithm_selection_problem_v6.svg`
- Previous candidates: `components_v3/02_algorithm_selection_problem_v4.svg`, `components_v3/02_algorithm_selection_problem_v5.svg`
- Status: v6 candidate for review. v3 rejected; do not use v3 in final deck.
- Reason:
  - messy composition;
  - toy-like bars and arrows;
  - title collision;
  - not scientific enough for MSc defense;
  - does not look like a serious Algorithm Selection Problem diagram;
  - weak mapping to Rice framework / thesis methodology.
- Use on: task slide.
- Purpose: explain the thesis task visually without generic boxes.
- Thesis terms:
  - Algorithm Selection Problem;
  - sorting numeric arrays;
  - structural array features;
  - algorithm portfolio;
  - selector.
- Animation groups:
  - `array_input`
  - `structure_highlights`
  - `feature_vector`
  - `algorithm_portfolio`
  - `selector_decision`
  - `chosen_algorithm`

## 03 Structural Feature Extraction

- File: `components/03_structural_feature_extraction.svg`
- Use on: feature explanation slide.
- Purpose: show how a numeric array becomes a 16-dimensional structural feature vector.
- Thesis terms:
  - feature-based instance characterisation;
  - size features;
  - sortedness and disorder features;
  - run-based features;
  - duplicate and frequency features;
  - entropy;
  - dispersion and robust spread;
  - shape and outlier features.
- Animation groups:
  - `source_array`
  - `feature_group_size`
  - `feature_group_order`
  - `feature_group_runs`
  - `feature_group_duplicates`
  - `feature_group_distribution`
  - `feature_vector_output`

## 04 Runtime-Based Label Generation

- File: `components/04_runtime_label_generation.svg`
- Use on: data/labelling slide.
- Purpose: explain runtime-based label generation from measured algorithm runtimes.
- Thesis terms:
  - runtime-based label generation;
  - structural transforms;
  - Introsort;
  - Heapsort;
  - Timsort;
  - fastest measured algorithm.
- Animation groups:
  - `transform_strip`
  - `runtime_race`
  - `measurement_table`
  - `label_assignment`
  - `quality_note`

## 05 SBS/VBS/Regret Landscape

- File: `components/05_sbs_vbs_regret_landscape.svg`
- Use on: evaluation slide before final result.
- Purpose: explain why accuracy alone is insufficient and how regret/gap closed are evaluated.
- Thesis terms:
  - Single Best Solver (SBS);
  - Virtual Best Solver (VBS);
  - regret;
  - gap closed.
- Animation groups:
  - `runtime_landscape`
  - `sbs_line`
  - `vbs_path`
  - `model_path`
  - `regret_gap`
  - `gap_closed_callout`
