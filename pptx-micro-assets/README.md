# Defense PPT Micro-Asset Kit

This folder stores reusable visual components for the official MSc defense deck.
It is not the final PPTX project. The purpose is to prepare detailed, thesis-specific
visual material before rebuilding the deck with `ppt-master`.

## Rules

- Use the submitted thesis terminology.
- Use CIU identity for official opening and closing.
- Prefer detailed visual explanations over generic boxes.
- Use the full 16:9 slide canvas. Do not default to left/right or one-side alignment.
- Prefer centered, radial, layered, full-width, full-height, or stage-like compositions.
- Keep every SVG animation-ready with meaningful top-level group ids.
- Reuse these assets during the later `ppt-master` build instead of redrawing from scratch.

## Assets

- `images/ciu-logo-tranpernt.png` - CIU transparent logo.
- `components/01_ciu_cover_background.svg` - official title/cover background system.
- `components/02_algorithm_selection_scene.svg` - detailed Algorithm Selection Problem visual.
- `components/03_structural_feature_extraction.svg` - array-to-features visual system.
- `components/04_runtime_label_generation.svg` - measured runtime and label-generation visual.
- `components/05_sbs_vbs_regret_landscape.svg` - SBS/VBS/regret/gap-closed explanation visual.

## Later PPT Master Use

During the final PPT build, these components should be imported as reference visuals or rebuilt directly into slide SVGs with the same group naming. The final deck should not simply place these as screenshots if editable SVG structure is better.
