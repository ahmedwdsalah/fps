# Official Defense PPT Knowledge Base

This file is the working memory for the MSc thesis defense presentation. Use it before creating or editing any defense slides.

## Evidence Hierarchy

Use this order when facts conflict:

1. Submitted thesis text in `marker-output/Ahmed-v3/Ahmed-v3.md` for what was officially submitted.
2. Final result artifacts in `results/**/evaluation_results.json`, `regret_analysis.json`, `router_eval.json`, and `strict_router_eval.json` for exact numbers.
3. Scripts in `scripts/` for actual pipeline mechanics.
4. Progress docs in `PROJECT.md`, `THESIS-PROGRESS.md`, and `docs/*.md` for research journey and interpretation.
5. Older docs may describe planned direction that was later superseded. Mark as early plan, not final claim.

## Slide Tooling Decision

Use `ppt-master` as the primary deck-building workflow.

Verified local context:

- Skill path: `/Users/ahmed/.agents/skills/ppt-master/SKILL.md`.
- Main output style: hand-authored SVG pages exported to PPTX.
- Export script: `/Users/ahmed/.agents/skills/ppt-master/scripts/svg_to_pptx.py`.
- Formula rendering script: `/Users/ahmed/.agents/skills/ppt-master/scripts/latex_render.py`.
- Bundled Python runtime has `python-pptx`, Pillow, lxml, NumPy, and Pandas.
- Local system PATH does not currently expose `pdflatex`, `xelatex`, `lualatex`, `latexmk`, `pandoc`, ImageMagick, Inkscape, or `rsvg-convert`.
- `ppt-master` LaTeX renderer does not require local TeX; it renders formulas through network providers such as CodeCogs, QuickLaTeX, MathPad, or Wikimedia.
- CodeCogs PNG rendering was tested successfully for the thesis regret formula.

Recommended practical path for this defense:

- Build slides with `ppt-master` SVG pipeline.
- Use LaTeX only for a few important equations: regret, SBS/VBS/gap closed, maybe selector mapping `S: f(x) -> A`.
- Render LaTeX as transparent PNG assets through `latex_render.py`, then place those PNGs inside SVG slides.
- Do not build the whole deck in Beamer/LaTeX. Beamer is good for math-heavy academic talks, but this defense needs visual evidence, diagrams, plots, and narrative. Full LaTeX deck would slow layout work and reduce visual polish.
- Do not rely on local TeX compilation unless a TeX distribution is installed later.
- If formula rendering fails during deck build, fallback is to typeset simple equations directly as SVG text, or use pre-rendered PNGs from `latex_render.py`.

## Locked Defense Preferences

- Speaking time: maximum 25 minutes.
- Defense deck target: content-led, currently 24 slides. Do not force 22 slides if the argument needs a clearer sequence.
- Speaker notes: include concise notes for rehearsal and delivery.
- Audience assumption: mixed computer science / engineering committee.
- Visual style: clean light academic theme; formal, readable, chart-friendly.
- Claim posture: measured strong. Lead with `93.1% gap closed`, `76.1% accuracy`, `89.6% zero regret`, but explain limitations clearly.
- Model-track strategy: v5 is the main story. F1 channel/flag models appear only as an integrated specialization caveat where they clarify the boundary of the work.
- Rigor strategy: include one integrated slide on what the strict checks taught (`v7`, `v8`, source-aware checks, F1 routing), not a separate appendix.
- No backup deck. Do not design this presentation as “main slides plus backup slides.” The defense must stand as one complete 22-slide talk.
- Slide density: sparse main slides with spoken detail in concise speaker notes.
- Build priority: defense clarity first, with strong visual polish and enough technical depth.
- Title date wording: `June 2026`.
- CIU branding: formal minimal; use CIU identity on title and closing only unless official assets are supplied.
- Output bundle: PPTX, PDF, speaker-notes document, and source/assets folder.
- Pacing: target a calm 20-minute main talk within the 25-minute maximum, leaving room for pauses, nerves, and committee interruption.
- Audience-facing slide titles: use clear academic claim titles, not neutral section labels and not dramatic marketing-style titles.
- Q&A posture: acknowledge the concern first, then defend with evidence and explain why the contribution still holds.
- Journey tone: controlled journey. Show failed paths only when they explain why the final method is credible.
- Speaker-note depth: rehearsal-script style, written in natural Ahmed defense voice with room to improvise.
- Build gate: do not start PPTX generation until the 22-slide main plan has audience-facing claim titles, one central visual/evidence item, slide purpose, speaker-note intent, and “do not say” guidance.

## Locked Story Decisions

- Opening hook: frame the defense through the algorithm selection problem, then immediately connect that framework to sorting numeric arrays.
- Main headline: runtime value first. Lead with `93.1% gap closed`, supported by `76.1% accuracy` and `89.6% zero regret`.
- Research journey: one compact slide only. It must explain the research logic, not become a separate history section.
- Data explanation: two main slides:
  - Dataset domains and scale: `1,188,265` arrays from 5 domains.
  - Structural transforms and labeling protocol.
- Math level: main deck includes only regret/SBS/VBS/gap-closed formulas. Avoid showing every feature formula in the main deck.
- F1 strategy: brief integrated caveat only. It is not a second presentation track and not a backup section.
- Figure strategy: mostly reuse existing thesis/methodology figures, with light slide-level framing and consistent title/caption treatment.
- Limitations tone: one main rigor/lessons slide that frames negative results as evidence of careful evaluation.
- Committee-risk handling: risks such as leakage, v6, F1 routing, metric definitions, feature cost, and why `76.1%` accuracy is still valuable must be handled inside the 22-slide story or in oral Q&A, not as a separate backup deck.

## Ahmed Defense Voice Pattern

This section captures the speaking and wording pattern to use when writing slides and speaker notes. It is extracted from the submitted thesis, especially the introduction, methodology, results, and conclusion. Use this instead of generic machine-style presentation wording.

Core voice:

- Speak as a direct researcher: clear, technical, and grounded in what was built, tested, and found.
- Use moderate first person in speaker notes: “I started from…”, “I found…”, “we evaluate…”, “the model shows…”.
- Keep slide text human and concise. Slides should carry claims and evidence; notes should carry the explanation.
- Avoid corporate phrases, inflated claims, and vague motivational language.
- Avoid machine-like labels such as “Backup Slide Inventory” or “Optimization Blueprint” in audience-facing slide titles.
- Do not sound like the thesis is being read aloud. Convert thesis wording into spoken defense language.

The recurring thesis argument pattern:

1. State the practical problem.
2. Explain why the obvious/simple solution is not enough.
3. Introduce the selector as the practical response.
4. Show the measured evidence.
5. Explain why accuracy alone is incomplete.
6. Acknowledge the limit.
7. State what the result means.

Preferred sentence shapes for speaker notes:

- “The problem I am addressing is not sorting in theory, but choosing the right sorting algorithm for each real input.”
- “The important point here is that the fastest algorithm changes with structure.”
- “This is why I do not evaluate the model only by accuracy.”
- “A wrong label is not always an expensive mistake.”
- “The strongest evidence is not the `76.1%` accuracy alone, but the `93.1%` gap closed and `89.6%` zero-regret predictions.”
- “This result showed me that timsort-friendly cases are visible from structure, but introsort and heapsort are much harder to separate.”
- “I treat this as a limitation of the current feature space, not just a training failure.”
- “The routed F1 experiment is useful, but it is not the main claim of the thesis.”
- “The strict split reduced the result, but it also made the evaluation more credible.”
- “So the contribution is a practical selector and an evaluation discipline, not a claim that the model is perfect.”

Preferred slide-title style:

- Use claim titles, not section labels.
- Good: “Accuracy alone does not explain the value.”
- Good: “Most wrong choices are low-cost.”
- Good: “Timsort is structurally visible; introsort vs heapsort is not.”
- Good: “Strict checks changed the interpretation, not the contribution.”
- Avoid: “Performance Metrics”, “Model Evaluation”, “Backup Results”, “Negative Results”.

How to speak about the journey:

- Use a confident honest tone.
- Failed models are not embarrassing; they show why the final design is disciplined.
- Phrase failures as research decisions:
  - “Regression learned runtime scale, not the winner boundary.”
  - “Timing-derived features proved the ceiling, but they were leakage.”
  - “Regret-aware training sounded aligned with the goal, but empirically it hurt the selector.”
  - “The binary cascade showed the real structure of the problem.”

How to speak about limitations:

- State limits directly, then explain why the contribution still stands.
- Do not hide v6/source-aware checks if asked.
- Do not volunteer a long caveat before the main result.
- Use this framing:
  - “This limitation matters, but it does not remove the main result, because the model still captures most of the practical runtime gap.”
  - “The result is strongest for structural regimes, especially timsort-friendly inputs.”
  - “The unresolved part is the low-level introsort/heapsort boundary.”

How to handle literature:

- Follow Prof. Emre’s instruction strictly: no standalone literature survey in the deck.
- Mention literature only while explaining the task and positioning:
  - Rice formalized algorithm selection.
  - SATzilla showed feature-based selection can work in solver portfolios.
  - AlphaDev discovers algorithms.
  - Learning-augmented sorting modifies algorithms.
  - This thesis selects among existing sorting algorithms before execution.

Audience-facing tone:

- Main slides should sound like a defense, not a report appendix.
- Technical detail should appear only when it strengthens the 22-slide story; otherwise it belongs in oral Q&A.
- Every slide should be speakable in one clear sentence.
- If a sentence would sound unnatural when spoken aloud, rewrite it.
- A thesis defense PPTX is not meant to contain the full thesis text. Its purpose is to support the oral defense: guide the committee through the argument, show the strongest evidence, and give the presenter room to explain.
- Main slides must be visual and sparse. Use figures, diagrams, numbers, and short claim titles; do not fill slides with paragraphs.
- Do not make slides that can only be understood by reading dense text. The presenter should explain the meaning aloud.
- Prefer one central idea per slide. If a slide needs many bullets to be understood, split it, move details to speaker notes, or leave it for oral Q&A.
- Speaker notes can hold rehearsal wording, but the visible slide should stay clean and defendable from the screen.
- Do not put internal planning language into the audience-facing PPTX or speaker notes. Words used in this knowledge file for coordination are not automatically slide language.
- Keep machine/planning terms internal unless they are real thesis terms. Acceptable thesis terms include `SBS`, `VBS`, `regret`, `gap closed`, `source-aware split`, `XGBoost`, and algorithm names.
- Avoid audience-facing words such as “blueprint”, “inventory”, “bundle”, “guardrails”, “locked”, “workflow”, “content architecture”, “claim posture”, “voice pattern”, “deck norms”, “committee-risk defense”, and “build priority”.
- Replace internal wording with natural defense wording:
  - “backup inventory” -> “questions I can answer orally”
  - “claim posture” -> “how I state the result”
  - “negative results” -> “what the experiments taught”
  - “rigor slide” -> “checks that made the result more credible”
  - “model-track strategy” -> “the general model and the F1-specific experiment”
  - “storyline” -> “the order of the explanation”
  - “build priority” -> “what the presentation should make clear”
  - “guardrails” -> “things I should not overclaim”

Additional patterns from the full thesis:

- Use contrast to make the idea clear:
  - “not sorting in theory, but choosing the right algorithm for real inputs”
  - “not modifying sorting algorithms, but selecting among existing ones”
  - “not accuracy alone, but runtime cost”
  - “not a perfect oracle, but a practical selector”
  - “not a training failure only, but a feature-space ceiling”
- Define boundaries before claims. The thesis often becomes credible by saying exactly what is excluded:
  - comparison-based C-level algorithms only;
  - no custom low-level reimplementation;
  - no non-comparison sort in the main portfolio;
  - no claim that LinUCB was empirically validated;
  - F1 routing is a separate specialised experiment, not the global model.
- Use “why this matters” after technical explanation. Do not end a slide on a mechanism; end on the implication.
  - Mechanism: “Timsort detects runs.”
  - Defense implication: “So run structure gives the model a visible signal.”
  - Mechanism: “Introsort and heapsort differ in cache and branch behaviour.”
  - Defense implication: “So value-level features cannot fully separate them.”
- Use the thesis habit of moving from local observation to general meaning:
  - observation: `76.1%` accuracy;
  - explanation: many errors are near-ties;
  - meaning: runtime regret is low;
  - contribution: selector quality must be judged with cost-aware metrics.
- Treat tables and figures as evidence, not decoration. When a figure appears, the spoken note should answer:
  - What should the committee notice first?
  - What does this prove or limit?
  - How does it move the argument forward?
- Use “practical selector” language often. This is the thesis identity.
  - Better: “This is a practical selector for realistic numeric arrays.”
  - Worse: “This is an ML model for sorting.”
- Use “structural signal” as the recurring explanation.
  - The model works when the fastest algorithm leaves a structural trace in the input.
  - The model struggles when the difference depends on hardware-level effects not visible in the feature vector.
- Use “evaluation discipline” as a contribution.
  - The thesis is not only a model result; it also argues for source-aware splits, near-tie handling, and regret-based metrics.
- Explain related work through difference, not chronology:
  - SATzilla: same algorithm-selection idea, different problem domain.
  - AlphaDev: discovers new low-level routines; this thesis selects among existing algorithms.
  - Sorting with Predictions: changes sorting internals; this thesis chooses before sorting.
  - Adaptive Hybrid Sort: modifies one algorithm internally; this thesis chooses among complete algorithms.
- Preserve the thesis’s repeated “because / therefore” logic in speech:
  - “Because sorting is repeated at scale, small per-instance gains matter.”
  - “Because algorithms are near-tied on many arrays, accuracy is not enough.”
  - “Because F1 channels are structurally different, routing is plausible.”
  - “Because strict splits reduce leakage, they are more credible.”

Audience-facing slide title rewrites from the thesis pattern:

- Instead of “Algorithm Portfolio”: “The algorithms are different enough to make selection useful.”
- Instead of “Feature Engineering”: “The array is converted into structural signals.”
- Instead of “Data Collection”: “Real arrays give the selector the structures synthetic data missed.”
- Instead of “Model Performance”: “The model is imperfect, but most mistakes are cheap.”
- Instead of “Domain Holdout”: “The selector transfers when structure transfers.”
- Instead of “Routed Models”: “Routing looked promising until strict separation tested it.”
- Instead of “Limitations”: “The remaining errors show where the features stop seeing.”
- Instead of “Future Work”: “The next step is hardware-aware and adaptive selection.”

Speaker-note rhythm to imitate:

1. “Here is the issue.”
2. “The simple interpretation would be X.”
3. “But the result shows Y.”
4. “That is why I use metric/design Z.”
5. “So the conclusion is limited but useful.”

Example in Ahmed defense voice:

“At first, `76.1%` accuracy looks only moderate. But in this task, the cost of a wrong label depends on how far the chosen algorithm is from the oracle. When I evaluate with regret, the interpretation changes: the selector closes `93.1%` of the SBS-to-VBS gap, and almost nine out of ten predictions have zero regret. So the model is not perfect at copying oracle labels, but it is strong at avoiding expensive runtime mistakes.”

## Defense Deck Norms

- Defense PPT is not a compressed thesis. It is a clear argument for committee: problem, method, evidence, limits, contribution.
- Typical MSc engineering/computer science defense deck: 15-25 slides for ~20-30 minutes, plus Q&A.
- Literature review should be minimal. Use only enough background to motivate algorithm selection and sorting input sensitivity.
- Advisor instruction from Prof. Emre on 20.06.2026: do not make a standalone literature survey in the presentation. Explain what the task is, why it matters, and mention literature inside that explanation with “who did what.”
- Methodology and results should dominate.
- Slides should be visual-first: diagrams, tables, plots, concise bullets. Avoid thesis paragraphs.
- Each slide should answer one question.
- Committee expects honesty: limitations and negative results are strengths when framed as rigorous evaluation.
- Every number in slides must trace to thesis/repo artifacts.
- Avoid overclaiming unvalidated work.

## Model Explanation Slide Rules

- The XGBoost/model explanation slides are not result slides. Do not lead them with headline numbers, dataset counts, percentages, or metric values.
- Their purpose is to explain the full method logic:
  - why the problem becomes supervised classification;
  - why raw arrays are converted into fixed structural features;
  - why XGBoost was chosen for tabular structural signals;
  - how the selector makes a decision before sorting;
  - what makes the training setup credible without turning the slide into a results table.
- Use numbers only if they are necessary to define the method, not to impress. Prefer equations, model formulation, feature groups, and training-protocol logic over large numeric callouts.
- Do not draw toy pipelines, playful tree cartoons, or decorative sample bars. The model section should look like a scientific method figure or compact model card.
- Better visual language for these slides:
  - formal mapping: `x -> f(x) -> S(f(x))`;
  - supervised dataset notation: `{(f(x_i), y_i)}`;
  - label definition: `y_i = argmin_a T(a, x_i)`;
  - XGBoost as additive boosted trees with `softmax` output;
  - compact model-card table for input, target, objective, training discipline, and output.
- Explain XGBoost through the thesis/codebase reasons:
  - the input is fixed-length tabular data, not sequence data;
  - feature interactions are non-linear;
  - tree ensembles work well on mixed structural features;
  - feature importance gives some interpretability;
  - probability output supports choosing among `introsort`, `heapsort`, and `timsort`.
- Keep the wording in Ahmed’s voice. Do not say generic phrases like “the model learns patterns.” Say what structural signal is visible and why that matters for sorting selection.
- The model story should answer “how and why this model was built,” before any result slide answers “how well it worked.”

## Main Deck Blueprint

Target length: content-led. Current official structure is 24 slides because the problem, formal framework, data, features, model, evaluation, results, and limits need separate breathing room. Do not force the deck back to 22 slides unless Ahmed explicitly asks to shorten it.

1. Title
   - Thesis title, candidate name, department, supervisor, date.
   - Keep visual quiet and formal.
2. Algorithm Selection Task
   - Start from Rice-style algorithm selection: map an instance to the best algorithm.
   - Connect immediately to sorting numeric arrays.
3. Why Sorting Selection Matters
   - Explain that sorting performance depends on input structure, not only `O(n log n)`.
   - Use practical framing: repeated sorting appears inside data processing, telemetry, finance, and scientific workflows.
4. Compact Related Work Inside Task Framing
   - No standalone literature survey.
   - Mention who did what: Rice formalized algorithm selection; SATzilla applied feature-based solver selection; AlphaDev discovers sorting routines; learning-augmented sorting modifies algorithms; this thesis selects among existing sorting algorithms before execution.
5. Research Objective
   - Main question: can cheap structural features predict the fastest sorting algorithm for numeric arrays?
   - State the target portfolio: `introsort`, `heapsort`, `timsort`.
6. System Idea
   - Array input -> structural features -> XGBoost selector -> predicted fastest algorithm.
   - Emphasize selection before sorting.
7. System Architecture
   - Use `methodology_assets/F1_system_architecture.png`.
   - Speaker note should explain the full data/model/evaluation loop.
8. Algorithm Portfolio
   - Introduce `introsort`, `heapsort`, `timsort`.
   - Explain why the final thesis kept production-grade comparison sorts.
9. Dataset Domains
   - Show 5 domains and scale: Formula 1 telemetry, stock, crypto, earthquake/seismic, weather.
   - Headline number: `1,188,265` arrays.
10. Transforms and Labeling Protocol
   - Show RAW, REV, SHUF, QBIN50, PSORT10.
   - Explain that all three algorithms were timed and the fastest measured runtime became the label.
11. Feature Groups
   - Show 16 structural features grouped as size, ordering, repetition, distribution, robust scale.
   - Use `methodology_assets/F2_feature_extraction.png` if it fits cleanly.
12. XGBoost v5 Production Model
   - Present v5 as the submitted general model: all domains, 3 algorithms.
   - Include balancing/noise-filtering at high level only.
13. Model Evolution Journey
   - One compact slide from v1 regression -> v2 classifier -> v3 leakage ceiling -> v5 production -> v6/v7/v8 lessons.
   - Use `methodology_assets/F3_model_evolution.png`.
14. Main Runtime Result
   - Lead with `93.1% SBS-to-VBS gap closed`.
   - Pair with `76.1% test accuracy` and `89.6% zero regret`.
   - Use `methodology_assets/F11_accuracy_vs_gap_closed.png` or `results/thesis_figures/06_vbs_sbs_model_comparison.png`.
15. Regret Metrics Explanation
   - Define SBS, VBS, per-instance regret, and gap closed.
   - Include only these formulas in main deck.
16. Confusion Matrix / Per-Class Behavior
   - Explain strong timsort detection and weaker introsort-vs-heapsort boundary.
   - Use `methodology_assets/F5_confusion_matrix.png` or `results/thesis_figures/03_v5_per_algorithm_recall.png`.
17. Feature Importance / Interpretation
   - Explain why length, repetition, runs, entropy, and duplicates matter.
   - Use `methodology_assets/F4_feature_importance.png`.
18. Domain Holdout Generalization
   - Show leave-one-domain-out results.
   - Use `methodology_assets/F10_domain_holdout_gap_closed.png`.
19. Rigor / Negative Results / Lessons
   - Cover v6/source-aware checks, v7 regret-aware training, v8 binary cascade, and F1 routing carefully.
   - Main message: strict checks reduced or rejected some paths, but clarified the real boundary of the method.
20. Limitations
   - Feature extraction has overhead, especially for very small arrays.
   - Introsort vs heapsort remains hard because current structural features miss low-level hardware effects.
   - v5 is not an online adaptive/bandit system.
21. Contributions
   - Per-instance sorting selector using structural features.
   - Large multi-domain dataset and evaluation protocol.
   - Runtime-sensitive evaluation showing why accuracy alone is incomplete.
   - Clear empirical finding: timsort-friendly cases are learnable; introsort/heapsort boundary is weak.
22. Future Work / Q&A
   - Hardware-aware features, source-aware larger dataset, online adaptation as future work, broader algorithm portfolio.
   - End with Q&A.

## Spoken Main Slide Plan

This is the audience-facing version of the main deck. Use these titles and speaking intents when creating slides. Do not copy internal planning labels into the PPTX.

1. Machine Learning Based Algorithm Selection for Sorting Numeric Arrays
   - Central visual/evidence: formal title slide with CIU text, thesis title, candidate, supervisor, and `June 2026`.
   - Purpose: establish the official defense context.
   - Speaker-note intent: briefly introduce yourself and say the defense is about choosing the fastest sorting algorithm for each numeric array before sorting begins.
   - Do not say: do not start with tooling, repository history, or implementation details.
2. The problem is choosing the right algorithm for each input
   - Central visual/evidence: simple mapping from input array to algorithm choice.
   - Purpose: make the task clear before showing models or results.
   - Speaker-note intent: explain that this is not sorting in theory; it is deciding which existing sorting algorithm is best for a real input.
   - Do not say: do not use long literature-review language.
3. Same complexity does not mean same runtime
   - Central visual/evidence: small comparison graphic showing same `O(n log n)` family but different behavior by structure.
   - Purpose: justify why selection is useful.
   - Speaker-note intent: say that input size matters, but order, runs, duplicates, and distribution shape also change practical runtime.
   - Do not say: do not over-explain every sorting algorithm yet.
4. This thesis selects before sorting begins
   - Central visual/evidence: contrast row: algorithm discovery vs algorithm modification vs algorithm selection.
   - Purpose: position related work without creating a literature survey.
   - Speaker-note intent: mention Rice, SATzilla, AlphaDev, and learning-augmented sorting only to show the difference: this work selects among existing algorithms before execution.
   - Do not say: do not title this as “Literature Review” or spend time summarizing papers.
5. The aim is a practical selector, not a perfect oracle
   - Central visual/evidence: objective statement plus `S: f(x) -> A`.
   - Purpose: define the research objective and the practical boundary.
   - Speaker-note intent: say the goal is to reduce runtime compared with a fixed algorithm while keeping feature extraction and prediction cheap.
   - Do not say: do not claim the selector always finds the oracle winner.
6. The array is converted into structural signals
   - Central visual/evidence: array -> 16 features -> XGBoost -> selected algorithm.
   - Purpose: explain the system idea in one step.
   - Speaker-note intent: describe features as the bridge between variable-length arrays and a fixed model input.
   - Do not say: do not list all 16 features on this slide.
7. The system makes one decision before execution
   - Central visual/evidence: `methodology_assets/F1_system_architecture.png`.
   - Purpose: show the full inference and evaluation flow.
   - Speaker-note intent: walk through feature extraction, model prediction, algorithm selection, and measured runtime.
   - Do not say: do not discuss every script or file.
8. The algorithms are different enough to make selection useful
   - Central visual/evidence: compact portfolio comparison for `introsort`, `heapsort`, `timsort`.
   - Purpose: explain why these three algorithms create a meaningful portfolio.
   - Speaker-note intent: say each algorithm reacts differently to structure: timsort to runs, introsort to general fast cases, heapsort as predictable fallback.
   - Do not say: do not introduce F1 five-algorithm experiments here.
9. Real arrays give the selector the structures synthetic data missed
   - Central visual/evidence: five-domain dataset view with `1,188,265` arrays.
   - Purpose: show the dataset scale and real-world coverage.
   - Speaker-note intent: explain that synthetic arrays helped early, but real domains were needed because real signals contain trends, repetition, spikes, and smooth regions.
   - Do not say: do not claim every domain has large speedup alone.
10. The labels come from measured runtime, not opinion
   - Central visual/evidence: `methodology_assets/F8_data_pipeline.png` or a clean transform/label pipeline.
   - Purpose: explain transforms and empirical labeling.
   - Speaker-note intent: say each array was transformed, each candidate algorithm was timed, and the fastest measured runtime became the label.
   - Do not say: do not bury the slide in benchmark protocol details.
11. The features describe structure, not raw values
   - Central visual/evidence: `methodology_assets/F2_feature_extraction.png` or grouped feature blocks.
   - Purpose: show what the model sees.
   - Speaker-note intent: explain the five groups: size, ordering, repetition, distribution, and robust scale.
   - Do not say: do not promise every implemented feature is strictly one-pass.
12. The main model is the general v5 selector
   - Central visual/evidence: v5 model card: all domains, 3 algorithms, 16 features, XGBoost.
   - Purpose: separate the main story from F1-specific experiments.
   - Speaker-note intent: say v5 is the submitted general model; F1 routing is a separate specialized experiment that only appears as a boundary/caveat.
   - Do not say: do not mix v5 with the F1 `actual_5` label set.
13. Each failed path clarified the final design
   - Central visual/evidence: `methodology_assets/F3_model_evolution.png`.
   - Purpose: show controlled research journey.
   - Speaker-note intent: explain that regression learned runtime scale, timing features were leakage, and later checks showed what was credible.
   - Do not say: do not make the journey sound like confusion or random trial-and-error.
14. The model is imperfect, but most mistakes are cheap
   - Central visual/evidence: `93.1% gap closed`, `76.1% accuracy`, `89.6% zero regret`.
   - Purpose: present the headline result.
   - Speaker-note intent: say accuracy alone looks moderate, but regret changes the interpretation because many wrong labels are near-ties.
   - Do not say: do not present `76.1%` alone without runtime value.
15. Accuracy alone does not explain the value
   - Central visual/evidence: SBS/VBS/model comparison and regret/gap formulas.
   - Purpose: define why gap closed matters.
   - Speaker-note intent: explain SBS as the best fixed algorithm, VBS as the oracle, and gap closed as how much useful runtime improvement the model captures.
   - Do not say: do not overdo formulas beyond SBS, VBS, regret, and gap closed.
16. Timsort is structurally visible; introsort vs heapsort is not
   - Central visual/evidence: confusion matrix or per-class recall.
   - Purpose: explain the error pattern.
   - Speaker-note intent: say the model sees timsort-friendly structure well, but introsort and heapsort often differ through lower-level effects.
   - Do not say: do not describe the confusion matrix as a failure only.
17. The model mostly uses features that make sorting sense
   - Central visual/evidence: feature importance chart.
   - Purpose: show interpretability and credibility.
   - Speaker-note intent: explain that length, repetition, runs, and ordering dominate, which matches the expected mechanics of sorting.
   - Do not say: do not imply feature importance proves causality by itself.
18. The selector transfers when structure transfers
   - Central visual/evidence: domain holdout chart.
   - Purpose: show generalization beyond one domain.
   - Speaker-note intent: explain that gap closed remains strong across held-out domains, while F1/weather show why structure and domain mix still matter.
   - Do not say: do not oversell domain holdout as universal deployment proof.
19. Strict checks changed the interpretation, not the contribution
   - Central visual/evidence: compact lessons from v6, v7, v8, and F1 routing.
   - Purpose: show rigor without derailing the main result.
   - Speaker-note intent: say later checks reduced some optimistic paths, rejected some ideas, and clarified that the strongest signal is timsort-vs-rest.
   - Do not say: do not call F1 routing a global improvement.
20. The remaining errors show where the features stop seeing
   - Central visual/evidence: limitation map: overhead on small arrays, hardware effects, unvalidated online adaptation.
   - Purpose: state limits clearly.
   - Speaker-note intent: explain that current features describe array values and order, but not cache behavior, branch prediction, or platform-specific effects.
   - Do not say: do not sound defensive or hide limitations.
21. The contribution is a selector and an evaluation discipline
   - Central visual/evidence: three contribution blocks: practical selector, large real dataset, runtime-sensitive evaluation.
   - Purpose: summarize what the thesis adds.
   - Speaker-note intent: say the thesis contributes both a working selection pipeline and a way to judge selector quality with regret and source-aware checks.
   - Do not say: do not claim the model is a complete replacement for low-level sorting research.
22. The next step is hardware-aware and adaptive selection
   - Central visual/evidence: future directions plus Q&A prompt.
   - Purpose: close with a forward-looking but bounded ending.
   - Speaker-note intent: mention hardware-aware features, broader portfolios, source-aware larger data, and validated online adaptation as future work, then invite questions.
   - Do not say: do not claim LinUCB has already been validated.

## Core Thesis Story

The thesis is about per-instance sorting algorithm selection for numeric arrays. Instead of always using one sorting algorithm, the system extracts cheap structural features from each input array and predicts which algorithm will sort it fastest.

There are two model tracks in the project:

- General production model: XGBoost v5 trained across all submitted thesis domains, selecting among `introsort`, `heapsort`, and `timsort`.
- F1-specific flagged/channel model family: domain-specific route for Formula 1 only. A known channel flag or channel classifier identifies `Speed`, `RPM`, `DRS`, `Distance`, `X`, `Y`, `Z`, `Throttle`, or `nGear`, then a channel-specific algorithm selector is used.

Do not collapse these into one model. They answer different deployment assumptions:

- v5 answers: “Given an arbitrary numeric array from mixed domains, which of the 3 production sorting choices should I use?”
- F1 channel route answers: “Given an F1 telemetry array and channel identity, can channel-specialized models choose better inside this one domain?”

The strongest narrative is not “we trained XGBoost.” The real story is:

- Proved sorting choice depends on input structure.
- Built and validated lightweight structural features.
- Tested several modeling paths.
- Found regression was not suitable for selection.
- Found timing-derived features give near-perfect accuracy but are leakage and not deployable.
- Scaled to real-world multi-domain data.
- Handled severe class imbalance and timing noise.
- Evaluated with runtime-sensitive metrics, not accuracy alone.
- Found the current feature space detects timsort-friendly cases well but cannot reliably separate introsort from heapsort.

## Project-Specific Facts

- Thesis title: Machine Learning Based Algorithm Selection for Sorting Numeric Arrays.
- Candidate algorithms in final production portfolio: introsort, heapsort, timsort.
- Implementation basis: C-level sorting via NumPy/Python standard library behavior.
- Final feature count in methodology/results: 16 structural features.
- Feature extraction goal: lightweight structural scan cheaper than sorting in intended range. Note: some implemented features use `np.unique`, percentile, or sampled inversion logic, so describe as “cheap structural features” rather than pretending every operation is strictly one-pass.
- Dataset size: 1,188,265 numeric arrays.
- Domains: Formula 1 telemetry, stock market, cryptocurrency, earthquake/seismic, weather.
- Raw class distribution:
  - timsort: 1,010,413 arrays, 85.0%
  - heapsort: 125,218 arrays, 10.5%
  - introsort: 52,634 arrays, 4.4%
- Production model: XGBoost v5.
- Production test accuracy: 76.1%.
- Main runtime result: 93.1% SBS-to-VBS gap closed.
- Zero-regret predictions: 89.6%.
- Mean per-instance regret: about 0.23 microseconds.
- Timsort recall in production model: 94.5%.
- Main weak boundary: introsort vs heapsort.
- Second model track: F1-specific channel-flag / per-channel models.
- F1 channel flag examples: Speed, Throttle, RPM, nGear, DRS, Distance, X, Y, Z.
- F1 channel classifier experiment: predicts channel from features with ~99.4% test accuracy on balanced F1 channel dataset (`results/xgboost_f1_channel/evaluation_results.json`).
- F1 dynamic routed model non-strict evaluation: routed acc ~0.860, channel SBS ~0.507, global SBS ~0.200 (`results/f1_9_channel_models_dynamic_v2/router_eval.json`).
- Latest F1 strict routed artifact: routed acc ~0.489, channel SBS ~0.289, global SBS ~0.255 (`results/f1_9_channel_models_dynamic_v2_strict/strict_router_eval.json`). Notes contain an earlier strict run around ~0.554; use JSON artifact unless intentionally discussing run history.
- Interpretation: F1-specific routing with explicit channel flags is a domain-specific experiment, not the general cross-domain production model.
- F1 branch used a different “actual_5” algorithm label set in later experiments: `quick_sort`, `introsort`, `merge_sort`, `heap_sort`, `shell_sort`. This is not the same as v5’s 3-class NumPy portfolio.

## Data Journey

Early work began with synthetic arrays to prove there was a real selection problem:

- Synthetic arrays varied size, distribution, and structure.
- VBS-SBS gap showed selection was worth attempting.
- Counting sort had 0 wins in early benchmarks and was dropped.
- Radix/custom Python-loop sorts were not competitive with C-level implementations.

Real-world validation then exposed more complexity:

- F1 fastest-lap arrays were too small; timsort dominated.
- F1 full-race telemetry produced larger arrays and real 3-way competition.
- Finance/seismic/weather domains often had one dominant algorithm, so aggregate domain-level VBS-SBS gap was small.
- Mixed structural workloads restored selection value.

Final large dataset:

- 1.18M arrays from 5 real-world domains.
- Structural transforms used to create broader structural coverage:
  - RAW: original order
  - REV: reversed
  - SHUF: shuffled
  - QBIN50: quantized into 50 bins
  - PSORT10: sorted then perturbed by swapping 10% of elements
- Every row had features plus measured runtimes for all three algorithms.
- Best measured runtime became label.
- Near-tie/noisy rows handled with margin filtering.
- Class imbalance handled with undersampling and inverse-frequency weighting.

Later honesty/audit branch:

- `scripts/fetch_diverse_data.py` and `scripts/train_xgboost_v6.py` created a smaller but stricter dataset: 37,976 arrays, 10,792 sources, 9 domains, GroupShuffleSplit by `source_id`.
- v6 result: 71.2% test accuracy, 66.4% balanced accuracy, 45.6% test gap closed, 72.0% zero regret.
- v6 notes argue some v5 numbers may be optimistic because v5 had weaker source separation. This is internal context for Q&A; do not volunteer it unless needed, but be ready to explain source-aware evaluation honestly.

## Model Journey

- Early v1 regression: failed as selector despite high runtime prediction quality. It learned absolute time dominated by size, not relative winner.
- Early v2 classifier: better because it directly learned the fastest-algorithm label.
- Early v3 timing/pairwise features: near-perfect accuracy but not deployable because features require running algorithms first. Use only as leakage/ceiling proof.
- v5 production/submitted model: balanced XGBoost classifier on large real-world dataset. Chosen final thesis model.
- v6 honest diverse audit: smaller, source-aware, GroupShuffleSplit model. Lower headline numbers but important evidence that the hard limit is not just tuning.
- v7 regret-aware training: rejected. Standard argmax on v7 split had ~72.9% accuracy and ~78.1% gap closed; regret-aware decision mode dropped to ~59.8% accuracy and ~66.0% gap closed.
- v8 binary cascade: rejected as architecture, but important finding. Stage 1 timsort vs rest AUC = 0.9818; Stage 2 introsort vs heapsort AUC = 0.6034.
- F1 channel classifier: predicts one of 9 F1 telemetry channels from structural features with ~99.4% test accuracy on a balanced F1 dataset.
- F1 channel-flag models: separate model track for F1 only. A known channel flag can route an F1 array to a channel-specific model (`scripts/predict_f1_by_channel_flag.py`). This is not used for all domains.
- F1 dynamic routed/channel models: non-strict results looked strong; latest strict artifact is much weaker but still above its baselines. Treat as F1-only specialization plus leakage/generalization lesson, not as global replacement for v5.
- LinUCB/contextual bandit: designed/scripted, not empirically validated. Do not claim as validated contribution in defense.

## Key Interpretation

Accuracy alone understates value. The model is not perfect at matching oracle labels, but most wrong labels happen when algorithms have almost same runtime. That is why 76.1% accuracy can still close 93.1% of the runtime gap.

The high-stakes decision is usually whether timsort should be used. Timsort-friendly inputs have visible structure: high sortedness, long runs, low entropy, strong repetition. The model detects those well.

The hard decision is introsort vs heapsort. Their differences often depend on low-level runtime effects such as cache behavior, branch prediction, and memory access patterns, which are not captured by current structural features.

Most important defense nuance: submitted thesis presents v5 as main production model. Repo also contains later honesty checks. If committee asks about leakage or robustness, answer with source-aware care:

- We recognized source leakage risk.
- We used/ran stricter source-aware checks.
- Strict checks reduce headline accuracy, but preserve the same interpretation: timsort-vs-rest is learnable; introsort-vs-heapsort is structurally weak and low-regret.

## Slide Guardrails

- No standalone “Literature Review” section. Follow advisor direction: introduce related work only while explaining task importance and positioning.
- Related work format should be short “who did what” lines, e.g. Rice formalized algorithm selection; SATzilla applied feature-based solver selection; AlphaDev discovers new sorting routines; Bai & Coester modify sorting using predictions; this thesis selects among existing sort algorithms before execution.
- When explaining models, distinguish clearly:
  - General model = v5, all domains, 3 algorithms.
  - F1-specific model = channel classifier or channel-flag/per-channel route, F1 only, later using an “actual_5” algorithm set in some experiments.
- Do not say “online bandit was validated.” Say it is future work / designed but not empirically validated.
- Do not present F1 routed/channel models as universal success. Present as F1-only specialization; strict split shows limited generalization.
- Do not claim 100% or near-perfect deployable accuracy from v3. It used timing leakage.
- Do not overstate aggregate speedup for single domains; many real domains had low SBS-VBS gap.
- Do not hide that v6/source-aware checks exist if asked. Frame them as rigorous follow-up confirming the main limitation rather than contradiction.
- Do not mix 22-feature literature discussion with final 16-feature methodology without explanation.
- Do not make deck too literature-heavy.
- Do not show every feature formula unless committee asks.
- Defend 76.1% accuracy by immediately pairing it with regret/gap closed.

## Integrated Committee-Risk Handling

Do not create a separate backup deck. The defense should be one complete 22-slide talk. Committee-risk material must be integrated into the main story only where it helps the argument, or handled orally in Q&A.

- Source leakage / v6 source-aware check:
  - Mention only inside the strict-checks slide or if asked.
  - State that stricter evaluation lowers headline metrics but preserves the same interpretation.
- v7 regret-aware failure:
  - Use briefly as evidence that attractive ideas were rejected when they hurt performance.
- v8 binary cascade:
  - Use as the main evidence that the hard boundary is timsort-vs-rest versus introsort-vs-heapsort.
- F1 channel classifier and routed models:
  - Mention only as a specialization caveat.
  - Keep separate from v5 and do not present it as the global selector.
- Feature formulas and extraction cost:
  - Show only the regret/SBS/VBS/gap formulas in the talk.
  - Explain detailed feature formulas orally only if asked.
- Model and portfolio choices:
  - Answer orally if asked: why XGBoost, why classification, why three algorithms, why counting/radix/custom Python-loop sorts were excluded.
- LinUCB/contextual bandit:
  - Mention only as designed/future work, not validated thesis evidence.

## Figure Mapping

Prefer existing thesis/methodology figures unless a slide needs a cleaner custom layout. Do not invent new results during PPTX creation.

- Architecture slide: `methodology_assets/F1_system_architecture.png`
- Dataset/transforms slide: `methodology_assets/F8_data_pipeline.png`
- Feature groups slide: `methodology_assets/F2_feature_extraction.png`
- Model evolution slide: `methodology_assets/F3_model_evolution.png`
- Main runtime result slide: `methodology_assets/F11_accuracy_vs_gap_closed.png` or `results/thesis_figures/06_vbs_sbs_model_comparison.png`
- Regret metrics slide: `methodology_assets/F9_regret_distribution.png` or `results/thesis_figures/05_regret_analysis.png`
- Confusion/per-class behavior slide: `methodology_assets/F5_confusion_matrix.png`, `results/thesis_figures/03_v5_per_algorithm_recall.png`, or `results/thesis_figures/08_confusion_matrix.png`
- Feature importance slide: `methodology_assets/F4_feature_importance.png` or `results/thesis_figures/04_feature_importance.png`
- Domain holdout slide: `methodology_assets/F10_domain_holdout_gap_closed.png` or `results/thesis_figures/07_domain_holdout.png`
- v5-v6 source-aware check: `results/thesis_figures/10_v5_vs_v6_comparison.png` only if it fits the integrated strict-checks slide.
- F1 routed-model figures: `results/f1_dynamic_figures/01_strict_metrics.png`, `02_channel_balanced_accuracy.png`, `03_pipeline.png`, `04_strict_split_sizes.png` only if they are needed for the integrated specialization caveat.
- Use `results/thesis_figures/*` when the thesis-style figure is clearer or more presentation-ready than the methodology version.
- Do not add extra slides just because a figure exists.

## Likely Committee Questions

- Why XGBoost instead of neural networks?
- Why classification instead of runtime regression?
- Why only three sorting algorithms?
- Why exclude counting/radix sorts?
- How is feature extraction cheaper than sorting?
- What prevents data leakage?
- Why is 76.1% accuracy acceptable?
- What are SBS and VBS?
- Why use gap closed and regret?
- Does the model generalize across domains?
- What happens for small arrays?
- Why is introsort vs heapsort difficult?
- Is online adaptation validated?
- Are there one model or two models?
- Is F1 channel routing part of the general selector?
- Did you test source leakage?
- What are the main limitations?

## Short Answer Bank

- XGBoost: strong for medium-sized tabular data, interpretable feature importance, low inference overhead.
- Classification vs regression: regression learned absolute runtime scale; classification learned winner boundaries.
- Three algorithms: final portfolio kept production-grade C-level comparison sorts to avoid implementation-level unfairness.
- Counting/radix: early experiments showed they were not competitive under thesis constraints or required assumptions outside scope.
- Feature overhead: features are mostly one-pass O(n), while sorting is O(n log n); very small arrays should bypass selector.
- 76.1% accuracy: many label errors are low-cost near-ties; runtime metrics show 93.1% gap closed.
- SBS: best single fixed algorithm over dataset.
- VBS: oracle that picks fastest algorithm per array; upper bound, not deployable.
- Generalization: leave-one-domain-out kept gap closed high across held-out domains, though F1/weather show limits.
- Two model tracks: v5 is the general 3-algorithm selector; F1 channel models are domain-specific specialization using channel identity.
- Source leakage: strict source-aware experiments were run; metrics decrease, but the central finding remains that timsort detection is strong and introsort/heapsort separation is weak.
- Main limitation: current features do not observe hardware-level effects needed for introsort/heapsort separation.

## V3 Rebuild Correction Plan

This section is mandatory before any further PPTX generation. The previous v2 deck is treated only as a rough content/script draft, not as the final visual or structural standard.

### Non-Negotiable Corrections

- The title slide must use the submitted thesis title exactly:
  - `MACHINE LEARNING BASED ALGORITHM SELECTION FOR SORTING NUMERIC ARRAYS`
- Do not invent new thesis framing terms for visible slides when the submitted thesis already has terms for the concept.
- Use thesis terminology consistently:
  - Algorithm Selection Problem
  - sorting numeric arrays
  - structural array features
  - feature-based instance characterisation
  - runtime-based label generation
  - XGBoost classifier
  - Single Best Solver (SBS)
  - Virtual Best Solver (VBS)
  - regret
  - gap closed
  - leave-one-domain-out testing
  - general, domain-specific, and routed specialised models
  - routed specialised models for F1 telemetry
- Do not use invented visible phrases such as “runtime value” as a replacement for thesis terms unless it is only spoken explanation.
- Do not define the thesis from scratch using new wording if the submitted thesis already defines it. Pull wording from the 85-page thesis and compress it for slides.
- No row of KPI/result cards on the opening title slide. A defense opening should establish thesis title, author, institution, supervisor, and the research direction. Headline results appear after method/evaluation setup.
- Remove left-side footer section labels from all slides. If a footer is used, keep it minimal: slide number only, or slide number plus very subtle date/institution.
- No decorative short title underlines. Use full template bars, clear spacing, or no rule.
- No visible machine-planning words or internal planning vocabulary.

### Correct Opening Standard

The first slide should not look like a dashboard. It should look like an official MSc defense title slide:

- Thesis title exactly as submitted.
- Ahmed full name as submitted: `AHMED SALAH ABDALHI MOHAMMED`.
- Degree/program: Master of Science (MSc) in Computer Engineering.
- Cyprus International University.
- Supervisor: `Assoc. Prof. Dr. Emre Özbilge`.
- Date: June 2026 / Nicosia 2026 as appropriate.
- CIU logo must be visible and properly placed on the title slide.
- CIU identity on the title slide should use the strong colored top bar: `CYPRUS INTERNATIONAL UNIVERSITY` in one line on the left, with the CIU logo on the same bar.
- Visual cue may hint at sorting/algorithm selection, but should not compete with the formal title.

### CIU Identity And Background Requirement

- Use the CIU logo asset:
  - `/Users/ahmed/Downloads/ciu-logo-tranpernt.png`
  - Transparent PNG, 709 × 709.
- Copy the logo into the PPT project `images/` directory during the next build so the deck is self-contained.
- Use CIU identity formally:
  - title slide: visible CIU logo, official institution name, supervisor/date;
  - closing slide: small CIU logo or formal CIU identity mark;
  - body slides: optional very subtle institutional mark only if it improves formality, not on every slide if it clutters the page.
- The title slide must have an intentional official-defense background, not a plain empty canvas and not a dashboard.
- The title slide must use the full page and center the thesis title. The title may stretch wide across the slide. Do not left-align the main title block.
- The title slide must say `Presented by` and `Supervised by`.
- Do not put Nicosia/location text, date/time text, decorative array imagery, or unnecessary degree sentence such as `A thesis for...` on the title slide.
- Acceptable background directions:
  - clean academic white with a large low-opacity CIU mark/watermark;
  - restrained CIU color accent band using the logo’s orange/maroon/red palette;
  - subtle geometric background inspired by sorting/array structure, kept behind the formal title;
  - full academic cover composition from `academic_defense` template with CIU branding integrated.
- Do not use generic gradient blobs, decorative orbs, or unrelated AI-style backgrounds.
- Do not let the background reduce readability of the submitted thesis title.
- If using Anthropic/Claude brand as a visual design reference, it must not replace CIU identity. CIU is the official presentation brand; Anthropic style can only influence restraint, spacing, and typography.

### Thesis-Language Source Rule

Before rebuilding each slide, map the slide to the relevant thesis section and reuse its terminology:

- Task/motivation: Chapter 1 and sections 2.3, 2.5.
- Related work inside task framing: sections 2.5, 2.6, 2.8, 2.9, 2.10.
- Feature explanation: sections 2.7 and 3.4.
- Methodology: sections 3.1-3.7.
- Dataset and labelling: section 3.5.
- Model: section 3.6.
- Evaluation: section 3.7 and section 2.13.
- Domain generalisation and routed models: section 2.12 plus later result artifacts.

If a phrase is not in the thesis, do not put it on the slide unless it is a simple connective phrase. Prefer thesis terms over new agent-written terms.

### PPT Master Usage Standard

The next deck must use `ppt-master` as a full harness, not only as an export pipeline:

- Re-read the current `ppt-master` skill and animation workflow before execution.
- Use the academic defense template geometry deeply, not only its colors.
- Use the selected brand/layout templates as design systems:
  - vary slide compositions;
  - use title, chapter, content, figure-led, comparison, and closing patterns intentionally;
  - avoid repeating the same header/card/footer shell.
- Use chart and infographic templates where they match the content:
  - agenda / defense route;
  - pipeline;
  - layered architecture;
  - chevron process;
  - comparison / pros-cons;
  - KPI only when it is truly a result slide, not as a title-slide decoration.
- Use image-as-evidence layouts for thesis figures. Do not place figures as small thumbnails in generic boxes when the figure is the main evidence.
- Add coverage notes or template mappings when image-led pages are intentionally used.

### Visual Construction Standard

The next build must not represent important ideas using only generic boxes, arrows, and dashboard cards. `ppt-master` can produce detailed SVG scenes, visual systems, generated assets, formulas, chart structures, and object-level animation. Use that capability.

- Use the full slide canvas. Do not default to left-aligned or right-aligned layouts with empty unused space.
- Prefer centered, radial, layered, full-width, full-height, or stage-like compositions that make the whole page participate in the explanation.
- A left/right split is allowed only when the content itself demands comparison or before/after structure. It must not be the default composition.
- Title slides, task slides, methodology scenes, evaluation explanations, and main-result slides should feel built for the full 16:9 page, not like a document section pasted onto one side.
- Avoid “text column on one side, diagram on the other side” unless there is a clear speaking reason.
- Treat method/task slides as visual explanations, not box diagrams.
- Build important slides as rich semantic scenes:
  - arrays shown as actual sequences / distributions / sortedness patterns;
  - candidate algorithms shown as competing paths or lanes;
  - feature extraction shown as a concrete transformation from an array into structural measurements;
  - the selector shown as a model decision surface / routing mechanism, not only a rectangle labelled “model”;
  - SBS/VBS/regret shown as a measured runtime landscape, not only formula boxes.
- Use generated or custom SVG visual assets where existing thesis figures are too small, too dense, or too paper-like for oral defense.
- Use `ppt-master` image acquisition/generation when a slide needs a background, conceptual scene, or consistent spot illustrations.
- Use formulas rendered through the formula pipeline only where mathematical fidelity matters.
- Use icons only when they clarify a visual role; do not rely on icons as decoration.
- Use chart templates and custom calibrated charts for real result figures when the existing PNG is not presentation-ready.
- Use full-slide or large-region visual compositions for key ideas, especially:
  - title/official opening;
  - Algorithm Selection Problem;
  - input-dependent sorting performance;
  - structural array features;
  - runtime-based label generation;
  - system architecture;
  - SBS/VBS/regret/gap closed;
  - main result;
  - strict checks and limitations.
- Avoid the weak pattern: title bar + three boxes + footer. That pattern can appear only when it is truly the best structure for a comparison slide.
- Every major visual slide must answer: “What would Ahmed point at while speaking?” If there is nothing to point at except boxes, redesign it.

### Prepared Micro-Asset Kit

Before the final deck rebuild, use the prepared micro-assets in:

- `/Users/ahmed/Desktop/thesis/My-Master-thesis/pptx-micro-assets/`

Current assets:

- `components/01_ciu_cover_background.svg`
  - official CIU defense cover background;
  - CIU logo and watermark;
  - exact thesis title area;
  - subtle sorting-array visual texture.
- `components/02_algorithm_selection_scene.svg`
  - detailed Algorithm Selection Problem scene;
  - input array, structural highlights, feature vector, algorithm portfolio, selector decision;
  - animation-ready groups.
- `components/03_structural_feature_extraction.svg`
  - feature-based instance characterisation visual;
  - thesis feature groups and 16-feature output;
  - animation-ready groups.
- `components/04_runtime_label_generation.svg`
  - runtime-based label generation visual;
  - transforms, measured runtimes, fastest measured algorithm, assigned label;
  - animation-ready groups.
- `components/05_sbs_vbs_regret_landscape.svg`
  - SBS/VBS/regret/gap-closed visual landscape;
  - supports explaining why accuracy alone is insufficient;
  - animation-ready groups.

When rebuilding the deck, either reuse these SVGs directly as editable components or rebuild their structure into final slide SVGs. Do not ignore them and recreate weaker generic boxes.

### Task Slide Redesign Requirement

The old page 3 task slide is rejected as too generic. The next task slide must be rebuilt as a strong visual explanation of the thesis task:

- Use the thesis concept: `Algorithm Selection Problem` applied to sorting numeric arrays.
- Show a real-looking numeric array input, with visible structural properties:
  - existing order / disorder;
  - duplicated values;
  - local runs;
  - value spread.
- Show feature extraction as measurable structural array features, not as a vague “cheap features” box.
- Show the algorithm portfolio as three competing candidate solvers:
  - Introsort;
  - Heapsort;
  - Timsort.
- Show the selector choosing one algorithm before execution.
- Make the visual detailed enough that Ahmed can explain the full task by pointing through the slide.
- Use staged animation:
  1. input array appears;
  2. structural signals are highlighted;
  3. feature vector appears;
  4. three candidate algorithms appear;
  5. selector chooses the predicted fastest algorithm.
- Do not use this slide to show final results. It is for task understanding only.

### Animation Requirement

The final PPTX must not be only static slides with fade transitions.

Animation should be formal and defense-appropriate:

- Title slide: subtle staged entrance of title, author/institution, supervisor/date.
- Defense route: reveal sections one by one.
- Task slide: reveal input array, feature extraction, selector, and algorithm choice in order.
- Dataset/labelling: reveal data sources, transforms, runtime measurement, label generation.
- Feature slide: reveal feature groups in the same order Ahmed speaks.
- System slide: reveal array features -> XGBoost classifier -> selected algorithm.
- Evaluation slide: reveal SBS, VBS, regret, gap closed in sequence.
- Main result slide: reveal `93.1% gap closed` first, then `76.1% accuracy`, then `89.6% zero regret`.
- Confusion/feature/domain slides: figure first, then interpretation callouts.
- Rigor slide: reveal v6, v7, v8 one at a time.
- Limits slide: reveal “I claim” before “I do not claim”.

Use `animations.json` with real SVG group IDs. Export with `svg_to_pptx.py` animation config or appropriate `-a auto` plus per-slide overrides. Validate the animation config before export.

### Content-Led Deck Length

Do not force the deck to 22 slides or any other fixed count. The next build must be content-led:

- Use as many main slides as the official defense story needs inside the 25-minute maximum.
- Prioritize quality of content, clarity of argument, and Ahmed’s ability to speak naturally.
- Do not compress complex thesis content just to fit a previous slide count.
- Do not add filler slides just to make the deck look larger.
- Split a slide when one slide would contain two speaking ideas, two major figures, or too much committee-risk material.
- Merge or remove a slide when it repeats an idea already clear from the previous slide.
- Still no separate backup appendix unless Ahmed explicitly asks.
- Practical expectation: likely around 24-30 main slides, but the final count must come from the content, not from a hard rule.

### External Standard Check

Before the rebuild, check current credible guidance/examples for thesis defense presentations and apply only stable principles:

- official title slide format;
- sparse slide text;
- one central idea per slide;
- staged explanations rather than dense pages;
- results shown after methodology/evaluation setup;
- animations used to guide attention, not decorate.

Do not imitate random online visual styles blindly. Use external guidance only to raise the academic defense standard.

### V3 Build Gate

Do not start V3 generation until these are done:

- Confirm exact title slide metadata from the thesis.
- Rewrite the visible slide titles using thesis terminology.
- Replace invented slide phrasing with submitted-thesis wording where possible.
- Remove footer section labels from the design spec.
- Define per-slide template mapping and per-slide animation intent.
- Decide the slide count from the rewritten content plan, with no fixed target number.

## Ahmed Repeated Preference Pattern

These are not optional style suggestions. They are extracted from repeated feedback and must guide every next PPTX decision so Ahmed does not have to repeat them.

### Overall Standard

- The deck must feel like an official MSc thesis defense, not a dashboard, not a generic AI-generated report, and not a web-app UI.
- Use the full 16:9 slide canvas. Do not waste half the page with lazy left/right alignment.
- Avoid generic box-and-arrow diagrams unless the content truly needs them. Important concepts need rich visual scenes that Ahmed can point through while speaking.
- Do not invent new visible terminology when the 85-page submitted thesis already has the wording.
- Every visible phrase should either come from the thesis, be a necessary short connective phrase, or be a formal defense label.
- The deck should support Ahmed speaking naturally. Slides should not force him to explain around awkward wording or machine-style labels.
- Quality of content and visual explanation matters more than hitting a fixed slide count.

### First Slide Pattern

- Use the exact submitted thesis title:
  - `MACHINE LEARNING BASED ALGORITHM SELECTION FOR SORTING NUMERIC ARRAYS`
- The title can be split across lines for layout, but the words must not be changed, shortened, or rephrased.
- Title should not be huge and cramped in the middle. Prefer a slightly smaller font that lets the title stretch wider across the page.
- Do not left-align the main title block. Use a formal centered/wide composition that takes advantage of the full page.
- Keep the colored CIU top bar and bottom bar.
- Put `CYPRUS INTERNATIONAL UNIVERSITY` in one line on the top colored bar, left side.
- Put the CIU logo on the same top bar.
- Use a subtle CIU watermark if useful.
- Include:
  - `Presented by`
  - `AHMED SALAH ABDALHI MOHAMMED`
  - `Supervised by`
  - `Assoc. Prof. Dr. Emre Özbilge`
- Do not include:
  - Nicosia/location text;
  - date/time text;
  - `A thesis for...` degree sentence;
  - decorative array image/texture;
  - dashboard KPI/result cards.

### Layout Pattern

- Use one unified slide chrome across the deck:
  - top bar: CIU maroon `#8F1D3A` from y=0 to y=18;
  - accent bar: CIU orange `#F4511E` from y=18 to y=28;
  - bottom bar: CIU maroon `#8F1D3A` from y=692 to y=720.
- Do not place `CYPRUS INTERNATIONAL UNIVERSITY` text inside the top bar by default.
- Use CIU logo/identity only where needed, especially title and closing; keep body slides clean.
- Do not default to left-aligned or right-aligned layouts.
- Do not default to “text on one side, visual on the other.”
- Use centered, radial, layered, full-width, full-height, or stage-like compositions.
- A left/right split is allowed only for a real comparison, before/after explanation, or two-track argument.
- If a slide has a major idea, the visual should occupy the page, not sit as a small object in a box.
- Do not use page-title short underlines. Use full bars, spacing, or no rule.
- Remove left footer labels. Footer should be minimal: slide number only, or extremely subtle institutional detail if needed.

### Content Pattern

- Do not start with results. Results should come after task, data, method, and evaluation setup.
- No row of result cards on the title slide.
- Use thesis terms:
  - Algorithm Selection Problem;
  - sorting numeric arrays;
  - structural array features;
  - feature-based instance characterisation;
  - runtime-based label generation;
  - XGBoost classifier;
  - Single Best Solver (SBS);
  - Virtual Best Solver (VBS);
  - regret;
  - gap closed;
  - leave-one-domain-out testing;
  - general, domain-specific, and routed specialised models;
  - routed specialised models for F1 telemetry.
- Do not replace thesis terms with invented softer phrases such as “runtime value” on visible slides.
- Do not add unnecessary definitions from scratch. Compress the thesis wording instead.
- Separate the two model tracks clearly:
  - v5 general selector across all domains and three algorithms;
  - F1-specific/channel/routed models as separate specialization.
- Do not mix F1 routing with the main v5 contribution.

### Visual Detail Pattern

- Task slide must show the actual Algorithm Selection Problem visually:
  - numeric array;
  - structural properties;
  - feature vector;
  - algorithm portfolio;
  - selector decision before execution.
- Feature slides must show real feature groups and extraction, not just “features” in a box.
- Runtime labelling must show measured runtimes becoming the fastest-algorithm label.
- Evaluation slides must make SBS/VBS/regret/gap closed visually understandable before presenting headline numbers.
- Main result slide should reveal one main result first, not show a row of equal KPI cards.
- Use prepared micro-assets from `pptx-micro-assets/` unless a better version is intentionally built.

### PPT Master Pattern

- Use `ppt-master` as a full harness, not just an SVG-to-PPTX exporter.
- Use templates deeply for structure and visual rhythm, not only colors/fonts.
- Use image/formula/chart generation and custom SVG construction where it improves explanation.
- Add real object-level animation with meaningful group IDs.
- Animations should support Ahmed’s speaking order:
  - reveal structure first;
  - then evidence;
  - then interpretation;
  - then result.
- Keep animations formal and academic, not flashy.

### Rebuild Discipline

- Before building, rewrite the content plan with thesis terms and slide-by-slide visual intent.
- Do not build from a weak previous version just because it exists.
- Do not patch around bad slides if the idea needs a complete redesign.
- If a slide looks like generic boxes, stop and redesign it before exporting.
- If a slide would embarrass Ahmed in an official defense, it is not acceptable as “good enough.”

### Official Versioning Rule

- Current official working deck: `projects/ahmed_defense_v5_ppt169_20260630/exports/ahmed_defense_1.2.3.pptx`.
- Latest accepted patch: `1.2.3` preserves the `1.2.2` slide order and layout, then grounds visible wording more directly in the submitted thesis. Slide 5 now uses actual literature anchors from the thesis (`Rice`, `SATzilla`, input-sensitive sorting, `AlphaDev`, and `Sorting with Predictions`) instead of generic background buckets. Slides 2, 6-9, 13-19, and 21-24 were revised toward thesis terms: empirical fastest-algorithm labels, sixteen `O(n)` features, SBS/VBS/regret/gap closed, model-evolution evidence, LODO results, v5 as the reported selector, and limitations/future work from Chapter Five. The non-thesis `79.7% weighted gap closed` wording was removed and replaced with the thesis-grounded LODO statement: gap closed exceeds 75% in every fold and F1 telemetry is the weakest held-out fold.
- Rejected version: `projects/ahmed_defense_v5_ppt169_20260630/exports/ahmed_defense_1.1.0.pptx` because it changed too much, touched slide 1/2, and broke the preferred clean style.
- Future edits must work on top of the latest `ahmed_defense_*.*.*.pptx` version, preserving Ahmed's manual changes.
- Do not overwrite older versioned files unless Ahmed explicitly asks.
- For each accepted deck update, save a new semantic version:
  - patch version for small visual/content fixes, for example `1.0.1`;
  - minor version for slide-level additions/reordering, for example `1.1.0`;
  - major version only for a major redesign, for example `2.0.0`.
- Update from July 1, 2026: future deck updates should produce PPTX only. Do not export PDF unless Ahmed explicitly asks for a PDF.

## Highest-Probability 2026 Use Cases From XGBoost Paper Context

For defense framing, the five strongest use-case connections from the XGBoost paper list are:

- Any pipeline where the fastest algorithm depends on input structure: very high probability.
- Any system choosing among multiple algorithm implementations before execution: very high probability.
- Domain-specific feature-engineered pipelines: high probability.
- General classification benchmarks: high probability.
- User-defined objective / custom machine-learning pipeline: high probability.



Financial rolling windows at scale involve continuously sliding a fixed time frame over massive datasets to compute moving statistics (e.g., 30-day volatility, 200-day moving averages, multi-asset correlations). Processing this efficiently at scale prevents rebuilding metrics from scratch every time new data arrives

In information retrieval and machine learning, top-K scoring identifies the K items (documents, products, or vectors) with the highest predicted relevance, classification, or similarity scores. It acts as a cutoff filter, retrieving only the top fraction of a dataset for final ranking, evaluation, or user display

Timestamp sorting in databases (DBs) provides several key performance and architectural benefits, primarily revolving around efficient data retrieval, concurrency control, and chronological analytics