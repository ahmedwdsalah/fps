# Ahmed Defense Script for `ahmed_defense_1.3.11`

Target delivery: about 22-24 minutes. Do not rush. If the committee interrupts, skip the optional sentences and keep the main sentence of each slide.

This is a rehearsal script, not text to read word-for-word. The goal is that what I say matches exactly what the committee sees on the slide.

## Slide 1 - Title

Time: 30-40 seconds

Good morning everyone. My name is Ahmed Salah Abd Alhi Mohammed, and today I will present my MSc thesis titled **Machine Learning Based Algorithm Selection for Sorting Numeric Arrays**.

The work is about a practical question: before sorting a numeric array, can we predict which sorting algorithm is likely to be fastest for that specific input?

So the thesis is not trying to design a new sorting algorithm. It is about choosing among existing sorting algorithms using cheap information extracted from the input array.

## Slide 2 - Research Objective

Time: 50-60 seconds

The objective is to learn a selector, written here as `S: f(x) -> A`.

Here, `x` is the input array, `f(x)` is the structural feature vector extracted from that array, and `A` is the algorithm selected from the portfolio.

The target is not only to predict labels. The real target is to reduce expected regret compared with the oracle choice. In other words, if the model is wrong, I care about how expensive that mistake is.

This objective has four parts. First, extract compact structural features. Second, train an offline XGBoost model. Third, evaluate under source-aware and leakage-resistant protocols. Fourth, keep the whole pipeline reproducible in Python.

## Slide 3 - Prediction Before Execution

Time: 55-70 seconds

This slide defines the boundary of the task.

At prediction time, the selector is allowed to look at the array and compute cheap `O(n)` structural features. It can measure order, runs, repetition, distribution, and rough scale.

But it is not allowed to sort the array with introsort, heapsort, and timsort and then choose after seeing all runtimes. That would be the oracle, not a usable selector.

So the decision must happen before execution. The model predicts one of the portfolio algorithms, and later evaluation compares this prediction against the true fastest algorithm measured offline.

That boundary is important because it prevents the system from using timing information that would not exist in real deployment.

## Slide 4 - Algorithm Selection Problem

Time: 60-75 seconds

This is the formal shape of the problem.

On the left, we start with one numeric array. From that array, I extract structural features: size, sortedness, runs, duplicates, entropy, spread, and related signals.

The selector maps the feature space into the algorithm space. In this thesis the algorithm space contains introsort, heapsort, and timsort.

At the bottom, the performance space is only used to create labels and evaluate the model. Each algorithm is timed on a fresh copy of the same input, and the fastest measured algorithm becomes the label.

So the selector does not see runtimes during prediction. It only sees the structural signature of the array.

## Slide 5 - System Architecture

Time: 65-80 seconds

This slide shows the system architecture that connects the earlier formal problem to the actual implementation.

On the left side of the figure, the input is a numeric array. The first active step is feature extraction, where the array is converted into the sixteen structural features. Those features go into the XGBoost classifier, and the output is the selected sorting algorithm.

The three cards on the right show the boundary of the system. The main track is all domains and three algorithms. The prediction path is features into the XGBoost classifier. The separate track is the F1-specific routing experiment.

The important point here is that the architecture keeps training and prediction separate. Timing all algorithms is used to build labels offline, but during prediction the system only extracts features and runs one model prediction.

## Slide 6 - Dataset Domains

Time: 60-70 seconds

The corpus contains **1.18 million labeled array instances** from five real-world domains.

These domains are Formula 1 telemetry, equity prices, cryptocurrency OHLCV data, weather measurements, and seismic measurements.

The reason for using real data is that synthetic distributions were not enough to represent the kinds of structure that matter for sorting. Real numeric sequences contain trends, repetitions, local runs, noise, jumps, and domain-specific patterns.

So the dataset was designed to expose the selector to structurally diverse arrays, not just random arrays.

## Slide 7 - Domain Holdout

Time: 55-70 seconds

This slide shows the leave-one-domain-out result.

For each bar group, one domain is held out. The model trains on the other four domains and is tested on the unseen domain. So this is not the same as a random split inside the same source.

The chart reports both accuracy and gap closed. The result is not identical across crypto, earthquake, F1, stock, and weather. Some domains transfer better than others.

The point I want to make from this slide is that cross-domain behavior depends on whether the same structural signals appear in the unseen domain. When structure transfers, the selector still captures useful runtime value. When the domain is different, the result becomes weaker, and that is part of the limitation.

## Slide 8 - One Sample From Each Domain

Time: 60-75 seconds

This slide shows actual sampled traces from the five domains.

The important thing is that these are not synthetic examples drawn for the slide. They are examples from the data sources: stock market, cryptocurrency, weather, Formula 1 telemetry, and seismic events.

The traces already show why one generic random-array assumption is not enough. Some sequences are smoother, some are noisier, some have local movement, and some have stronger trend structure.

For the selector, this matters because the sorting workload is created by the input structure. A trace with visible order, repetitions, or local disorder can favor a different algorithm than a fully random-looking trace.

## Slide 9 - Sixteen Structural Features

Time: 70-85 seconds

Each array is converted into sixteen `O(n)` structural features.

The key rule is that these features must be cheap to compute and available before sorting. They describe the input, but they do not use candidate algorithm runtimes.

The features include size-related information, ordering and sortedness signals, run structure, duplicate and repetition signals, entropy-like distribution information, and spread or outlier information.

Timing-derived features are excluded at prediction time because they would leak information from the evaluation process into the model.

So this feature vector is the common representation used across training, validation, testing, and later prediction.

## Slide 10 - Feature Importance

Time: 60-75 seconds

This figure shows that the model is mainly using structural sorting signals.

Length, repetition, and ordering-related features account for much of the discriminative signal. That makes sense for this task. Timsort, for example, benefits strongly from natural runs and ordered structure. If that structure is visible in the features, the model can identify many timsort-friendly cases.

At the same time, this also explains a limitation. Introsort and heapsort can differ because of lower-level effects such as cache behavior and branch behavior. Those effects are not fully visible from simple structural features.

So the feature importance supports the main interpretation: the model works where the fastest algorithm leaves a visible structural trace.

## Slide 11 - Runtime Labels

Time: 65-80 seconds

The labels are empirical fastest-algorithm measurements.

For every array, each candidate algorithm is timed on a fresh copy. The minimum observed runtime becomes the label. This means the label is not chosen by theory or by assumption; it is measured.

The transformations at the bottom, such as raw, reversed, shuffled, quantized, and partially sorted versions, create different structural regimes. They force the selector to deal with non-trivial input structure.

This is important because algorithm selection only matters when the input structure creates a real decision boundary. If every input behaved the same way, there would be no meaningful selector to learn.

## Slide 12 - Training Dataset and Split

Time: 65-80 seconds

The raw collected corpus starts from **1,188,265** arrays.

Then near-tie filtering removes ambiguous labels. This matters because if two algorithms are almost equal, forcing one of them as the label can add noise without adding much practical value.

After that, the majority class is capped so timsort does not dominate the entire training process. Then inverse-frequency sample weights are used so minority classes still matter in the loss.

Finally, the data is split into train, validation, and test sets using a fixed seed. So the training setup is not only about fitting a model; it is also about controlling label noise and class imbalance.

## Slide 13 - The model learns a probability for each sorting choice

Time: 70-85 seconds

This slide is the start of the model explanation.

The left card is the input: `f(x)` in `R^16`, which is the sixteen-feature representation of the array. The middle card is the class probability model. The right card is the output: one probability score for introsort, one for heapsort, and one for timsort.

This is why the model uses `multi:softprob` with `mlogloss`. `multi:softprob` means the model outputs class probabilities. Then the selected algorithm is the one with the highest probability.

So the model is not asked to output a runtime value on this slide. It is asked to turn structural features into a three-class probability decision.

## Slide 14 - The training signal is balanced before the trees learn

Time: 70-90 seconds

The raw fastest-label signal is imbalanced. Timsort wins many cases, while introsort and heapsort are smaller classes.

If I trained naively, the model could learn a very simple behavior: predict timsort too often. That might give acceptable accuracy, but it would not be a disciplined selector.

So the training signal is controlled before the trees learn. Near-tie filtering removes unstable labels, the majority cap reduces dominance from the largest class, and inverse-frequency weights make minority-class examples contribute more to the loss.

The goal is not to make the dataset artificial. The goal is to stop the model from ignoring important but less frequent decision boundaries.

## Slide 15 - XGBoost corrects class scores using first and second order information

Time: 85-105 seconds

This slide explains what XGBoost is doing during training.

For each array, the model has raw class scores for introsort, heapsort, and timsort. Softmax turns those scores into probabilities.

Under `multi:softprob` and `mlogloss`, XGBoost compares the current probabilities with the true fastest label. Then it computes two pieces of information.

The gradient is first-order information. It tells the model the direction of the correction: should this class score go up or down?

The Hessian is second-order information. It tells the model about the curvature or strength of that correction: how careful should the update be?

This is why XGBoost is stronger than simply fitting errors. Each new tree is built using both direction and curvature, and those terms affect split decisions and leaf corrections.

## Slide 16 - The ensemble is constrained so it does not overfit timing noise

Time: 70-90 seconds

The model is powerful, so it needs constraints.

In the final setup, the ensemble uses many trees, but each tree is controlled by depth, learning rate, row subsampling, feature subsampling, regularisation, and the histogram tree method.

The row subsampling means each tree does not rely on exactly the same rows. The feature subsampling means each tree does not rely on exactly the same feature set.

The learning rate controls how big each correction step is, and regularisation prevents the leaves from becoming too specific to timing noise.

So the model is not just a large ensemble. It is a constrained ensemble designed to learn stable structural boundaries.

## Slide 17 - Strict checks define the boundary of the reported result

Time: 75-95 seconds

This slide shows one of the strict checks that defines the boundary of the reported result.

The chart compares the v5 production result with the v6 source-aware check across accuracy, balanced accuracy, and weighted F1-score.

The source-aware result is lower. That is expected, because it is a harder test. It reduces the chance that train and test examples are too close in source.

So I do not use this slide to pretend the stricter check improved everything. I use it to show that the thesis tested the result against leakage risk and kept the final claim bounded.

The reported result remains v5, but the source-aware check changes how strongly I state the claim. It makes the evaluation more honest.

## Slide 18 - Model evolution reduced the problem to its deployable form

Time: 75-90 seconds

This figure summarizes the model evolution.

The chart starts with v1 regression, then v2 classification, then v3 with timing leakage. The timing-leakage model reaches the ceiling, but it is not deployable because those timing signals are not available before prediction.

Then v5 becomes the production model. After that, v6, v7, and v8 are stricter or alternative variants: source-aware split, regret weighting, and binary cascade.

The important message is that the final choice was not simply the highest bar. The final choice had to be deployable, not leakage-based, and stable under evaluation.

That is why v5 is retained as the main model, while later variants explain the boundary and limitations of the result.

## Slide 19 - Evaluation measures distance from SBS to VBS

Time: 80-95 seconds

This slide defines the evaluation language used in the result slides.

The three boxes are SBS, VBS, and the model. SBS is the best fixed algorithm over the evaluation set. VBS is the oracle fastest algorithm for each array. The model chooses before execution using structural features.

The formulas at the bottom define regret and gap closed. Regret measures how far the model is from the oracle. Gap closed measures how much of the SBS-to-VBS improvement the selector captures.

This is why accuracy alone is not enough. If two algorithms are almost tied, a wrong class label may have very low runtime cost. Regret and gap closed measure that cost directly.

## Slide 20 - Accuracy alone is insufficient for algorithm selection

Time: 65-80 seconds

This slide shows the main evaluation idea visually.

A model can have moderate accuracy but still recover most of the runtime value, because many wrong choices happen near the oracle boundary.

So I do not interpret the selector only through top-1 accuracy. Accuracy tells me how often the class label matches the oracle. Regret tells me whether mistakes actually cost runtime.

This is why the thesis reports accuracy together with gap closed.

The practical question is not only “did the model copy the oracle label?” The practical question is “did the model avoid expensive sorting choices?”

## Slide 21 - The v5 selector closes most of the SBS-to-VBS gap

Time: 75-95 seconds

This is the headline result.

The final selector reaches **76.1% top-1 accuracy**, but the stronger evidence is the runtime value: it closes **93.1% of the SBS-to-VBS gap**, and **89.6% of predictions have zero regret**.

So the model is not perfect at reproducing the oracle label. But it is strong at avoiding expensive runtime mistakes.

This is the central result of the thesis: a compact structural selector can recover most of the practical runtime opportunity, even when the classification boundary is not perfectly solved.

That is why I present the result as a practical selector, not as a perfect oracle.

## Slide 22 - Confusion / Class Behavior

Time: 75-90 seconds

The confusion matrix explains where the model is strong and where it struggles.

Timsort has strong recall because timsort-friendly structure is often visible: runs, order, and repeated patterns leave clear signals in the feature vector.

The harder region is the introsort-heapsort boundary. These algorithms can be closer when the difference depends on lower-level execution effects that are not fully captured by the current features.

So I interpret the residual error as a feature-space limitation, not only a training failure.

The important point is that many of these boundary errors have low measured regret, which is why the runtime result is stronger than accuracy alone suggests.

## Slide 23 - F1 routing was treated as a separate specialization

Time: 65-85 seconds

This slide defines the second model track.

The right-side cards are the key part. The main track is the v5 general selector: all domains and three algorithms. The F1 specialization is separate: channel or flag routes F1 arrays to specialized channel models.

The diagram on the left shows that the F1 route adds a routing decision before the final predicted algorithm.

This is useful because F1 telemetry is structurally difficult and channel-dependent. But I do not mix this with the main v5 result.

So the claim boundary is clear: v5 remains the main cross-domain result, and F1 routing is a separate specialization experiment.

## Slide 24 - The contribution is a selector and an evaluation discipline

Time: 75-90 seconds

The contributions are threefold.

First, I built a deployable selector: sixteen cheap structural features are used with a gradient-boosted classifier to choose among introsort, heapsort, and timsort.

Second, I built an evaluation discipline around the selector. The thesis uses source-level separation, near-tie filtering, domain holdout testing, and regret-based metrics, because accuracy alone is incomplete for this task.

Third, I provide a reproducible artifact: one source-of-truth feature extractor, the data pipeline, and exported XGBoost models in JSON and ONNX formats.

So the contribution is not only the final percentage. It is the full method for building and evaluating a practical sorting selector.

## Slide 25 - The remaining errors show where the features stop seeing

Time: 75-95 seconds

The limitations are important.

The current portfolio is limited to three C-level comparison sorts. The absolute regret values are tied to one hardware and software environment. LinUCB was designed but not validated as a final empirical result. Also, feature extraction becomes less worthwhile for very small arrays.

The future work follows from these limits: a larger portfolio, validated online adaptation, stronger domain embeddings or channel-level routing, and cross-platform deployment with feature extraction overhead included.

The key point is that these limitations do not remove the main result. They define where the result should and should not be claimed.

The selector captures most of the practical runtime gap under the reported evaluation, but the next step is broader, hardware-aware, and adaptive selection.

## Slide 26 - Thank You / Questions

Time: 15-25 seconds

Thank you for listening.

I am happy to answer your questions.

If a difficult question comes, answer in this order:

1. Acknowledge the concern.
2. Point to the evidence.
3. State the limit clearly.
4. Explain why the contribution still holds.

Example: “Yes, that limitation matters. That is why I separated the general model from the F1 route and why I report regret and source-aware checks. The model is not a perfect oracle, but it still captures most of the practical runtime gap under the reported evaluation.”

## Short Emergency Version

If time is short, use this 12-minute path:

- Slides 1-5: problem, objective, boundary, formal selection, architecture.
- Slides 6-12: dataset, features, labels, split.
- Slides 13-16: XGBoost classifier, balanced signal, gradient/Hessian, constraints.
- Slides 17-18: strict checks and model evolution.
- Slides 19-22: regret, accuracy vs runtime, headline result, confusion matrix.
- Slides 23-26: F1 boundary, contributions, limits, questions.

For the short version, say only the first paragraph of each slide and keep the result slide almost unchanged.
