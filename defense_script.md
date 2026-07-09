# Ahmed Defense Script

Target: calm 22-24 minutes. At about 4 words per second, do not rush. Use this as a speaking script, not memorized word-for-word text.

## Slide 1 — Title

Good morning. My name is Ahmed Salah AbdAlhi Mohammed. Today I will present my MSc thesis, Machine Learning Based Algorithm Selection for Sorting Numeric Arrays, supervised by Assoc. Prof. Dr. Emre Ozbilge.

The main idea of this work is simple: sorting is usually taught as if we choose one algorithm and use it everywhere, but in practice, the fastest sorting algorithm depends on the structure of the input array. My thesis studies whether we can look at an array before sorting, extract cheap structural features, and predict which sorting algorithm should be used.

## Slide 2 — Why sorting selection matters at scale

This slide shows why the problem matters. In many systems, data must be ordered before later work can happen. Databases sort for queries and joins. Analytics systems sort timestamps and IDs. Ranking systems sort scores. Financial systems sort prices and signals. Telemetry pipelines sort sensor traces.

The operation is still sorting, but the input structure is not always the same. Some arrays are almost ordered, some have duplicates, some have bursts, some have noise, and some have long runs. If we always use one fixed algorithm, we ignore this structure. The goal is to choose before sorting is executed, because after sorting we already paid the cost.

## Slide 3 — Research Objective

The research objective is to learn a selector. I represent the problem as mapping structural information from an input array to an algorithm choice. The selector should reduce expected regret compared with an oracle that always knows the fastest algorithm.

The objective has four parts. First, build compact structural features from the array. Second, train an offline XGBoost model to predict the best algorithm. Third, evaluate in a leakage-resistant way, so the result is not just memorizing a source or a domain. Fourth, keep the work reproducible as a Python pipeline with saved models and result artifacts.

## Slide 4 — Prediction before execution

This slide is important because it defines what information the selector is allowed to use. Before sorting, I allow only features that can be extracted directly from the input array, such as order, runs, repetition, dispersion, and robust scale. These are structural signals.

What is not allowed at prediction time is running introsort, heapsort, and timsort, then selecting the fastest afterward. That would be cheating for deployment. Runtime measurements are used to create labels during evaluation and training, but the model prediction itself happens before any candidate sorting algorithm is executed.

## Slide 5 — Candidate algorithms

The action space is three algorithms: introsort, heapsort, and timsort. These are the labels the selector can output.

They are not random choices. Introsort gives a strong general-purpose quicksort-based behavior with heap fallback. Heapsort gives bounded memory and a fixed comparison-sort structure. Timsort is run-aware and can exploit existing ordered segments. Because they react differently to structure, the same numeric array can make one of them faster than the others.

So the model is not predicting a number. It is choosing one action from this portfolio.

## Slide 6 — System Architecture

This is the system architecture. The input is a numeric array. From that array, the pipeline extracts structural features. Those features become the feature vector. The XGBoost classifier uses that vector and predicts the best sorting algorithm.

The important part is the direction of the pipeline: input array, feature extraction, feature vector, classifier, predicted algorithm. The lower boxes show the meaning of the system: feature interaction, runtime supervision, and class decision. The model learns because the training label comes from measured runtime, but at prediction time it only sees the array features.

## Slide 7 — Corpus domains

The corpus uses five real-world domains. The total raw corpus is about 1.18 million labeled array instances. The domains include Formula 1 telemetry, equity prices, cryptocurrency OHLCV, weather, and seismic measurements.

This matters because synthetic random distributions were not enough for the problem. They are too clean and do not create the same structure as real sources. Real domains give different patterns: repeated values, ordered windows, spikes, smooth sequences, noisy changes, and domain-specific behavior. That gives the selector a more realistic test.

## Slide 8 — Sample arrays

Here I show concrete examples from the five domains. The point is not that the values themselves are meaningful to the model as finance or weather. The point is that each domain produces a different numeric trace.

Weather can contain smooth continuous changes. Formula 1 telemetry can have repeated or sensor-like patterns. Stock and cryptocurrency arrays can include jumps and small variations. Seismic arrays can have different bursts and scales. These examples make the task visible: the model is not reading domain names; it is reading structure inside numeric arrays.

## Slide 9 — Runtime labels

This slide explains how labels are produced. For each array, the pipeline applies structural transforms, measures candidate algorithm runtimes, and assigns the fastest algorithm as the label. So the label is empirical: it comes from measurement, not from a theoretical rule.

The transforms such as RAW, REV, SHUF, QBIN50, and PSORT10 make the decision boundary non-trivial. They force the dataset to include different types of structure. Then feature extraction creates the 16-feature representation used by the model. This is how the supervised learning problem is formed.

## Slide 10 — Training Dataset and Split

This slide shows the dataset filtering path. The starting corpus has 1,188,265 arrays. After near-tie filtering, it becomes 1,082,547. Then the majority class is capped, producing 196,624 rows for the training experiment. From there, the split is 137,636 for training and 29,494 each for validation and testing.

The lower boxes show why these controls exist. Near-tie filtering removes ambiguous labels, where two algorithms are too close to call. The majority cap reduces timsort dominance. Inverse-frequency weights make minority classes matter during the loss. Seed 42 keeps the split reproducible.

## Slide 11 — Train, validation, and test roles

Train, validation, and test have separate roles. The train set is where the model learns. The validation set is where I check tuning and behavior during development. The test set is held for final reporting.

All three use the same feature extractor and the same target definition: the measured fastest algorithm. The split is stratified 70, 15, 15, so class distribution is controlled across the three sets. This is important because without this separation, the reported result could be too optimistic.

## Slide 12 — Sixteen structural features

This slide summarizes the feature groups. The feature extractor builds sixteen O(n) structural features. They describe size, ordering, repetition, distribution, and robust scale.

Examples include length normalization, adjusted sorted ratio, run ratio, inversion ratio, duplicate ratio, entropy ratio, skewness, kurtosis, outlier ratio, IQR norm, and MAD norm. The key point is that all features are extracted before sorting. Timing features are excluded at prediction time. So the model is allowed to inspect the array, but not to benchmark candidate algorithms.

## Slide 13 — Feature importance

The feature importance result shows which signals the model uses most. Length norm is strongest. Top frequency, longest run ratio, duplicate ratio, and entropy ratio also matter.

This makes sense for sorting. Size affects constants and cache behavior. Repetition affects comparison patterns and duplicates. Runs and ordering affect how useful run-aware behavior can be. Entropy and distribution features describe how concentrated or scattered the values are. So the model is not learning arbitrary noise; the strongest features match structural sorting behavior.

## Slide 14 — XGBoost model

XGBoost is used as a decision-tree based gradient boosting model. In this thesis, the final selector is a multiclass classifier. The objective is `multi:softprob`, which means the model outputs class probabilities for introsort, heapsort, and timsort. The loss is multiclass log loss.

This model fits the problem because the input is tabular structural features, not images or text. Tree ensembles are strong for tabular data and can capture interactions between features, such as size combined with runs or duplicate ratio combined with ordering.

## Slide 15 — Sequential correction

This slide explains how boosting works at a high level. XGBoost builds trees sequentially. The first tree does not solve the whole problem. It makes an initial correction. The next tree focuses on what is still wrong. Then more trees are added, each correcting the current model.

So the final prediction is not one tree. It is an ensemble. Each tree contributes a correction to the class scores. After many trees, the model has learned a stronger decision rule from many smaller rules.

## Slide 16 — Class probabilities

Here the model output is shown as probabilities. The input feature vector goes into the model. The model produces one score for each sorting choice. Then the selector chooses the algorithm with the highest probability.

This is useful because the model does not only output a hard label internally. It has a confidence distribution over the three choices. For this thesis, the deployment decision is the maximum-probability class, but the probabilities also help us understand confidence and mistakes.

## Slide 17 — First and second order information

XGBoost corrects class scores using first and second order information. The first-order term is the gradient. It tells the model the direction of correction: should a class score go up or down. The second-order term is the Hessian. It tells how strong or careful that correction should be based on curvature.

For this model, that happens under multiclass log loss. XGBoost compares current class probabilities with the true fastest label, computes gradient and Hessian information, and uses them to choose splits and leaf values. That is why it is stronger than a simple error-counting method.

## Slide 18 — Ensemble constraints

The ensemble is constrained so it does not overfit timing noise. The model uses learning rate, max depth, row sampling, feature sampling, and L1/L2 regularization.

These controls matter because runtime labels are measured from real executions and can contain noise. A very flexible model could memorize unstable details. The constraints force the ensemble to learn broader structural patterns. In the final v5 model, the important settings include 500 trees, depth 7, learning rate 0.05, subsample 0.8, colsample 0.8, and regularization.

## Slide 19 — Test examples

This slide gives examples of successful and failed predictions. The successful examples show that the model can select the true fastest algorithm when the runtime measurements support it. The failed case shows that when the model chooses the wrong algorithm, we can still inspect the true runtimes and regret.

This is important because evaluation is not only label accuracy. For algorithm selection, a wrong label can be cheap or expensive depending on runtime distance. So success and failure are both evaluated by measured runtime, not only by whether the class name matched.

## Slide 20 — Experimental result summary

This slide summarizes the main reported numbers. The final v5 test accuracy is 76.1 percent. Balanced accuracy is 70.1 percent. Weighted F1 is 77.0 percent. The runtime result is stronger: 93.1 percent of the SBS-to-VBS gap is closed, and 89.6 percent of predictions have zero regret.

The important interpretation is that classification accuracy and runtime value are related but not identical. The model does not need perfect classification to produce strong runtime behavior, because some wrong choices are close in runtime to the true fastest choice.

## Slide 21 — Final v5 benchmark table

This table gives the final v5 metrics across train, validation, and test. The important pattern is that validation and test are close. Test accuracy is 76.1 percent and weighted F1 is 77.0 percent. The training score is higher, which is normal, but it is not the number I use as the final claim.

Balanced accuracy is lower than accuracy because the classes are not equally easy. Timsort is easier, while introsort and heapsort create a harder boundary. This table is the standard machine learning view of the final model.

## Slide 22 — Per-class metrics

This table explains the class behavior. Timsort has very high precision, recall, and F1. That means when the model predicts timsort, it is usually right, and it also finds most true timsort cases.

Heapsort is medium. Introsort is the hardest class. Its precision and F1 are lower because many boundary cases between introsort and heapsort are structurally close. This is why I do not rely only on accuracy. Precision, recall, and F1 show where the selector is strong and where it still struggles.

## Slide 23 — SBS, VBS, regret

This slide defines the runtime evaluation. SBS means single best solver: the best fixed algorithm over the evaluation set. VBS means virtual best solver: an oracle that always chooses the fastest algorithm for each array. The model is compared between these two references.

Regret measures how far the model runtime is from VBS. Gap closed measures how much of the SBS-to-VBS improvement the model captures. This evaluation is necessary because the thesis is not only about predicting a label; it is about reducing sorting runtime.

## Slide 24 — Runtime gap closed

This is the main runtime result. VBS is the oracle lower bound at 17.195 seconds. The v5 model is 17.475 seconds. SBS is 21.267 seconds. So the model is much closer to the oracle than to the single fixed algorithm.

The model closes 93.1 percent of the SBS-to-VBS gap. This is the main practical value of the work. Even though the classification accuracy is 76.1 percent, the runtime result is strong because many remaining mistakes have small regret.

## Slide 25 — Residual error pattern

This slide shows the confusion behavior. Most remaining error is concentrated at the introsort-heapsort boundary. Timsort is predicted very well, with 94.5 percent recall on the test set.

This tells us the errors are not random. The selector has a specific weakness: distinguishing introsort and heapsort when the structural features look similar. That weakness is understandable, because both are comparison sort algorithms and can be close in measured runtime for some arrays.

## Slide 26 — Clean confusion matrix

This confusion matrix shows counts and percentages. The diagonal cells are correct predictions. The off-diagonal cells show where the model sends one true class into another predicted class.

The key message is the same: timsort mostly stays on the diagonal, while introsort and heapsort exchange many cases. This gives the committee a direct view of the model’s error structure. It supports the claim that the model is useful, but not uniformly strong for every class.

## Slide 27 — F1 KFold benchmark table

This table answers the KFold question for the F1-specific experiment. It reports mean, standard deviation, minimum, and maximum across five folds. Accuracy, balanced accuracy, macro F1, and weighted F1 are all around 61 to 62 percent, with standard deviation around 0.6. Gap closed is about 59.8 percent, and zero regret is about 66.9 percent.

The important point is stability. The folds are close, so this F1-only result is not one lucky split. But this is separate from the main all-domain v5 selector.

## Slide 28 — KFold process

This slide explains how the KFold check was run. The F1-only sample is split into five folds. Each run trains on four folds and tests on the remaining fold. Then the process repeats until every fold has been used as the test fold.

This is not the same as domain holdout. KFold checks stability across row-level splits inside the F1 experiment. Domain holdout checks whether structure transfers when an entire domain is hidden from training. I keep these two evaluations separate.

## Slide 29 — Strict checks

This slide defines the boundary of the reported result. The v5 production result is compared with the v6 source-aware check. V6 uses non-overlapping sources and is stricter. Its accuracy and weighted F1 are lower.

This is not a failure of the thesis; it is a rigor check. It shows that when the evaluation becomes stricter, some performance drops. I report that clearly, because the contribution is not only a high number. It is also an evaluation discipline that shows where the model is strong and where it should not be overstated.

## Slide 30 — Model evolution

This slide shows the model journey. Earlier versions were not the final claim. v1 regression failed. v2 was based on synthetic data. v3 had leakage through log-pairwise timing. v5 is the production selector used as the main result. v6 is source-aware and stricter. v7 and v8 were rejected directions.

The point of this slide is that the final method was not chosen immediately. The work went through failed and stricter checks, and the final v5 story is the one that remained defensible.

## Slide 31 — Domain holdout

Leave-one-domain-out testing asks a different question: if one full domain is hidden during training, can the model still use transferable structural features on that hidden domain?

The gap closed result stays high in several held-out domains, but not equally for all. Weather has lower accuracy but still high gap closed. F1 has a weaker gap result than other domains. So this supports cross-domain robustness, but it does not claim perfect domain generalization. It shows useful transfer with visible limits.

## Slide 32 — F1 routing specialization

This slide separates the F1-specific route from the main v5 result. The main track is the general selector across all domains and three algorithms. The F1 specialization uses channel flag routing, where F1 arrays can be routed to channel-specific models.

This does not replace v5. It is a separate specialization. I include it because the thesis explored it, but I do not mix its numbers with the main v5 claim. The main contribution remains the general selector and runtime evaluation.

## Slide 33 — Optuna tuning

This slide shows the Optuna tuning result for the F1 channel models. Optuna was used to search hyperparameters for each channel model. The table compares selected before and after values and shows improvements in several F1 channel metrics.

The important interpretation is narrow: Optuna strengthened the F1 specialization, not the main v5 all-domain result. It shows that the channel models benefited from hyperparameter search, but the F1 track remains separate from the main contribution.

## Slide 34 — Where selector can be used

This slide shows possible use cases. The selector can be useful wherever large numeric arrays need to be sorted repeatedly and where input structure changes across cases.

Examples include database query engines, ETL and analytics pipelines, ranking systems, financial time-series processing, and scientific or telemetry processing. I am not claiming the model is deployed in these systems. The point is that the algorithm-selection idea is relevant wherever sorting is repeated and input structure varies.

## Slide 35 — Contributions

The first contribution is a deployable selector: a compact pipeline that combines array features with a gradient-boosted classifier. The second contribution is the evaluation framework: SBS, VBS, regret, gap closed, source-aware checks, and KFold checks. The third contribution is reproducibility: open scripts, saved artifacts, exported models, and a clear feature extractor.

Together, these contributions show not just that a model can predict a sorting label, but that the prediction can be evaluated in terms of practical runtime value.

## Slide 36 — Limitations and future work

The remaining errors show where the feature representation stops seeing enough information. Introsort and heapsort can be close when the current features do not separate their runtime behavior. Timing also depends on hardware and software environment. Domain transfer is useful but not perfect.

Future work should expand the algorithm portfolio, test more domains, improve feature extraction, evaluate deployment overhead, and validate stricter online or domain-specific settings. LinUCB and online adaptation should remain future work unless fully validated.

## Slide 37 — Thank you

Thank you for listening. I am ready for your questions.

If asked for the core result, I would summarize it like this: using only features extracted before sorting, the final v5 selector reaches 76.1 percent test accuracy and closes 93.1 percent of the runtime gap between the single best fixed algorithm and the oracle selector. The strongest result is runtime value, with limitations shown clearly through per-class metrics, KFold, and stricter checks.

## Expansion Layer For 22-24 Minutes

Use these lines during rehearsal if the talk is too short. They are not separate slides. Add them naturally after the matching slide.

Slide 1 add: I will focus on the problem, the data, the model, and the experimental results. I will not present literature as a separate block. Instead, I connect prior ideas directly to the task: algorithm selection, empirical runtime modeling, and gradient-boosted tabular classification.

Slide 2 add: The motivation is not that sorting algorithms are weak. The motivation is that input structure changes, and the best algorithm can change with it. At scale, even a small runtime difference becomes important because sorting is repeated many times inside larger systems.

Slide 3 add: The selector is trained offline, but the decision must be made before sorting for a new array. That distinction is central. Training can use measured runtimes to create labels, but prediction cannot use measured runtime for the same array.

Slide 4 add: This prevents a wrong interpretation of the thesis. I am not running all candidate algorithms first and then choosing the fastest. That would be an oracle after the fact. The challenge is choosing from cheap structural information before execution.

Slide 5 add: The algorithm portfolio is intentionally limited in the main experiment. With three algorithms, the behavior is easier to evaluate and explain. A larger portfolio is future work, but first the thesis asks whether selection is possible under a controlled setting.

Slide 6 add: The system is reproducible because the same feature extractor is used across data generation, training, and inference. This avoids a mismatch where training and deployment see different representations of the same array.

Slide 7 add: Real domains matter because synthetic data can be too clean. Real numeric traces contain repeated values, ordered regions, noisy jumps, and local structure. These patterns are exactly what can make one sorting algorithm faster than another.

Slide 8 add: The examples are not shown because the model understands weather or finance. They are shown because the raw arrays make the structure visible. The model receives numeric structure, not semantic domain labels.

Slide 9 add: The label is empirical. For every array, candidate runtimes are measured and the fastest algorithm becomes the class. This makes the task supervised learning, but the supervision comes from measured algorithm behavior.

Slide 10 add: Near-tie filtering is important because if two algorithms are extremely close, the label can become unstable. Removing ambiguous cases gives the classifier a cleaner signal and reduces noise in the target.

Slide 11 add: The final claim comes from the test set. Training score shows the model can fit. Validation checks development choices. Test gives the reported performance. This is why I keep those roles separate.

Slide 12 add: O(n) feature extraction matters because the selector itself must not become more expensive than the sorting decision it is trying to improve. The features must be cheap enough to justify using the selector.

Slide 13 add: Feature importance is not a perfect explanation, but it is a sanity check. The strongest features match sorting intuition: size, frequency, runs, duplicates, and distribution shape.

Slide 14 add: XGBoost is a good fit because the input is tabular. There are sixteen structural features, not raw images or language. Tree ensembles are strong for this kind of data and can model feature interactions.

Slide 15 add: Boosting means each tree is added after the previous ones. The new tree focuses on remaining correction. So the final prediction is accumulated from many trees, not from one standalone decision tree.

Slide 16 add: The model outputs class probabilities. If timsort has very high probability, the decision is confident. If introsort and heapsort are close, the case is near the boundary, which matches the confusion behavior later.

Slide 17 add: This is the technical part of XGBoost. The gradient tells the correction direction. The Hessian tells how strong or careful the correction should be. This second-order information is one reason XGBoost is more powerful than simple boosting.

Slide 18 add: Regularization matters because runtime labels can contain measurement noise. The constraints stop the model from memorizing small timing fluctuations and encourage broader structural rules.

Slide 19 add: The failed case is useful, not embarrassing. It shows how the model can be wrong as a class prediction, and then we can measure how expensive that wrong choice is in runtime.

Slide 20 add: This is why I report both machine learning metrics and runtime metrics. Accuracy and F1 explain prediction behavior. Gap closed and regret explain whether the prediction creates practical runtime value.

Slide 21 add: The validation and test values are close. That is important because if validation was much higher than test, the model would look tuned to validation only. Here, the reported test behavior is consistent with validation.

Slide 22 add: Precision means how trustworthy a predicted class is. Recall means how many true cases of that class the model finds. F1 combines them. For this task, these metrics show that timsort is easy and introsort is hard.

Slide 23 add: SBS is a realistic baseline because a system can choose one fixed algorithm. VBS is not deployable, but it defines the best possible per-array choice. The selector is useful if it moves close to VBS.

Slide 24 add: The key insight is that not every wrong class has the same cost. If the chosen algorithm is second-best and very close to the fastest, classification is wrong but runtime regret is small.

Slide 25 add: The confusion matrix shows the model weakness clearly. Timsort has a strong structural signature, but introsort and heapsort can be close when runs and ordering features do not separate them enough.

Slide 26 add: I use this slide to avoid overstating the model. The model is useful, but not uniformly strong across all classes. The error pattern is visible and specific.

Slide 27 add: This KFold result is for the F1-specific experiment. It should not be mixed with the main all-domain v5 claim. It answers the advisor’s cross-validation question for the F1 route.

Slide 28 add: KFold and domain holdout test different risks. KFold tests row-split stability. Domain holdout tests whether a full unseen source domain can still be handled from transferable structure.

Slide 29 add: The strict check is included because I want the reported result to have a clear boundary. The thesis does not hide that stricter evaluation lowers some numbers.

Slide 30 add: The model evolution slide shows why the final result is v5. Earlier versions gave useful lessons, but some had leakage or weaker assumptions. v5 became the main defensible selector.

Slide 31 add: Domain holdout does not prove perfect generalization. It shows useful transfer with limits. That is why I present both accuracy and gap closed by held-out domain.

Slide 32 add: The F1 routing route is a separate specialization. The main story remains the general selector over all domains and three sorting algorithms.

Slide 33 add: Optuna was used to tune the F1 channel models. It improved channel-specific behavior, but it is not the main v5 result. This keeps the claims separated.

Slide 34 add: These are possible use cases, not deployment claims. Any real system would need its own validation under its hardware, data, and pipeline constraints.

Slide 35 add: The contribution is not just one model. It is the selector pipeline, the runtime-aware evaluation, and the reproducible artifact set.

Slide 36 add: The biggest limitation is that the selector depends on the feature representation and the runtime environment. New hardware, new algorithms, or new domains can change the decision boundary.

Slide 37 add: For questions, I will keep the answer measured: v5 is the main general selector, F1 routing is separate, KFold is F1-only, and LinUCB or online adaptation is future work.
