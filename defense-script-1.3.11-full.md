# Defense Script 1.3.11 - Full Speaking Version

## Slide 1

Good morning everyone. My name is Ahmed Salah Abd Alhi Mohammed, and today I will present my MSc thesis, Machine Learning Based Algorithm Selection for Sorting Numeric Arrays.

The work is about a practical question in sorting. When we have a numeric array, we normally choose a sorting algorithm before we see how that specific input behaves. But different sorting algorithms do not react to input structure in the same way. Some arrays are already close to sorted. Some have repeated values. Some have local runs. Some are noisy and irregular. So the fastest algorithm is not always the same for every input.

The question I worked on is this: can we look at the array before sorting, extract cheap structural information, and use that information to choose the best sorting algorithm from a small portfolio?

So this thesis is not about inventing a new sorting algorithm. It is about algorithm selection. The goal is to build a practical selector that chooses among existing algorithms before execution. This distinction is important for the whole presentation. I am not claiming that introsort, heapsort, or timsort are replaced. I am claiming that the input structure can guide which one should be used for a specific array.

## Slide 2

The research objective is shown here as a mapping: `S : f(x) -> A`.

The input array is `x`. From this array, I compute a feature vector `f(x)`. This feature vector describes the structure of the array. Then the selector `S` maps that feature vector to an algorithm choice `A`.

The objective is not only to get high classification accuracy. In this problem, a wrong class label does not always mean a bad runtime decision. If introsort and heapsort are very close on one array, choosing the second fastest one is technically wrong, but it may cost almost nothing. Because of that, the real objective is to reduce regret with respect to the oracle choice.

The slide shows four parts of the objective. First, the features must be compact and cheap to extract. Second, the model is trained offline as an XGBoost model for per-instance algorithm selection. Third, the evaluation must be leakage-resistant, because it is easy to overestimate performance if related sources appear in both training and test. Fourth, the whole method should be reproducible as a Python library, not only as one isolated experiment.

This is the frame for the whole defense: extract structure, train a selector, evaluate with runtime-aware metrics, and keep the pipeline reproducible. I will keep returning to this same chain because it is the logic of the thesis. If one part is weak, the result becomes hard to defend. The features must be cheap, the model must predict before sorting, and the evaluation must measure runtime value, not only label matching.

## Slide 3

This slide defines what is allowed and what is not allowed at prediction time.

On the left side, the selector is allowed to extract `O(n)` features before sorting. It can measure things like order, runs, repetition, distribution, and robust scale. These are properties of the input array itself. They are available before choosing the sorting algorithm.

On the right side, the selector is not allowed to sort with introsort, heapsort, and timsort, and then pick after seeing all runtimes. That would solve the problem only by already doing the expensive evaluation. It would not be a practical selector.

So the important boundary is this: prediction happens before any candidate sorting algorithm is executed. The oracle is used only offline for labeling and evaluation. It is not available at run time.

This boundary matters because without it, the task becomes misleading. If the model uses timing information from candidate algorithms, then it is no longer predicting the best algorithm before sorting. It is just reading the answer after doing the work. My task is stricter: choose using only structural information from the array. This is also why the thesis separates deployable features from timing-derived features. Timing can be used to create the label offline, but it cannot be used as an input to the final selector.

## Slide 4

This slide shows the algorithm selection problem in formal terms.

On the left, there is the problem space. In this thesis, one problem instance is one numeric array. That array has structure: maybe it has ordered regions, duplicates, drops, long tails, or local irregularities.

The next box is the feature space. The array is converted into structural features. These features include size, sortedness and disorder, runs, duplicates and entropy, and spread or outlier information. The feature vector is the representation that the model can use.

Then the selector maps from feature space into algorithm space. Here the algorithm portfolio contains three algorithms: introsort, heapsort, and timsort.

At the bottom is the performance space. This is where the true label is created. For each array, every candidate algorithm is timed, and the fastest measured algorithm becomes the label. But this measurement is only for training and evaluation. It is not part of prediction.

So the formal task is: given an array, compute its structural features, predict the best algorithm, and later compare that prediction with the measured oracle. The selector is judged by how close it comes to the oracle while still obeying the prediction-before-execution rule. That is the reason this is an algorithm-selection problem and not simply a benchmarking table.

## Slide 5

This slide shows the system architecture.

The flow starts with an input numeric array. The first processing step is feature extraction. This is where the array is converted into sixteen structural features. Those features go into the XGBoost classifier. The classifier predicts the sorting algorithm to use.

On the right side, the first card says the main track uses all domains and three algorithms. That is the main thesis result. The second card says the prediction path is features into XGBoost classifier. That is the deployable path. The third card says the F1-specific routing stays separate from the main result.

This separation is important. The general selector is trained across domains and chooses among the three sorting algorithms. The F1 route is a later specialization experiment for a difficult domain. I do not mix those two claims.

The architecture also separates offline training from prediction. During training, I can time all candidate algorithms to build labels. But during prediction, the system only extracts features and runs one XGBoost prediction. That is what makes it a practical selector rather than an offline oracle. The architecture also makes the two model tracks clear. The general v5 selector is the main story. The F1 channel route is useful, but it stays separate because it answers a more specialized question inside one difficult domain.

## Slide 6

This slide shows the scale and source of the data.

The corpus contains 1,188,265 numeric arrays. These arrays were extracted from five structurally different real-world domains: Formula 1 telemetry, equity prices, cryptocurrency OHLCV, weather, and seismic measurements.

The reason for using real data is that sorting behavior depends on input structure. Synthetic random arrays are useful for controlled experiments, but they do not always create the structures that appear in real numeric sequences. Real sequences can have trends, local runs, repeated values, plateaus, spikes, noise, and domain-specific patterns.

For algorithm selection, these structures matter. A nearly sorted array and a noisy array may not favor the same sorting algorithm. An array with many repeated values can behave differently from an array with mostly unique values. A trace with long ordered regions can create a different workload from a shuffled trace.

So the dataset was built to expose the selector to diverse structural regimes, not only to one clean synthetic distribution. This matters because the model can only learn a useful selector if the training data actually contains different kinds of sorting behavior. If the dataset is too simple, the selector may look good but will not say much about real numeric arrays.

## Slide 7

This slide shows leave-one-domain-out testing.

In this evaluation, one domain is held out as the test domain. The model trains on the other four domains and then tests on the unseen domain. So, for example, if weather is held out, the model learns from the other domains and then must generalize to weather.

The chart reports accuracy and gap closed for each held-out domain. The results are not the same for every domain. Some domains transfer better because their structural signals overlap more with the training domains. Other domains are harder because their traces have different structure.

This is important because a normal random split can make the model look stronger if very similar sources appear in both train and test. Leave-one-domain-out testing asks a harder question: did the model learn structural signals that can transfer, or did it only learn source-specific patterns?

The result is mixed but useful. The selector does not transfer equally everywhere, but it still captures meaningful runtime value when the structural patterns are visible. This also defines a limitation: the model is strongest when the unseen domain has structure that the feature space can recognize. I do not interpret cross-domain robustness as perfect generalization. I interpret it as evidence that some structural signals transfer, while some domain-specific behavior still remains hard.

## Slide 8

This slide shows real sampled traces from the five domains.

The purpose here is to make the data concrete. These are not just names of datasets. Each domain produces a different kind of numeric trace. Stock market data can show movement and trend. Cryptocurrency can be more volatile. Weather traces can be smoother. Formula 1 telemetry has channel and lap-related behavior. Seismic traces can contain noisy bursts and local changes.

For sorting, these differences matter because the algorithm does not only see a list of numbers. It sees a structure. If the array has ordered regions, timsort can benefit from them. If the values are irregular or the boundary is more subtle, the advantage may be less visible.

This is why the thesis focuses on structural features. The model is not looking at the domain name as the main explanation. It is trying to learn which structural patterns in the numeric sequence indicate which sorting algorithm is likely to be fastest.

So this slide connects the dataset to the main idea: real arrays create real structural differences, and those differences are what the selector tries to use. When I later talk about feature importance and confusion behavior, this is the background. The model succeeds when these visible structures are informative, and it struggles when the important difference is not visible in the feature vector.

## Slide 9

This slide shows the structural feature extraction stage.

Each array is converted into sixteen `O(n)` structural features. The features must be cheap because they are computed before sorting. If feature extraction is too expensive, it can destroy the benefit of selecting a faster sorting algorithm.

The feature groups include size information, ordering information, repetition information, distribution information, and robust scale information. Examples on the slide include length normalization, size effects, cache or constant-size effects, adjusted sorted ratio, runs ratio, and inversion ratio.

The key rule is that timing features are excluded at prediction time. The model cannot use candidate algorithm runtimes as input, because those runtimes would only exist after executing the algorithms. That would be leakage.

So the feature extractor is a single canonical representation used across data generation, training, validation, and inference. This is important for reproducibility. The model should see the same definition of features everywhere, not one version during experiments and another version during deployment. It also keeps the evaluation honest: if a feature is not available before sorting, it does not belong in the final prediction path.

## Slide 10

This slide shows feature importance.

The main message is that the model relies heavily on structural sorting signals. Length-related features, repetition-related features, and ordering-related features appear among the most important features.

This matches the intuition of the problem. Sorting algorithms are affected by input order, repeated values, and array size. Timsort, for example, is designed to exploit natural runs. So if an array has visible run structure or ordered regions, that gives the model a strong signal that timsort may be a good choice.

But the feature importance also helps explain where the model is limited. Some differences between introsort and heapsort depend on lower-level behavior, such as memory access, cache effects, branching, or implementation details. Those effects may not be fully visible in the sixteen structural features.

So I interpret this slide in two ways. First, the model is not random; it is using meaningful structural information. Second, the remaining errors are partly explained by the limits of what the features can see. This is why I do not claim that the model understands every low-level runtime effect. It learns from structural signals, and those signals are strong for some algorithm boundaries and weaker for others.

## Slide 11

This slide explains how labels are produced.

The labels are empirical fastest-algorithm measurements. For each array, each candidate algorithm is timed on a fresh copy. The algorithm with the minimum observed runtime becomes the label.

This is important because the label is not assigned by theoretical complexity. All three algorithms are comparison-based and have similar asymptotic behavior in many cases, but their practical runtimes can differ because of input structure and implementation behavior.

The bottom of the slide shows the transformations: raw, reversed, shuffled, quantized, and partially sorted. These transformations create different structural regimes. They make the decision boundary more meaningful because the selector is not trained only on one kind of sequence.

So this slide is the bridge between data and supervised learning. The arrays come from real domains and transforms. The algorithms are timed. The fastest measured algorithm becomes the target label. This also explains why the label is empirical. The model is learning from measured behavior, not from an assumption that one algorithm should always dominate.

## Slide 12

This slide shows the training dataset and split.

The process starts with 1,188,265 collected arrays. After near-tie filtering, the dataset becomes 1,082,547. After the majority cap, the training set used for the final split is 196,624. Then it is split into 137,636 training examples and 29,494 validation plus 29,494 test examples.

Near-tie filtering removes ambiguous fastest labels. This matters because if two algorithms are nearly equal, forcing one label can add noise. A label may be technically correct but practically meaningless if the runtime difference is tiny.

The majority cap is used because timsort dominates many raw labels. Without controlling this, the model could learn to overpredict timsort and ignore smaller classes. Inverse-frequency weights then make minority classes matter during loss calculation.

The fixed seed makes the split reproducible. So this slide is not just about dataset size. It shows how label noise and class imbalance were controlled before model training. Without these controls, the model could appear accurate while mainly following the dominant timsort class. The filtering and weighting make the training signal more useful for learning actual decision boundaries.

## Slide 13

This slide starts the model explanation.

The left card is the input: `f(x)` in `R^16`. That means each array is represented by sixteen structural features. The middle card is the class probability model. The right card is the output: three probability scores, one for introsort, one for heapsort, and one for timsort.

This is why the model is a multi-class classifier. It is not a binary classifier, because there are three candidate algorithms. It is also not a runtime regression model in the final result, because the final prediction needed is the algorithm class.

The objective shown on the slide is `multi:softprob`, and the metric is `mlogloss`. `multi:softprob` makes XGBoost output class probabilities. `mlogloss` is the multi-class log-loss used to evaluate and guide probability quality during training.

At prediction time, the model selects the algorithm with the highest probability. So the model learns a probability distribution over sorting choices from structural features. This is useful because the model can express uncertainty. For a clear timsort case, timsort can receive a high probability. For a boundary case, the probabilities can be closer, which matches the idea that some arrays are genuinely harder to separate.

## Slide 14

This slide shows how the training signal is balanced before the trees learn.

The first chart represents the raw fastest-label signal. It is not naturally balanced. Timsort wins many cases, while introsort and heapsort are smaller classes. If this is used directly without control, the model can become biased toward the dominant class.

The middle card shows the training controls. Near-tie filtering removes ambiguous labels. The majority cap limits class dominance. Inverse-frequency sample weights make minority-class examples contribute more during loss optimization.

The final chart shows the balanced loss pressure. This does not mean the real world is artificially balanced. It means the training process is prevented from ignoring smaller but important decision regions.

This matters because algorithm selection is not useful if the model simply predicts the majority winner. The model must learn the structural cases where another algorithm is actually faster. That is why this slide comes before the tree and gradient explanation. Before asking how XGBoost learns, we first need to make sure the signal it receives is not dominated by one class.

## Slide 15

This slide explains the gradient and Hessian part of XGBoost training.

For each array, the model has class scores. These scores correspond to introsort, heapsort, and timsort. Softmax converts the scores into probabilities. Then the model compares those probabilities with the true fastest label.

The left formula represents the gradient. This is first-order information. It tells XGBoost the direction of correction. In simple terms, for a class score, should the model push it up or push it down?

The right formula represents the Hessian. This is second-order information. It describes the curvature or strength of the correction. In simple terms, how careful should the model be with that update?

Because the objective is `multi:softprob` with `mlogloss`, these gradient and Hessian terms come from multi-class log loss. They are used when XGBoost chooses splits and when it computes leaf corrections.

So XGBoost is not only fitting mistakes in a simple way. Each tree is built using both direction and curvature, which is one reason it works well for structured tabular problems like this one. In this thesis, that means each new tree is trying to correct the class scores for sorting choices, not changing the array itself. The corrections happen inside the model scores.

## Slide 16

This slide shows the constraints on the ensemble.

The model uses 500 trees, but those trees are not allowed to grow without control. The depth is limited to 7, the learning rate is 0.05, the histogram method is used, and L1/L2 regularisation is applied.

The row sample is 80 percent, meaning each tree trains on only part of the rows. The feature sample is also 80 percent, meaning each tree sees only part of the features. This adds controlled randomness and reduces dependence on one exact sample or one exact feature set.

The learning rate controls how much each tree can correct the model. A smaller learning rate makes the ensemble learn more gradually. The regularisation terms reduce the chance that leaves become too specific to noisy timing behavior.

So the model is powerful, but constrained. This is important because the labels come from runtime measurements, and runtime measurements can contain noise. The goal is to learn stable structural boundaries, not memorize timing artifacts. This is why the final model uses both model controls and data controls. The data controls reduce ambiguous or imbalanced signal, and the ensemble controls reduce overfitting inside XGBoost.

## Slide 17

This slide shows a strict check on the reported result.

The chart compares the v5 production result with the v6 source-aware check. It compares accuracy, balanced accuracy, and weighted F1-score.

The source-aware check is harder because it reduces the chance that similar sources appear in both training and test. If the model performs well only because source-related examples are repeated across splits, then the result is less credible.

Here, the stricter source-aware result is lower. I do not hide that. The lower result shows that source separation matters. It also helps define the boundary of the claim.

So this slide is not a failure slide. It is a credibility slide. It shows that the thesis tested the model under a stricter condition and used that result to avoid overclaiming.

The main result remains the v5 production selector, but the v6 source-aware check tells us how careful we should be when interpreting generalization. This is also why I present the result as measured strong, not absolute. The model is useful, but the stricter check shows the boundary of that usefulness.

## Slide 18

This slide summarizes the model evolution.

The chart starts with v1 regression. That early direction was useful, but regression was not the best final formulation because predicting exact runtime can learn scale more than the winner boundary.

Then v2 is classification, which is closer to the real decision: choose the fastest algorithm class. v3 includes timing leakage, and it reaches 100 percent because timing-related information makes the task too easy. But that is not deployable, because those timing signals are not available before prediction.

After that, v5 becomes the production model. Then v6 tests source-aware splitting. v7 tests regret weighting. v8 tests a binary cascade. These later variants did not replace v5, but they explain the boundary of the problem.

The message is that the final model was not chosen only because of one score. It was chosen because it was deployable, avoided timing leakage, and gave the best practical balance under the evaluation. The models after v5 are still important because they show that I tested alternatives instead of stopping at the first good result.

## Slide 19

This slide defines SBS, VBS, regret, and gap closed.

SBS is the single best solver. It means the best fixed algorithm over the evaluation set. If we had to choose only one algorithm for everything, SBS is the best such choice.

VBS is the virtual best solver. It is the oracle that chooses the fastest algorithm for each individual array. VBS is not available at prediction time, but it defines the upper bound.

The model sits between these two. It chooses before execution using structural features.

The regret formula measures the distance between the model runtime and the VBS runtime. The gap-closed formula measures how much of the improvement from SBS to VBS is captured by the model.

This is why the evaluation is runtime-aware. A classification error is not always equally bad. If the selected algorithm and the oracle algorithm have nearly equal runtime, then the regret is small even if the label is wrong. This is the central reason SBS, VBS, regret, and gap closed are used. They translate classification behavior into runtime meaning.

## Slide 20

This slide explains why accuracy alone is insufficient.

The chart compares accuracy and gap closed across model versions. The important message is that accuracy and runtime value do not always move together.

A model can have moderate accuracy but still close most of the runtime gap if its mistakes are cheap. At the same time, a model can improve accuracy in a way that does not fully translate into runtime benefit.

For this task, the cost of being wrong depends on the runtime distance from the oracle. If the chosen algorithm and the optimal algorithm are almost tied, then the mistake is not expensive. But if the chosen algorithm is much slower, then the mistake matters more.

So I use accuracy, but I do not stop there. I interpret the model through gap closed and regret because these metrics match the practical goal of algorithm selection. This slide prepares the interpretation of the next result slide. The question is not only whether the model predicts the exact oracle label, but whether it captures most of the available runtime improvement.

## Slide 21

This is the headline result.

The v5 selector reaches 76.1 percent top-1 accuracy. If we looked only at that number, we might think the model is only moderately successful.

But the runtime-aware metrics give the more important interpretation. The selector closes 93.1 percent of the SBS-to-VBS gap, and 89.6 percent of predictions have zero regret.

That means most of the practical runtime opportunity is recovered, even though not every oracle label is predicted exactly.

This is the central argument of the thesis. The model is not a perfect oracle. It does not solve every class boundary. But it is strong at avoiding expensive runtime mistakes. That is why the result should be judged by both classification accuracy and runtime regret. If I only reported 76.1 percent accuracy, the result would look incomplete. If I only reported gap closed, the classification boundary would be hidden. Reporting both gives a more honest picture.

## Slide 22

This slide explains the class behavior.

The confusion matrix shows where the model predicts correctly and where it makes mistakes. The strongest class behavior is timsort. The slide shows 94.5 percent timsort recall.

This makes sense because timsort-friendly structure is often visible in the input. Runs, ordering, and repeated structure give the feature vector a strong signal.

The harder part is the introsort-heapsort boundary. These algorithms can remain close in the current feature space. Some of their differences depend on effects that are not fully captured by the structural features.

So the residual error is not just random failure. It tells us where the current features stop seeing. The slide also notes that most residual boundary errors carry low measured regret, which connects back to why the runtime result is stronger than accuracy alone. So the confusion matrix is not only a mistake table. It is evidence about which classes are structurally separable and which boundaries remain hard.

## Slide 23

This slide explains the F1 routing experiment.

The main track is shown on the right: the v5 general selector uses all domains and three algorithms. That is the main result of the thesis.

The F1 specialization is separate. For Formula 1 telemetry, the route uses channel or flag information to send arrays to specialized channel models. The diagram shows this routing step before the final predicted algorithm.

This separation is important because F1 telemetry is structurally difficult. Different channels can behave differently, so one global decision boundary may not capture all cases well.

But I do not present this routed model as replacing v5. It stays separate from the main cross-domain result. The correct interpretation is that routing is a promising specialization direction for difficult domains, not the main thesis claim. This is important because otherwise the committee may think there are two competing main models. There are not. The main story is v5; the F1 route is a specialized extension.

## Slide 24

This slide summarizes the contributions.

The first contribution is a deployable selector. The pipeline combines sixteen `O(n)` structural features with a gradient-boosted classifier to choose among introsort, heapsort, and timsort.

The second contribution is the evaluation framework. The thesis uses source-level separation, near-tie filtering, leave-one-domain-out testing, and regret metrics. This gives a stronger view than accuracy alone.

The third contribution is the reproducible artifact. The work includes one source-of-truth feature extractor, the data pipeline, and exported XGBoost models in JSON and ONNX formats.

So the contribution is not only the final number. It is the full method: how the selector is built, how it is evaluated, and how the result can be reproduced. This is why I describe the work as both a practical selector and an evaluation discipline.

## Slide 25

This slide shows the limitations and future work.

The current portfolio is limited to three C-level comparison sorts. That means the result should not be generalized to all possible sorting algorithms or all hardware settings.

The absolute regret values are tied to one hardware and software environment. If the platform changes, the runtime differences may change as well.

LinUCB was designed but not validated as a final empirical result, so I do not claim it as a proven contribution. Also, feature extraction becomes less useful below about 1,000 elements because the overhead can dominate the sorting time.

The future work follows directly from these limits: expand the portfolio, validate online adaptation, use stronger domain embeddings or channel-level routing, and test cross-platform deployment with feature extraction overhead included.

The important point is that these limitations do not remove the main result. They define the boundary of the result. The selector works as a practical structural selector under the reported evaluation, and the next step is broader and more adaptive selection. So I am not claiming the problem is finished. I am claiming that this thesis shows a defensible path for using structural machine learning to make sorting decisions before execution.

## Slide 26

Thank you for listening.

I am happy to answer your questions.

If I get a difficult question, I should answer it directly. First, acknowledge the concern. Second, point to the evidence. Third, state the limitation clearly. Fourth, explain why the contribution still holds.

For example, if I am asked about source leakage, I should say that this is exactly why the source-aware check was included. It reduced the result, but it made the evaluation more credible.

If I am asked about accuracy, I should say that 76.1 percent is not the whole claim. The stronger result is 93.1 percent gap closed and 89.6 percent zero-regret predictions.

If I am asked about F1 routing, I should say that it is a separate specialization experiment and not the main v5 result.
