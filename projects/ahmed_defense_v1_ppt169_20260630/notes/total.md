# 01_title

Good morning. My thesis is about machine learning based algorithm selection for sorting numeric arrays. The main idea is simple: before we sort an array, can we look at its structure and choose the algorithm that is most likely to run fastest? I will focus on the general v5 selector and on runtime value, not only classification accuracy.

# 02_problem

Sorting is a familiar task, but choosing the sorting algorithm is still input-dependent. The same program may receive arrays with different sizes, repetitions, order, and distribution. My work treats this as a selection problem: given the array structure, choose introsort, heapsort, or timsort before execution.

# 03_complexity

The motivation starts from a practical gap. These algorithms can sit in the same broad complexity class, but their measured runtime is not the same on every input. Complexity tells us the high-level behavior, but it does not answer which algorithm is best for this specific array.

# 04_select_before

The selector works before sorting begins. I do not run all algorithms and then choose the fastest, because that would already spend the runtime. Instead, I extract cheap structural features, pass them to the model, and then execute one selected algorithm.

# 05_practical_selector

The aim is not to build a perfect oracle. The aim is a practical selector. A wrong class label is not always a serious runtime mistake, because two algorithms can be very close in time. That is why I evaluate accuracy together with regret and gap closed.

# 06_structural_signals

This slide shows the feature extraction idea. The raw array is converted into a compact description: size, ordering, repetition, and distribution. These are the signals that sorting algorithms can react to. I will not go through every formula here; the important point is that the selector sees structure, not the full raw sequence.

# 07_system_decision

The full system is a single decision pipeline. The array enters, the structural features are computed, XGBoost predicts the algorithm, and then only that algorithm runs. This keeps the method practical and separates the decision cost from the sorting runtime.

# 08_algorithm_portfolio

The portfolio has three algorithms: introsort, heapsort, and timsort. I kept the portfolio focused because the defense needs a clear measured comparison. They are also different enough: timsort benefits strongly from existing runs, while introsort and heapsort are closer and harder to separate.

# 09_real_arrays

The dataset is one of the main changes from the early experiments. The final v5 work uses about 1.18 million arrays from five domains: F1 telemetry, stock, crypto, earthquake, and weather. This matters because real arrays provide structures that synthetic data alone did not show.

# 10_labels_runtime

The labels were created from measured runtime. For each array, the candidate algorithms were benchmarked, and the fastest algorithm became the label after filtering noisy cases. So the model is not learning my preference. It is learning from measured winners.

# 11_feature_groups

The 16 features are grouped by what they describe. Some describe size, some repetition, some ordering, and some distribution. This makes the model easier to defend, because the features are connected to how sorting algorithms behave.

# 12_v5_selector

The main model in the thesis is the general v5 selector. It uses XGBoost, 16 structural features, and three labels: introsort, heapsort, and timsort. The F1 channel routing work is separate. I mention it as a specialization, but I do not mix it with the general v5 result.

# 13_failed_paths

The research journey is important, but I keep it compact. Regression was not the right framing. Timing-based features gave a ceiling but were not deployable because they used information that would not be available before sorting. Later strict checks and alternative models helped define the limits of the final claim.

# 14_main_result

The main result is that v5 closes 93.1 percent of the runtime gap between the single best solver and the virtual best solver. Its test accuracy is 76.1 percent, and 89.6 percent of selections have zero regret. So the classifier is not perfect, but most mistakes are cheap in runtime terms.

# 15_accuracy_regret

This is why I do not stop at accuracy. Regret measures how much slower the selected algorithm is compared with the best available algorithm for that array. Gap closed measures how much of the practical improvement over a fixed baseline is recovered. These metrics explain the real value of the selector.

# 16_confusion_boundary

The confusion matrix shows the main behavior. Timsort is structurally visible, with strong recall. The hard part is separating introsort and heapsort. That weakness is important, and I do not hide it. It tells us where the current features stop being enough.

# 17_feature_importance

The feature importance is consistent with the story. Length, frequency features, longest runs, and duplicate ratio are among the strongest signals. I do not claim feature importance proves causality, but it supports that the model is using meaningful structural information.

# 18_domain_transfer

The domain holdout results show a measured generalization picture. The selector does not transfer equally to every domain, but the runtime value survives in several leave-one-domain-out tests. My interpretation is that transfer works when the held-out domain shares relevant structure.

# 19_strict_checks

Strict checks changed the interpretation. The source-aware v6 split lowers the numbers, which is expected when we reduce leakage risk. v7 regret-aware training did not improve the result and was rejected. v8 showed that the intro-heap boundary is genuinely difficult. These results make the final claim narrower, but more honest.

# 20_limits

The main limitation is clear: the current features can identify timsort-friendly structure much better than they separate introsort from heapsort. The F1 channel routing work is also separate from the general selector. It shows a specialization path, but the main contribution remains the v5 three-algorithm selector.

# 21_contributions

The contribution has three parts. First, a dataset and labeling protocol based on measured runtime. Second, a structural machine learning selector for sorting algorithm choice. Third, an evaluation discipline that combines accuracy with regret, SBS, VBS, and gap closed.

# 22_future

The next step is to make the selector more hardware-aware and more adaptive. Hardware context, memory behavior, and online feedback could all affect the best algorithm choice. I also want to be clear that the bandit direction is future work, not a validated result in this thesis. Thank you, I am ready for your questions.
