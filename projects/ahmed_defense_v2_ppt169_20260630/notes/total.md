# P01_title
This thesis is about a small decision that happens before sorting: which implementation should we use for this input array. I will keep coming back to three numbers: the model closes 93.1 percent of the runtime gap, reaches 76.1 percent test accuracy, and makes zero-regret decisions for 89.6 percent of the test cases.

# P02_route
I will first explain the task and why it matters. Then I will explain the data, the extracted features, and the model. After that I will defend the result with runtime metrics, then show the limits and what I do not claim.

# P03_task
The task is not to invent a new sorting algorithm. The task is: given one array before sorting, choose introsort, heapsort, or timsort. The model sees cheap structural features and outputs one algorithm choice.

# P04_importance
This matters because the same input size can have different structure. A sorted-like array, random array, and duplicated array can reward different algorithms. So the useful target is not only classification accuracy; it is how much runtime loss we avoid compared with the best per-input choice.

# P05_related
The work connects to algorithm selection, adaptive sorting, and learned performance prediction. I am not presenting literature as a separate survey. I use it to position the task: previous work gives the idea of selecting from features, and this thesis applies that idea to measured sorting behavior.

# P06_objective
The research question is whether simple array structure can predict the best sorting algorithm. I evaluate that question in two ways: whether the label is correct, and how expensive a mistake is when it is not correct.

# P07_domains
The dataset was built from five domains: F1 telemetry, stock, crypto, earthquake, and weather. The reason for using multiple domains was to avoid building a selector that only understands one family of arrays.

# P08_pipeline
For each domain, I generated transformed arrays and measured runtime across the three sorting algorithms. The label is not guessed; it is the fastest measured algorithm for that instance. The transformations create raw, reversed, shuffled, quantized, and partially sorted cases.

# P09_features
The features are the bridge between the raw array and the model. They describe scale, order, duplicates, runs, inversions, and distribution. The important point is that these are structural features, not timing features.

# P10_feature_cost
At prediction time, the selector cannot sort the array with all three algorithms and then choose the fastest. That would be the oracle, not a usable method. The selector must use cheaper information from the input before sorting.

# P11_system
The main system is v5. It is one general XGBoost classifier trained across all domains and choosing among the three algorithms. The F1-specific route is separate and I will only mention it as a specialization, not the main result.

# P12_portfolio
The portfolio is deliberately limited to three real implementations: introsort, heapsort, and timsort. The model is not choosing from every sorting algorithm in theory. It is choosing from the measured implementations in this experiment.

# P13_model_journey
The final model came after earlier paths were rejected. Regression was not strong enough, timing features were not deployable, and strict checks changed the interpretation. v5 became the main model because it gave the best practical balance under the submitted evaluation.

# P14_evaluation
SBS means one algorithm used for all test inputs. VBS means the fastest measured algorithm per input, which is an oracle for evaluation. The model is judged by how close it gets to VBS without having oracle information at prediction time.

# P15_main_result
The main result is runtime value: v5 closes 93.1 percent of the gap between SBS and VBS. Accuracy is 76.1 percent, but this does not tell the whole story. The zero-regret rate is 89.6 percent, which shows many decisions are either correct or not costly.

# P16_accuracy_regret
This slide explains why I do not defend the model only as a classifier. Some label mistakes happen between algorithms whose measured runtimes are very close. In those cases, the label is wrong, but the runtime regret is small.

# P17_confusion
The confusion matrix shows that timsort is often easier to identify when order signals are strong. The harder boundary is between introsort and heapsort because their runtime behavior is often closer. This is a limitation, but it is also why regret is necessary.

# P18_importance
Feature importance shows that the model is using structural signals. This supports the method because the important features relate back to the properties that sorting algorithms react to: order, duplicates, distribution, and size.

# P19_domain_holdout
Domain holdout is stricter because the held-out domain is unseen during training. The weighted gap closed is 79.7 percent. F1 is harder than the other domains, and that is why the F1-specific route exists separately.

# P20_rigor
The negative results are part of the rigor. v6 checked source-awareness, v7 showed that a regret-aware attempt did not improve the final result, and v8 showed that the intro-versus-heap boundary remains difficult. These results keep the claim measured.

# P21_limits
My claim is that v5 closes most of the measured runtime gap with cheap structural features and known limitations. I do not claim a perfect classifier, I do not claim the F1 route is the main result, and I do not claim LinUCB is validated in this thesis.

# P22_qa
The contribution is the framing, the data and feature pipeline, and the measured selector result. The future work is stronger domain adaptation, validated online selection, and deeper F1-specific evaluation.
