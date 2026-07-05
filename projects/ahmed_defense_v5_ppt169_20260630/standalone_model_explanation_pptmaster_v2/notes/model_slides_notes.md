# Standalone Model Slide Notes

These notes are for review only. They are not inserted into the official deck yet.

## M01: The model learns class probabilities, not exact runtime

I should explain this as: our final model is not trying to predict exact runtime. For each array, the 16 structural features go into XGBoost, and the model outputs probabilities for introsort, heapsort, and timsort. The selected algorithm is the one with the highest probability.

The training objective is `multi:softprob`, and the evaluation metric during training is `mlogloss`. This is the multi-class version of log loss. Runtime regret is not the training loss; regret is used after training to judge whether a wrong selection actually costs runtime.

## M02: The training signal was corrected before trees learned structure

The raw labels are biased toward timsort. If we trained naively, the model could learn an easy majority behavior and predict timsort too often.

So before the trees learn structure, the training setup controls the signal with near-tie filtering, majority-class undersampling, and inverse-frequency sample weights. The weight formula is `total / (num_classes * class_count)`. This makes introsort and heapsort matter more in the loss than they would in the raw dataset.

The simple speaking point: XGBoost starts from a baseline, but our data setup stops the baseline from becoming the whole model.

## M03: Each tree corrects the class scores

XGBoost does not move the data point. It moves the raw class scores for introsort, heapsort, and timsort. Softmax turns those scores into probabilities.

For each array, XGBoost compares the current probabilities with the true fastest label. Then it computes first-order and second-order information under multi-class log loss. The gradient tells the direction: should this class score go up or down? The Hessian tells the curvature: how careful should the correction be?

That is why XGBoost is not just fitting mistakes like a simple boosting model. Each tree uses direction and curvature to choose splits and leaf values.

## M04: The ensemble is powerful because each correction is constrained

The final model is an ensemble of many trees, but no single tree is allowed to dominate. The learning rate is small, depth is limited, small leaves are controlled, rows and features are subsampled, and regularisation is active.

For final v5, the important settings are `n_estimators = 500`, `max_depth = 7`, `learning_rate = 0.05`, `subsample = 0.8`, `colsample_bytree = 0.8`, `min_child_weight = 5`, `reg_alpha = 0.1`, `reg_lambda = 1.0`, `tree_method = hist`, `objective = multi:softprob`, and `eval_metric = mlogloss`.

Say `row subsampling` and `feature subsampling`, not bootstrap, unless we prove sampling with replacement.
