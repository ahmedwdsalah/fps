# Model PPTX Notes

This file is for us, not final slide text. Keep it simple. Write it like normal explanation, not formal thesis language.

## Loss / Objective

Our final model is not regression and not binary classification.

It is: one array comes in, and the model chooses one of three algorithms:

- introsort
- heapsort
- timsort

So we used XGBoost as a three-class classifier:

- `objective = multi:softprob`
- `eval_metric = mlogloss`

`multi:softprob` means the model outputs probabilities for all three algorithms.

Example:

- introsort: `0.12`
- heapsort: `0.18`
- timsort: `0.70`

Then we pick the highest probability.

`mlogloss` is just multi-class log loss. It is the multi-class version of logistic/log loss.

So if someone says "XGBoost usually uses logistic loss or log loss", our answer is:

yes, and ours is the multi-class version of that because we have three classes, not two.

We did not keep regression loss because predicting exact runtimes made the model care too much about array size and noisy timing values.

Regret was not the final training loss. Regret is how we evaluate if a wrong class actually costs runtime.

We tried regret-aware training later, but the final model stayed with `multi:softprob` because it was more stable.

## Baseline Prediction

XGBoost starts with a baseline before it grows trees.

For our model, that baseline is an initial score/probability tendency over the three classes.

Do not confuse this with SBS/VBS:

- XGBoost baseline = internal starting prediction before trees.
- SBS/VBS = runtime evaluation baselines in the thesis.

Our raw data was very biased:

- timsort wins about `85.0%`
- heapsort wins about `10.5%`
- introsort wins about `4.4%`

So if we trained naively, the starting behavior could become basically "timsort wins everything".

We did not want that.

So we controlled it with:

- near-tie filtering,
- majority-class undersampling,
- inverse-frequency sample weights.

The sample weight formula in code is:

`weight = total / (num_classes * class_count)`

Meaning: smaller classes get stronger weight, so they are not drowned by timsort.

So the simple way to say it:

XGBoost starts from a baseline, but our training setup made the loss more balanced before the trees started learning feature structure.

## What Moves Up Or Down

Important: XGBoost does not move the data point up or down.

It moves the model scores.

For each array, the model has three raw scores:

- introsort score
- heapsort score
- timsort score

Then softmax turns those scores into probabilities.

During training, XGBoost asks:

if I push one of these scores slightly up or down, does `mlogloss` improve?

That is where gradients and Hessians come in:

- gradient = which direction to move the score,
- Hessian = how strong / curved that correction is.

Then the next tree learns those corrections.

Example:

If the true label is timsort but the model gives timsort low probability, the gradient pushes the timsort score up and/or pushes the wrong class scores down.

If the model is already correct and confident, the correction is smaller.

If the row is from a minority class, sample weighting makes that row matter more.

## Second-Order Information

Yes, we have this in our model.

The "second information" is the Hessian.

In XGBoost training:

first-order information = gradient

- tells the model the direction to move,
- asks: "should this class score go up or down?"

second-order information = Hessian

- tells the model how curved/strong the loss is at that point,
- asks: "how careful should the model be with this correction?"

For our case, this happens under:

`multi:softprob` + `mlogloss`

So for every array, XGBoost looks at the current class probabilities for:

- introsort,
- heapsort,
- timsort.

Then it compares those probabilities with the true fastest label.

Then it computes both:

- gradient of multi-class log loss,
- Hessian of multi-class log loss.

That second-order information is one reason XGBoost is not just "fit errors like a simple boosting model." It uses both the direction and the curvature to decide better splits and leaf values.

## What We Controlled

Final v5 artifact values from `results/xgboost_v5/evaluation_results.json` and `methodology_assets/metrics_summary.json`:

- `n_estimators = 500`
- `max_depth = 7`
- `learning_rate = 0.05`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `min_child_weight = 5`
- `reg_alpha = 0.1`
- `reg_lambda = 1.0`
- `tree_method = hist`
- `objective = multi:softprob`
- `eval_metric = mlogloss`
- `random_state = 42`

What these mean in simple words:

- `learning_rate`: how big each correction step is.
- `max_depth`: how complex each tree can be.
- `n_estimators`: how many trees we allow.
- `min_child_weight`: stops tiny weak leaves from being trusted too much.
- `subsample`: each tree sees only part of the rows.
- `colsample_bytree`: each tree sees only part of the features.
- `reg_alpha`: L1 regularisation.
- `reg_lambda`: L2 regularisation.
- `sample_weight`: makes minority classes matter more.

Data controls:

- near-tie filtering removed ambiguous labels,
- majority undersampling capped majority at `3x` minority class,
- inverse-frequency sample weights balanced class contribution,
- `70/15/15` train/validation/test split,
- seed `42`.

## Randomness / Stability

Yes, we used this idea.

But say it carefully.

For our final v5 artifact:

- `subsample = 0.8`
- `colsample_bytree = 0.8`

`subsample = 0.8` means each tree trains using only `80%` of the training rows.

This is row subsampling.

`colsample_bytree = 0.8` means each tree sees only `80%` of the features.

This is feature subsampling per tree.

So yes, we injected controlled randomness/stability into the ensemble.

Why this matters:

- it reduces overfitting,
- it stops every tree from depending on the exact same rows,
- it stops every tree from depending on the exact same feature set,
- it makes the trees more diverse,
- the ensemble becomes more stable.

Important wording caution:

Do not call it "bootstrap" unless we prove sampling with replacement.

For our PPT/model explanation, call it:

- row subsampling,
- feature subsampling.

Good way to say it:

In our final v5 XGBoost model, `subsample = 0.8` means each tree was trained on only 80% of the rows, and `colsample_bytree = 0.8` means each tree saw only 80% of the features. This adds controlled randomness, reduces overfitting, and makes the ensemble less dependent on one exact sample or one exact feature set.

## TikTok-Style Explanation Adapted To Our Thesis

This is the kind of explanation style we like for the model slides.

Use this idea, but always keep it tied to our thesis.

XGBoost turns the abstract idea of "learning from mistakes" into a very concrete training loop.

For our thesis, the model is always asking three practical questions:

- what sorting choice is the model currently getting wrong?
- what correction would move the prediction closer to the true fastest algorithm?
- how do we make that correction without letting the model become unstable or overfit to timing noise?

In our case, the model is not trying to predict the exact runtime directly.

It is trying to choose one class:

- introsort,
- heapsort,
- timsort.

So the training problem becomes:

given the 16 structural features of an array, learn which algorithm should receive the highest probability.

The process begins with the loss function.

For us this is the multi-class log-loss setup:

- `objective = multi:softprob`
- `eval_metric = mlogloss`

This loss punishes the model when it gives high confidence to the wrong sorting algorithm.

Example:

if the true fastest algorithm is timsort, but the model gives high probability to heapsort, that is a bad mistake.

If the model is uncertain or gives more probability to the correct algorithm, the penalty is lower.

So the loss function defines the personality of the model:

it does not only ask "was the class right or wrong?"

It also asks "how confident was the model when it was wrong?"

After that, XGBoost starts from a baseline prediction.

For our data, this matters because the raw labels were very skewed toward timsort.

If we did nothing, the model could start behaving like:

"just predict timsort most of the time."

But that is not what we wanted.

So before the trees learn structure, we controlled the training data with:

- near-tie filtering,
- majority-class undersampling,
- inverse-frequency sample weights.

This means the model is not allowed to only learn the easy timsort majority pattern.

The smaller classes, especially introsort and heapsort, still get enough force in the loss.

Once the baseline is set, XGBoost looks at every array and asks:

if I slightly move the score for introsort, heapsort, or timsort up or down, how does the log loss change?

That first signal is the gradient.

The gradient does not just say:

"this array was wrong."

It says:

"this class score should move in this direction, and this is how urgent the correction is."

Then XGBoost also computes the Hessian.

That is the second-order information.

The Hessian tells the model about the curvature of the loss at the current point.

So now the model knows two things:

- gradient: which direction the score should move,
- Hessian: how careful or strong the correction should be.

This matters because not every mistake should cause a huge correction.

If the loss is very sensitive at that point, a big move can make the model unstable.

If the loss is flatter, the model can move more safely.

So XGBoost is not blindly chasing mistakes.

It is using both direction and curvature.

Then the decision trees come in.

Each new tree tries to group arrays that need similar correction.

For our thesis, this means the tree might learn things like:

- arrays with long natural runs push probability toward timsort,
- arrays with more disorder may push probability away from timsort,
- duplicated or low-diversity arrays may create different boundaries,
- size and structural features interact, so the same sortedness signal may not mean the same thing at every length.

Every possible split is tested by asking:

does this split reduce the multi-class log loss enough to justify adding another branch?

That is where gain comes in.

A split is useful only if it creates groups where the correction becomes clearer.

This is also why our feature importance is based on gain:

features with high gain are the features that repeatedly helped the model reduce the objective.

Once the tree structure is chosen, every leaf gets a correction value.

That leaf correction says:

for arrays that land in this leaf, adjust the class scores by this amount.

Then the learning rate controls how big that correction is.

For us:

- `learning_rate = 0.05`

So every tree is intentionally small in its effect.

No single tree is allowed to take over the whole model.

The model improves through many controlled corrections.

After one tree is added, XGBoost recomputes the predictions.

Then it recomputes the gradients and Hessians.

Then the next tree works on the new remaining mistakes.

So each tree is not repeating the same work.

The early trees learn the big obvious patterns.

Later trees focus on smaller remaining boundaries, like the harder introsort-heapsort region.

This is why XGBoost can model non-linear feature interactions in our data.

It can learn that sorting behavior is not controlled by one feature alone.

It is controlled by combinations:

- length,
- runs,
- sortedness,
- duplicates,
- entropy,
- outliers,
- dispersion.

To avoid overfitting, our model also adds controlled randomness.

For final v5:

- `subsample = 0.8`
- `colsample_bytree = 0.8`

So each tree sees only 80% of the rows and 80% of the features.

This makes the trees less identical to each other.

It also stops the model from depending too much on one exact sample or one exact feature set.

We should call this row subsampling and feature subsampling.

Do not call it bootstrap unless we prove sampling with replacement.

At the end, the prediction is the sum of many small corrections.

The final output is a probability distribution over the three algorithms.

Then we choose the algorithm with the highest probability.

And after training, we do not judge the model only by accuracy.

Accuracy tells us whether the top class matched the fastest label.

But our real question is runtime value.

So we evaluate with:

- regret,
- zero-regret percentage,
- SBS-to-VBS gap closed.

That is why the final story is:

XGBoost learns the class probabilities with multi-class log loss, but the thesis proves the selector's value using runtime-aware metrics.

## Version Caution

There is a mismatch we must not accidentally mix.

The final saved v5 artifact used for the deck numbers reports:

- `500` trees,
- depth `7`,
- `subsample = 0.8`,
- `colsample_bytree = 0.8`.

The submitted thesis text has a table with a later/alternate setup:

- `800` trees,
- depth `8`,
- `subsample = 0.85`,
- `colsample_bytree = 0.85`,
- early stopping.

For the defense deck, use the final artifact values when talking about the reported v5 result, unless we decide to align the slide wording with the submitted thesis table.
