# Per-Channel Training Notes

## Current Balanced Training Dataset (for algorithm labels)

- Path: `/Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2_hard_m0p05_balanced.csv`
- Total rows: `13,215`

### Class counts (`best_algorithm_v2`)

- `quick_sort`: `2,643`
- `introsort`: `2,643`
- `merge_sort`: `2,643`
- `heap_sort`: `2,643`
- `shell_sort`: `2,643`

## Important context

- This dataset is globally balanced across algorithm classes.
- For one-model-per-channel training, we still need to verify/adjust class balance **inside each channel**.

## Agreed order (must follow)

1. Build a correct per-channel dataset (enough samples per algorithm inside each channel).
2. Train baseline per-channel models on that dataset.
3. Run Optuna tuning only after dataset quality is confirmed.

### Why

- Hyperparameter tuning cannot fix missing/rare classes in a channel.
- Model optimization before dataset correctness wastes time and compute.

## Target algorithms (latest)

- `quick_sort`
- `introsort`
- `merge_sort`
- `heap_sort`
- `shell_sort`

## 9-model training paths

- Training script: `/Users/ahmed/Desktop/thesis/My-Master-thesis/scripts/train_f1_9_channel_models.py`
- Models root (9 channel models): `/Users/ahmed/Desktop/thesis/My-Master-thesis/models/xgboost_f1_9_channels/`
- Results root (per-channel eval): `/Users/ahmed/Desktop/thesis/My-Master-thesis/results/xgboost_f1_9_channels/`
- Dataset loaded by script (`DATA_CSV`):
  `/Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2_hard_m0p05_balanced.csv`
- Index loaded by script (`INDEX_CSV`):
  `/Volumes/k/thesis_data/f1_only_1m_packed/index.csv`

## Raw actual arrays (source of features)

- Raw arrays file: `/Volumes/k/thesis_data/f1_only_1m_packed/raw_arrays.h5`
- Index map file: `/Volumes/k/thesis_data/f1_only_1m_packed/index.csv`
- Join key: `file`
  - Example key format: `f1_2024_R4_Q_20_L1_RPM.h5arr`
  - Same `file` key appears in index and training CSV rows.

### Live sample (first 5 values per channel, from `raw_arrays.h5`)

- `Speed` (`f1_2024_R1_R_1_L1_Speed.h5arr`): `[0.0, 0.0, 3.0, 11.0, 17.0]`, `370 values in the array, per lap, per driver, per session`
- `Throttle` (`f1_2024_R1_R_1_L1_Throttle.h5arr`): `[15.0, 15.0, 15.0, 15.0, 15.0]`, `370 values in the array, per lap, per driver, per session`
- `RPM` (`f1_2024_R1_R_1_L1_RPM.h5arr`): `[9963.0, 9755.0, 8495.0, 6815.0, 5695.0]`, `370 values in the array, per lap, per driver, per session`
- `nGear` (`f1_2024_R1_R_1_L1_nGear.h5arr`): `[1.0, 1.0, 1.0, 1.0, 1.0]`, `370 values in the array, per lap, per driver, per session`
- `DRS` (`f1_2024_R1_R_1_L1_DRS.h5arr`): `[1.0, 1.0, 1.0, 1.0, 1.0]`, `370 values in the array, per lap, per driver, per session`
- `X` (`f1_2024_R1_R_1_L1_X.h5arr`): `[-280.0, -280.0, -280.0, -280.0, -279.0]`, `372 values in the array, per lap, per driver, per session`
- `Y` (`f1_2024_R1_R_1_L1_Y.h5arr`): `[3550.0, 3550.0, 3558.0, 3574.0, 3585.0]`, `372 values in the array, per lap, per driver, per session`
- `Z` (`f1_2024_R1_R_1_L1_Z.h5arr`): `[-157.0, -157.0, -157.0, -157.0, -157.0]`, `372 values in the array, per lap, per driver, per session`
- `Distance` (`f1_2024_R1_R_1_L1_Distance.h5arr`): `[0.0, 0.0, 0.2, 0.9333333333333333, 1.688888888888889]`, `370 values in the array, per lap, per driver, per session`

## Focus (new direction)

- Focus root only: `/Volumes/k/thesis_data/f1_only_1m_packed`
- Focus training file: `/Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2_hard_m0p05_balanced.csv`
- Use index only for channel mapping: `/Volumes/k/thesis_data/f1_only_1m_packed/index.csv`
- Treat `/Volumes/k/thesis_data/f1_only` as archive (do not mix outputs).

## What v5 does at inference

- Input: one numeric array (example: `[1,2,3,4,...]`).
- Step 1: extract array-structure features (e.g., sortedness, runs, entropy, skewness).
- Step 2: send feature vector to trained XGBoost v5 model.
- Step 3: model outputs probabilities for 3 classes:
  - `introsort`
  - `heapsort`
  - `timsort`
- Step 4: top-probability class is predicted best algorithm.

Example output shape:
- `best_algorithm: timsort`
- `probs: {introsort: 0.05, heapsort: 0.12, timsort: 0.83}`

Exactly. Final output is one algorithm.

But training setup still matters:

Model chooses one from candidate set.
If candidate set = 5 classes, model must learn boundaries among those 5.
In some channels, not all 5 are present/support enough.
So do this:

Per-channel model: use channel-specific candidate set (only algorithms with enough rows in that channel).
Output still one algorithm, not five.
If needed for thesis consistency, map missing algorithms as “not eligible for this channel.”


ahmed@Ahmeds-Mac-mini My-Master-thesis % source venv/bin/activate 
(venv) ahmed@Ahmeds-Mac-mini My-Master-thesis % python3 scripts/train_f1_9_channel_models_dynamic_v2.py

================================================================================
TRAIN F1 9 CHANNEL MODELS (DYNAMIC CLASSES V2)
================================================================================
Data:  /Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2_hard_m0p05_balanced.csv
Index: /Volumes/k/thesis_data/f1_only_1m_packed/index.csv
Min class count per channel: 20
[DRS] classes=['heap_sort', 'merge_sort', 'shell_sort'] acc=0.790 bal_acc=0.549
[Distance] classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort'] acc=0.601 bal_acc=0.399
[RPM] classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort'] acc=0.468 bal_acc=0.412
[Speed] classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort'] acc=0.411 bal_acc=0.437
[Throttle] classes=['heap_sort', 'introsort', 'merge_sort', 'shell_sort'] acc=0.595 bal_acc=0.372
[X] classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort'] acc=0.564 bal_acc=0.489
[Y] classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort'] acc=0.524 bal_acc=0.452
[Z] classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort'] acc=0.546 bal_acc=0.480
[nGear] classes=['heap_sort', 'merge_sort', 'shell_sort'] acc=0.696 bal_acc=0.582

Saved: /Users/ahmed/Desktop/thesis/My-Master-thesis/results/f1_9_channel_models_dynamic_v2/manifest.json
(venv) ahmed@Ahmeds-Mac-mini My-Master-thesis % python3 scripts/eval_f1_dynamic_router_v2.py

================================================================================
EVAL F1 DYNAMIC ROUTER V2
================================================================================
Data:   /Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2_hard_m0p05_balanced.csv
Index:  /Volumes/k/thesis_data/f1_only_1m_packed/index.csv
Models: /Users/ahmed/Desktop/thesis/My-Master-thesis/models/f1_9_channel_models_dynamic_v2
Test rows: 3,965
Metrics:
  routed      acc=0.860  bal_acc=0.860
  channel_sbs acc=0.507  bal_acc=0.507
  global_sbs  acc=0.200  bal_acc=0.200
Saved: /Users/ahmed/Desktop/thesis/My-Master-thesis/results/f1_9_channel_models_dynamic_v2/router_eval.json
(venv) ahmed@Ahmeds-Mac-mini My-Master-thesis % python3 scripts/train_eval_f1_dynamic_router_v2_strict.py

================================================================================
STRICT TRAIN+EVAL: F1 DYNAMIC ROUTER V2
================================================================================
Data:  /Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2_hard_m0p05_balanced.csv
Index: /Volumes/k/thesis_data/f1_only_1m_packed/index.csv
Min class count per channel: 20
[DRS] trained classes=['heap_sort', 'merge_sort', 'shell_sort']
[Distance] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[RPM] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Speed] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Throttle] trained classes=['heap_sort', 'introsort', 'merge_sort', 'shell_sort']
[X] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Y] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Z] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[nGear] trained classes=['heap_sort', 'merge_sort', 'shell_sort']

Strict test metrics:
  routed      acc=0.554  bal_acc=0.554
  channel_sbs acc=0.507  bal_acc=0.507
  global_sbs  acc=0.200  bal_acc=0.200
Saved: /Users/ahmed/Desktop/thesis/My-Master-thesis/results/f1_9_channel_models_dynamic_v2_strict/strict_router_eval.json
(venv) ahmed@Ahmeds-Mac-mini My-Master-thesis % python3 scripts/train_eval_f1_dynamic_router_v2_strict.py --min-class-count 50

================================================================================
STRICT TRAIN+EVAL: F1 DYNAMIC ROUTER V2
================================================================================
Data:  /Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2_hard_m0p05_balanced.csv
Index: /Volumes/k/thesis_data/f1_only_1m_packed/index.csv
Min class count per channel: 50
[DRS] trained classes=['heap_sort', 'shell_sort']
[Distance] trained classes=['heap_sort', 'introsort', 'quick_sort', 'shell_sort']
[RPM] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Speed] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Throttle] trained classes=['heap_sort', 'merge_sort', 'shell_sort']
[X] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Y] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Z] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[nGear] trained classes=['heap_sort', 'merge_sort']
Traceback (most recent call last):
  File "/Users/ahmed/Desktop/thesis/My-Master-thesis/scripts/train_eval_f1_dynamic_router_v2_strict.py", line 301, in <module>
    main()
    ~~~~^^
  File "/Users/ahmed/Desktop/thesis/My-Master-thesis/scripts/train_eval_f1_dynamic_router_v2_strict.py", line 243, in main
    pred_idx = int(model.predict(x)[0])
TypeError: only 0-dimensional arrays can be converted to Python scalars
(venv) ahmed@Ahmeds-Mac-mini My-Master-thesis % python3 scripts/train_eval_f1_dynamic_router_v2_strict.py --min-class-count 50

        ================================================================================
STRICT TRAIN+EVAL: F1 DYNAMIC ROUTER V2
================================================================================
Data:  /Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2_hard_m0p05_balanced.csv
Index: /Volumes/k/thesis_data/f1_only_1m_packed/index.csv
Min class count per channel: 50
[DRS] trained classes=['heap_sort', 'shell_sort']
[Distance] trained classes=['heap_sort', 'introsort', 'quick_sort', 'shell_sort']
[RPM] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Speed] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Throttle] trained classes=['heap_sort', 'merge_sort', 'shell_sort']
[X] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Y] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Z] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[nGear] trained classes=['heap_sort', 'merge_sort']

Strict test metrics:
  routed      acc=0.554  bal_acc=0.554
  channel_sbs acc=0.507  bal_acc=0.507
  global_sbs  acc=0.200  bal_acc=0.200
Saved: /Users/ahmed/Desktop/thesis/My-Master-thesis/results/f1_9_channel_models_dynamic_v2_strict/strict_router_eval.json
(venv) ahmed@Ahmeds-Mac-mini My-Master-thesis %         python3 scripts/train_eval_f1_dynamic_router_v2_strict.py --min-class-count 100

================================================================================
STRICT TRAIN+EVAL: F1 DYNAMIC ROUTER V2
================================================================================
Data:  /Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2_hard_m0p05_balanced.csv
Index: /Volumes/k/thesis_data/f1_only_1m_packed/index.csv
Min class count per channel: 100
[DRS] trained classes=['heap_sort', 'shell_sort']
[Distance] trained classes=['introsort', 'shell_sort']
[RPM] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Speed] trained classes=['heap_sort', 'introsort', 'merge_sort', 'quick_sort', 'shell_sort']
[Throttle] trained classes=['heap_sort', 'merge_sort']
[X] trained classes=['merge_sort', 'quick_sort', 'shell_sort']
[Y] trained classes=['merge_sort', 'quick_sort']
[Z] trained classes=['heap_sort', 'merge_sort', 'quick_sort', 'shell_sort']
[nGear] trained classes=['heap_sort', 'merge_sort']

Strict test metrics:
  routed      acc=0.555  bal_acc=0.555
  channel_sbs acc=0.507  bal_acc=0.507
  global_sbs  acc=0.200  bal_acc=0.200
