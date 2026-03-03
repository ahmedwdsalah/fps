#!/usr/bin/env python3
"""
Step 3 v3: XGBoost Classifier with Log-Transformed Features & Pairwise Ranking
=============================================================================
- Uses log(time) as features to reduce scale dominance
- Adds pairwise time-difference features
- Trains classifier to predict best_algorithm
- Outputs: models/xgboost_v3_logpairwise/, results/xgboost_v3_logpairwise/
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_extraction import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "xgboost_v3_logpairwise"
RESULTS_DIR = ROOT / "results" / "xgboost_v3_logpairwise"
ALGORITHMS = ["introsort", "heapsort", "timsort"]
SEED = 42

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=0.8,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
    objective="multi:softprob",
    num_class=3,
)

def load_split(name):
    return pd.read_parquet(ROOT / f"data/benchmark/{name}.parquet")

def load_real_world():
    df = pd.read_parquet(ROOT / "data/real_world_v4/real_world_v4_combined.parquet")
    return df[~df["domain"].isin(["synthetic", "largescale"])].copy()

def add_log_and_pairwise(df):
    # Add log-times
    for algo in ALGORITHMS:
        df[f"log_time_{algo}"] = np.log1p(df[f"time_{algo}"])
    # Add pairwise time differences
    df["diff_introsort_heapsort"] = df["log_time_introsort"] - df["log_time_heapsort"]
    df["diff_introsort_timsort"] = df["log_time_introsort"] - df["log_time_timsort"]
    df["diff_heapsort_timsort"] = df["log_time_heapsort"] - df["log_time_timsort"]
    return df

def main():
    print("="*70)
    print("STEP 3 v3: XGBoost Classifier with Log+Pairwise Features")
    print("="*70)
    t0 = time.time()
    # Load data
    train_df = add_log_and_pairwise(load_split("train"))
    val_df = add_log_and_pairwise(load_split("val"))
    test_a_df = add_log_and_pairwise(load_split("test_A"))
    test_b_df = add_log_and_pairwise(load_split("test_B"))
    real_df = add_log_and_pairwise(load_real_world())
    # Features
    extra_feats = [
        "log_time_introsort", "log_time_heapsort", "log_time_timsort",
        "diff_introsort_heapsort", "diff_introsort_timsort", "diff_heapsort_timsort"
    ]
    ALL_FEATURES = FEATURE_NAMES + extra_feats
    # Prepare features/labels
    X_train = train_df[ALL_FEATURES].values
    X_val = val_df[ALL_FEATURES].values
    X_test_a = test_a_df[ALL_FEATURES].values
    X_test_b = test_b_df[ALL_FEATURES].values
    X_real = real_df[ALL_FEATURES].values
    y_train = train_df["best_algorithm"].values
    y_val = val_df["best_algorithm"].values
    y_test_a = test_a_df["best_algorithm"].values
    y_test_b = test_b_df["best_algorithm"].values
    y_real = real_df["best_algorithm"].values
    # Encode labels
    le = LabelEncoder().fit(ALGORITHMS)
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_a_enc = le.transform(y_test_a)
    y_test_b_enc = le.transform(y_test_b)
    y_real_enc = le.transform(y_real)
    # Train classifier
    print("\n[1/4] Training XGBoost classifier...")
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train_enc,
        eval_set=[(X_train, y_train_enc), (X_val, y_val_enc)]
    )
    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / "xgb_classifier_v3_logpairwise.json"))
    # Evaluate
    print("\n[2/4] Evaluating on all splits...")
    splits = {
        "train": (X_train, y_train, y_train_enc),
        "val": (X_val, y_val, y_val_enc),
        "test_A": (X_test_a, y_test_a, y_test_a_enc),
        "test_B": (X_test_b, y_test_b, y_test_b_enc),
        "real": (X_real, y_real, y_real_enc),
    }
    all_results = {}
    for split, (X, y, y_enc) in splits.items():
        y_pred_enc = model.predict(X)
        y_pred = le.inverse_transform(y_pred_enc)
        acc = accuracy_score(y_enc, y_pred_enc)
        cm = confusion_matrix(y, y_pred, labels=ALGORITHMS)
        report = classification_report(y, y_pred, labels=ALGORITHMS, output_dict=True)
        all_results[split] = dict(
            accuracy=round(acc, 4),
            confusion_matrix=cm.tolist(),
            confusion_labels=ALGORITHMS,
            classification_report=report,
        )
        # Save per-sample predictions
        df = pd.DataFrame({
            "true": y,
            "pred": y_pred,
        })
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULTS_DIR / f"predictions_{split}.csv", index=False)
    # Feature importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(ALL_FEATURES, importance), key=lambda x: -x[1])
    # Save results
    results_out = dict(
        timestamp=datetime.now().isoformat(),
        xgb_params=XGB_PARAMS,
        features=ALL_FEATURES,
        algorithms=ALGORITHMS,
        results=all_results,
        feature_importance=[dict(feature=f, importance=float(i)) for f, i in feat_imp],
    )
    (RESULTS_DIR / "evaluation_results.json").write_text(json.dumps(results_out, indent=2, default=str))
    print("\n[3/4] Results summary:")
    for split, res in all_results.items():
        print(f"  {split:7s}: accuracy={res['accuracy']*100:.1f}%")
    print("\n[4/4] Feature importance:")
    for f, i in feat_imp[:8]:
        print(f"  {f:>22s}: {i:.4f}")
    print(f"\nArtifacts saved in models/xgboost_v3_logpairwise/ and results/xgboost_v3_logpairwise/")
    print(f"Step 3 v3 complete in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
