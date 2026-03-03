#!/usr/bin/env python3
"""
Step 3 v2: XGBoost Classifier for Algorithm Selection
====================================================

- Trains a single XGBoost classifier to predict the best algorithm (introsort, heapsort, timsort) from 16 features.
- Outputs: models/xgboost_classifier_v2/, results/xgboost_classifier_v2/
- Evaluates on train, val, test_A, test_B, and real-world splits.
- Saves per-split predictions, confusion matrices, and feature importance.
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
MODEL_DIR = ROOT / "models" / "xgboost_classifier_v2"
RESULTS_DIR = ROOT / "results" / "xgboost_classifier_v2"
ALGORITHMS = ["introsort", "heapsort", "timsort"]
SEED = 42

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
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

def main():
    print("="*70)
    print("STEP 3 v2: XGBoost Classifier for Algorithm Selection")
    print("="*70)
    t0 = time.time()
    # Load data
    train_df = load_split("train")
    val_df = load_split("val")
    test_a_df = load_split("test_A")
    test_b_df = load_split("test_B")
    real_df = load_real_world()
    # Prepare features/labels
    X_train = train_df[FEATURE_NAMES].values
    X_val = val_df[FEATURE_NAMES].values
    X_test_a = test_a_df[FEATURE_NAMES].values
    X_test_b = test_b_df[FEATURE_NAMES].values
    X_real = real_df[FEATURE_NAMES].values
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
    model.save_model(str(MODEL_DIR / "xgb_classifier_v2.json"))
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
    feat_imp = sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])
    # Save results
    results_out = dict(
        timestamp=datetime.now().isoformat(),
        xgb_params=XGB_PARAMS,
        features=FEATURE_NAMES,
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
    print(f"\nArtifacts saved in models/xgboost_classifier_v2/ and results/xgboost_classifier_v2/")
    print(f"Step 3 v2 complete in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
