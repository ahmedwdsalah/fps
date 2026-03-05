#!/usr/bin/env python3
"""
Domain Holdout Test (Leave-One-Domain-Out)
==========================================
The strongest generalization test: for each of the 5 domains,
train on the other 4 and test on the held-out domain the model
has NEVER seen.

Also tests the existing v5 model per-domain for comparison.

Outputs:
    results/domain_holdout/domain_holdout_results.json

Usage:
    python3 scripts/domain_holdout_test.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
DATA_CSV    = ROOT / "data" / "training_dataset.csv"
MODEL_PATH  = ROOT / "models" / "xgboost_v5" / "xgb_v5.json"
OUT_DIR     = ROOT / "results" / "domain_holdout"

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
SEED = 42

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
)


def balanced_undersample(df: pd.DataFrame, label_col: str,
                         max_ratio: float = 3.0) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    min_count = counts.min()
    cap = int(min_count * max_ratio)
    parts = []
    for cls in counts.index:
        subset = df[df[label_col] == cls]
        if len(subset) > cap:
            subset = subset.sample(n=cap, random_state=SEED)
        parts.append(subset)
    result = pd.concat(parts, ignore_index=True)
    return result.sample(frac=1, random_state=SEED).reset_index(drop=True)


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[yi] for yi in y], dtype=np.float64)


def regret_metrics(y_true, y_pred, time_cols_df):
    """Compute regret metrics: how much time does the model waste?"""
    t = time_cols_df[["time_introsort", "time_heapsort", "time_timsort"]].values
    algo_idx = {"introsort": 0, "heapsort": 1, "timsort": 2}

    vbs_total = t.min(axis=1).sum()
    model_total = sum(t[i, algo_idx[p]] for i, p in enumerate(y_pred))
    sbs_algo = ALGORITHMS[np.argmin([t[:, j].sum() for j in range(3)])]
    sbs_total = t[:, algo_idx[sbs_algo]].sum()

    gap = (sbs_total - vbs_total) / vbs_total * 100
    if sbs_total - vbs_total > 0:
        gap_closed = (1 - (model_total - vbs_total) / (sbs_total - vbs_total)) * 100
    else:
        gap_closed = 100.0
    model_regret = (model_total - vbs_total) / vbs_total * 100

    per_regret = np.array([
        t[i, algo_idx[p]] - t[i].min() for i, p in enumerate(y_pred)
    ])

    return dict(
        vbs_total_s=round(vbs_total, 4),
        sbs_total_s=round(sbs_total, 4),
        sbs_algo=sbs_algo,
        model_total_s=round(model_total, 4),
        gap_pct=round(gap, 2),
        gap_closed_pct=round(gap_closed, 2),
        model_regret_pct=round(model_regret, 2),
        per_instance_regret_mean_us=round(per_regret.mean() * 1e6, 2),
        perfect_pick_pct=round((per_regret == 0).mean() * 100, 2),
    )


def evaluate_fold(train_df, test_df, fold_name, le):
    """Train on train_df, test on test_df, return metrics dict."""
    t_start = time.time()

    # Balance training set
    train_bal = balanced_undersample(train_df, "best_algorithm", max_ratio=3.0)

    X_train = train_bal[FEATURE_NAMES].values
    y_train = train_bal["best_algorithm"].values
    y_train_enc = le.transform(y_train)
    weights = compute_sample_weights(y_train_enc)

    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df["best_algorithm"].values

    # Train
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train_enc, sample_weight=weights, verbose=0)

    # Predict
    y_pred = le.inverse_transform(model.predict(X_test))

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=ALGORITHMS,
                                   output_dict=True, zero_division=0)

    # Regret
    regret = regret_metrics(y_test, y_pred, test_df)

    elapsed = time.time() - t_start

    return dict(
        fold=fold_name,
        train_size=len(train_bal),
        test_size=len(test_df),
        accuracy=round(acc, 4),
        balanced_accuracy=round(bal_acc, 4),
        per_class_recall={a: round(report[a]["recall"], 4) for a in ALGORITHMS},
        regret=regret,
        train_time_s=round(elapsed, 1),
    )


def main():
    print("=" * 70)
    print("  DOMAIN HOLDOUT TEST — Leave-One-Domain-Out")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/3] Loading training_dataset.csv ...")
    df = pd.read_csv(DATA_CSV)
    print(f"  {len(df):,} rows")

    domains = sorted(df["domain"].unique())
    print(f"  Domains: {domains}")
    for d in domains:
        n = (df.domain == d).sum()
        print(f"    {d:>12s}: {n:>10,}")

    le = LabelEncoder().fit(ALGORITHMS)

    # ── Test existing v5 model per domain ────────────────────────────────
    print("\n[2/3] Testing EXISTING v5 model per domain ...")
    existing_model = xgb.XGBClassifier()
    existing_model.load_model(str(MODEL_PATH))

    existing_results = {}
    for domain in domains:
        mask = df.domain == domain
        test_d = df[mask].reset_index(drop=True)
        X_d = test_d[FEATURE_NAMES].values
        y_d = test_d["best_algorithm"].values
        y_pred = le.inverse_transform(existing_model.predict(X_d))

        acc = accuracy_score(y_d, y_pred)
        bal_acc = balanced_accuracy_score(y_d, y_pred)
        regret = regret_metrics(y_d, y_pred, test_d)

        existing_results[domain] = dict(
            accuracy=round(acc, 4),
            balanced_accuracy=round(bal_acc, 4),
            regret=regret,
        )
        print(f"    {domain:>12s}: acc={acc*100:5.1f}%  bal_acc={bal_acc*100:5.1f}%  "
              f"gap_closed={regret['gap_closed_pct']:5.1f}%")

    # ── Leave-One-Domain-Out ─────────────────────────────────────────────
    print("\n[3/3] Leave-One-Domain-Out (train on 4, test on 1) ...")
    print("      This trains 5 fresh models — one per fold.\n")

    holdout_results = {}

    for i, holdout_domain in enumerate(domains, 1):
        print(f"  ── Fold {i}/5: hold out {holdout_domain} ──")

        train_df = df[df.domain != holdout_domain].reset_index(drop=True)
        test_df  = df[df.domain == holdout_domain].reset_index(drop=True)

        print(f"     Train: {len(train_df):,} (from {', '.join(d for d in domains if d != holdout_domain)})")
        print(f"     Test:  {len(test_df):,} ({holdout_domain})")

        result = evaluate_fold(train_df, test_df, holdout_domain, le)
        holdout_results[holdout_domain] = result

        r = result
        print(f"     Accuracy:          {r['accuracy']*100:5.1f}%")
        print(f"     Balanced accuracy: {r['balanced_accuracy']*100:5.1f}%")
        print(f"     Per-class recall:  intro={r['per_class_recall']['introsort']*100:.1f}%  "
              f"heap={r['per_class_recall']['heapsort']*100:.1f}%  "
              f"tim={r['per_class_recall']['timsort']*100:.1f}%")
        print(f"     Gap closed:        {r['regret']['gap_closed_pct']:5.1f}%")
        print(f"     Perfect picks:     {r['regret']['perfect_pick_pct']:5.1f}%")
        print(f"     Trained in {r['train_time_s']:.0f}s\n")

    # ── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY: DOMAIN HOLDOUT TEST")
    print("=" * 70)
    print(f"\n  {'Holdout':>12s}  {'N':>8s}  {'Acc':>6s}  {'BalAcc':>6s}  "
          f"{'Gap%':>6s}  {'Closed':>7s}  {'Perfect':>7s}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")

    for domain in domains:
        r = holdout_results[domain]
        rg = r["regret"]
        print(f"  {domain:>12s}  {r['test_size']:>8,}  "
              f"{r['accuracy']*100:>5.1f}%  {r['balanced_accuracy']*100:>5.1f}%  "
              f"{rg['gap_pct']:>5.1f}%  {rg['gap_closed_pct']:>6.1f}%  "
              f"{rg['perfect_pick_pct']:>6.1f}%")

    # Weighted average
    total_n = sum(holdout_results[d]["test_size"] for d in domains)
    w_acc = sum(holdout_results[d]["accuracy"] * holdout_results[d]["test_size"] for d in domains) / total_n
    w_bal = sum(holdout_results[d]["balanced_accuracy"] * holdout_results[d]["test_size"] for d in domains) / total_n
    w_closed = sum(holdout_results[d]["regret"]["gap_closed_pct"] * holdout_results[d]["test_size"] for d in domains) / total_n
    w_perf = sum(holdout_results[d]["regret"]["perfect_pick_pct"] * holdout_results[d]["test_size"] for d in domains) / total_n

    print(f"  {'─'*12}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")
    print(f"  {'WEIGHTED':>12s}  {total_n:>8,}  "
          f"{w_acc*100:>5.1f}%  {w_bal*100:>5.1f}%  {'':>6s}  {w_closed:>6.1f}%  "
          f"{w_perf:>6.1f}%")

    print(f"\n  Comparison: existing v5 model (trained on ALL domains):")
    print(f"  {'Domain':>12s}  {'Acc':>6s}  {'BalAcc':>6s}  {'Closed':>7s}")
    print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*7}")
    for domain in domains:
        e = existing_results[domain]
        print(f"  {domain:>12s}  {e['accuracy']*100:>5.1f}%  "
              f"{e['balanced_accuracy']*100:>5.1f}%  "
              f"{e['regret']['gap_closed_pct']:>6.1f}%")

    # ── Save results ─────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = dict(
        timestamp=datetime.now().isoformat(),
        description="Leave-One-Domain-Out: train on 4 domains, test on held-out domain",
        domains=domains,
        holdout_results=holdout_results,
        existing_model_per_domain=existing_results,
        weighted_avg=dict(
            accuracy=round(w_acc, 4),
            balanced_accuracy=round(w_bal, 4),
            gap_closed_pct=round(w_closed, 2),
            perfect_pick_pct=round(w_perf, 2),
        ),
    )
    out_file = OUT_DIR / "domain_holdout_results.json"
    out_file.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved: {out_file.relative_to(ROOT)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
