#!/usr/bin/env python3
"""
F1-only XGBoost v5 Stratified K-Fold Cross Validation
====================================================

Implements the K-fold flow described in ee.txt:
  1. load one labeled table,
  2. shuffle/split rows into K folds,
  3. for each fold train a fresh model on K-1 folds,
  4. test on the held-out fold,
  5. save each fold result,
  6. average metrics and compute standard deviation.

Default input:
  /Volumes/k/thesis_data/f1_only/training_dataset.csv

Outputs:
  results/xgboost_v5_f1_kfold/
    fold_1/metrics.json, predictions.csv, confusion_matrix.csv
    ...
    summary.json
    summary.csv
    figures/*.png
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

try:
    from rich import box
    from rich.align import Align
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except Exception:  # pragma: no cover - fallback for minimal environments
    box = Align = Console = Panel = Table = Text = None

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
DATA_CSV = Path("/Volumes/k/thesis_data/f1_only/training_dataset.csv")
RESULTS_DIR = ROOT / "results" / "xgboost_v5_f1_kfold"

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


class StoryPrinter:
    def __init__(self, enabled: bool, plain: bool = False):
        self.enabled = enabled
        self.use_rich = enabled and not plain and Console is not None
        self.width = min(110, shutil.get_terminal_size((100, 24)).columns)
        self.console = Console(width=self.width) if self.use_rich else None

    def center(self, text: str = "") -> None:
        if not self.enabled:
            return
        print(text.center(self.width))

    def banner(self, title: str, subtitle: str = "") -> None:
        if not self.enabled:
            return
        if self.use_rich:
            body = Text(title, style="bold white")
            if subtitle:
                body.append("\n" + subtitle, style="orange3")
            self.console.print(Align.center(Panel(body, box=box.DOUBLE, border_style="orange3", width=min(90, self.width - 4))))
        else:
            self.center("═" * min(78, self.width))
            self.center(title)
            if subtitle:
                self.center(subtitle)
            self.center("═" * min(78, self.width))

    def panel(self, title: str, lines: list[str], style: str = "cyan") -> None:
        if not self.enabled:
            return
        if self.use_rich:
            self.console.print(
                Align.center(
                    Panel(
                        "\n".join(lines),
                        title=title,
                        title_align="center",
                        box=box.ROUNDED,
                        border_style=style,
                        width=min(90, self.width - 4),
                    )
                )
            )
        else:
            self.center(f"[ {title} ]")
            for line in lines:
                self.center(line)

    def fold_map(self, fold_id: int, folds: int) -> None:
        if not self.enabled:
            return
        if self.use_rich:
            table = Table(box=box.SIMPLE_HEAVY, show_header=False, pad_edge=True)
            for i in range(1, folds + 1):
                table.add_column(justify="center", width=14)
            row = []
            for i in range(1, folds + 1):
                if i == fold_id:
                    row.append(f"[bold white on red] F{i} TEST [/]")
                else:
                    row.append(f"[green]F{i} train[/]")
            table.add_row(*row)
            self.console.print(Align.center(table))
        else:
            fold_map = []
            for i in range(1, folds + 1):
                fold_map.append(f"[F{i}: TEST]" if i == fold_id else f"[F{i}: train]")
            self.center("  ".join(fold_map))

    def metrics_table(self, metrics: dict, regret: dict) -> None:
        if not self.enabled:
            return
        rows = [
            ("Accuracy", f"{metrics['accuracy']*100:.2f}%"),
            ("Balanced accuracy", f"{metrics['balanced_accuracy']*100:.2f}%"),
            ("Macro F1", f"{metrics['macro_f1']*100:.2f}%"),
            ("Weighted F1", f"{metrics['weighted_f1']*100:.2f}%"),
            ("Gap closed", f"{regret['gap_closed_pct']:.2f}%"),
            ("Zero regret", f"{regret['zero_regret_pct']:.2f}%"),
        ]
        if self.use_rich:
            table = Table(title="Fold score", box=box.ROUNDED, border_style="orange3", width=52)
            table.add_column("Metric", style="bold white")
            table.add_column("Value", justify="right", style="bold orange3")
            for k, v in rows:
                table.add_row(k, v)
            self.console.print(Align.center(table))
        else:
            for k, v in rows:
                self.center(f"{k}: {v}")

    def summary_table(self, aggregate: dict) -> None:
        if not self.enabled:
            return
        rows = [
            ("Accuracy", aggregate["accuracy"]["mean"] * 100, aggregate["accuracy"]["std"] * 100),
            ("Balanced accuracy", aggregate["balanced_accuracy"]["mean"] * 100, aggregate["balanced_accuracy"]["std"] * 100),
            ("Macro F1", aggregate["macro_f1"]["mean"] * 100, aggregate["macro_f1"]["std"] * 100),
            ("Weighted F1", aggregate["weighted_f1"]["mean"] * 100, aggregate["weighted_f1"]["std"] * 100),
            ("Gap closed", aggregate["gap_closed_pct"]["mean"], aggregate["gap_closed_pct"]["std"]),
            ("Zero regret", aggregate["zero_regret_pct"]["mean"], aggregate["zero_regret_pct"]["std"]),
        ]
        if self.use_rich:
            table = Table(title="Final mean ± standard deviation", box=box.DOUBLE_EDGE, border_style="green", width=70)
            table.add_column("Metric", style="bold white")
            table.add_column("Mean", justify="right", style="bold green")
            table.add_column("Std", justify="right", style="orange3")
            for name, mean, std in rows:
                table.add_row(name, f"{mean:.2f}%", f"± {std:.2f}%")
            self.console.print(Align.center(table))
        else:
            for name, mean, std in rows:
                self.center(f"{name}: {mean:.2f} ± {std:.2f}%")


def balanced_undersample(
    df: pd.DataFrame,
    label_col: str,
    max_ratio: float = 3.0,
    random_state: int = SEED,
) -> pd.DataFrame:
    counts = df[label_col].value_counts()
    min_count = int(counts.min())
    cap = int(min_count * max_ratio)
    parts = []
    for cls, subset in df.groupby(label_col):
        n = min(len(subset), cap)
        parts.append(subset.sample(n=n, random_state=random_state))
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[yi] for yi in y], dtype=np.float64)


def apply_v5_filter(df: pd.DataFrame) -> pd.DataFrame:
    time_cols = df[["time_introsort", "time_heapsort", "time_timsort"]].values
    sorted_times = np.sort(time_cols, axis=1)
    margin = (sorted_times[:, 1] - sorted_times[:, 0]) / (sorted_times[:, 1] + 1e-15)
    keep = (margin >= 0.05) | (df["n_elements"].values >= 2000)
    return df[keep].reset_index(drop=True)


def regret_metrics(y_pred: np.ndarray, time_df: pd.DataFrame) -> dict:
    times = time_df[["time_introsort", "time_heapsort", "time_timsort"]].values
    algo_idx = {"introsort": 0, "heapsort": 1, "timsort": 2}

    vbs_total = float(times.min(axis=1).sum())
    model_total = float(sum(times[i, algo_idx[p]] for i, p in enumerate(y_pred)))
    sbs_totals = {a: float(times[:, j].sum()) for a, j in algo_idx.items()}
    sbs_algorithm = min(sbs_totals, key=sbs_totals.get)
    sbs_total = sbs_totals[sbs_algorithm]

    gap_den = sbs_total - vbs_total
    gap_closed = 100.0 if gap_den <= 0 else (1 - (model_total - vbs_total) / gap_den) * 100
    per_regret = np.array([times[i, algo_idx[p]] - times[i].min() for i, p in enumerate(y_pred)])

    return {
        "vbs_total_s": round(vbs_total, 6),
        "sbs_total_s": round(sbs_total, 6),
        "sbs_algorithm": sbs_algorithm,
        "model_total_s": round(model_total, 6),
        "gap_closed_pct": round(float(gap_closed), 2),
        "zero_regret_pct": round(float((per_regret == 0).mean() * 100), 2),
        "mean_regret_us": round(float(per_regret.mean() * 1e6), 3),
        "p95_regret_us": round(float(np.percentile(per_regret, 95) * 1e6), 3),
    }


def save_confusion_matrix_csv(cm: np.ndarray, out_path: Path) -> None:
    pd.DataFrame(cm, index=ALGORITHMS, columns=ALGORITHMS).to_csv(out_path)


def plot_metric_bars(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
        ("gap_closed_pct", "Gap closed"),
        ("zero_regret_pct", "Zero regret"),
    ]

    for col, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(summary_df["fold"].astype(str), summary_df[col] * (100 if col.endswith("f1") or "accuracy" in col else 1), color="#F4511E")
        ylabel = "%" if col != "mean_regret_us" else "microseconds"
        ax.set_title(f"F1 KFold {title} by fold")
        ax.set_xlabel("Fold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{col}_by_fold.png", dpi=220)
        plt.close(fig)

    agg_cols = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
    means = [summary_df[c].mean() * 100 for c in agg_cols]
    stds = [summary_df[c].std(ddof=1) * 100 for c in agg_cols]
    labels = ["Accuracy", "Balanced acc.", "Macro F1", "Weighted F1"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, means, yerr=stds, color=["#F4511E", "#8F1D3A", "#53657D", "#F4511E"], capsize=5)
    ax.set_title("F1 KFold mean score with standard deviation")
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "mean_std_core_metrics.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_CSV)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--sample", type=int, default=50_000)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--filter-mode",
        choices=["none", "v5"],
        default="none",
        help="Use no label filter, or apply the original v5 near-tie filter before KFold.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["balanced", "random"],
        default="balanced",
        help="Balanced samples preserve minority labels; random samples preserve natural F1 imbalance.",
    )
    parser.add_argument(
        "--story",
        action="store_true",
        help="Print a live ASCII explanation of the KFold process while running.",
    )
    parser.add_argument(
        "--plain-story",
        action="store_true",
        help="Disable rich colors/panels and print plain centered story text.",
    )
    args = parser.parse_args()
    story = StoryPrinter(args.story, args.plain_story)

    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out / "figures"

    print("=" * 78)
    print("  F1-ONLY XGBOOST v5 STRATIFIED K-FOLD CROSS VALIDATION")
    print("=" * 78)
    print(f"  Data:   {args.data}")
    print(f"  Output: {args.out}")
    print(f"  Folds:  {args.folds}")
    print(f"  Sample: {args.sample:,}")
    print(f"  Filter: {args.filter_mode}")
    print(f"  Sample mode: {args.sample_mode}")
    print(f"  Story mode: {'on' if args.story else 'off'}")

    df = pd.read_csv(args.data)
    missing = sorted(
        (set(FEATURE_NAMES) | {"best_algorithm", "n_elements", "time_introsort", "time_heapsort", "time_timsort"})
        - set(df.columns)
    )
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df[df["best_algorithm"].isin(ALGORITHMS)].reset_index(drop=True)
    print(f"\n[1] Loaded F1 rows: {len(df):,}")
    print(df["best_algorithm"].value_counts().to_string())

    if args.filter_mode == "v5":
        df = apply_v5_filter(df)
        step_label = "After v5 near-tie filter"
    else:
        step_label = "No near-tie filter applied"
    print(f"\n[2] {step_label}: {len(df):,}")
    print(df["best_algorithm"].value_counts().to_string())

    if args.sample > 0 and len(df) > args.sample and args.sample_mode == "random":
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
    elif args.sample > 0 and args.sample_mode == "balanced":
        counts = df["best_algorithm"].value_counts()
        per_class_target = max(1, args.sample // len(ALGORITHMS))
        per_class = min(per_class_target, int(counts.min()))
        if per_class * len(ALGORITHMS) < args.sample:
            print(
                f"  Balanced sample capped at {per_class:,} per class "
                f"because minority class has {int(counts.min()):,} rows."
            )
        parts = [
            df[df["best_algorithm"] == algo].sample(
                n=per_class, random_state=args.seed + i
            )
            for i, algo in enumerate(ALGORITHMS)
        ]
        df = (
            pd.concat(parts, ignore_index=True)
            .sample(frac=1.0, random_state=args.seed)
            .reset_index(drop=True)
        )
    else:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    print(f"\n[3] KFold table after sampling/shuffle: {len(df):,}")
    print(df["best_algorithm"].value_counts().to_string())

    if args.story:
        story.banner(
            "LIVE K-FOLD STORY",
            "one table -> five folds -> five fresh models -> mean ± standard deviation",
        )
        story.panel(
            "Full KFold table",
            [
                f"rows = {len(df):,}",
                "labels = introsort / heapsort / timsort",
                "features = 16 structural features",
                "target = empirical fastest sorting algorithm",
            ],
            style="cyan",
        )

    le = LabelEncoder().fit(ALGORITHMS)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_rows = []

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(df[FEATURE_NAMES], df["best_algorithm"]), 1):
        fold_start = time.time()
        fold_dir = args.out / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        train_bal = balanced_undersample(train_df, "best_algorithm", random_state=args.seed + fold_id)

        X_train = train_bal[FEATURE_NAMES].values
        y_train = train_bal["best_algorithm"].values
        y_train_enc = le.transform(y_train)
        X_test = test_df[FEATURE_NAMES].values
        y_test = test_df["best_algorithm"].values

        if args.story:
            story.banner(f"ROUND {fold_id}/{args.folds}", f"fold {fold_id} becomes hidden test data")
            story.fold_map(fold_id, args.folds)
            story.panel(
                "Derived tables",
                [
                    f"TRAIN = {len(train_df):,} raw rows -> {len(train_bal):,} balanced rows",
                    f"TEST  = {len(test_df):,} hidden rows",
                    "SOURCE CSV stays untouched",
                ],
                style="cyan",
            )
            story.panel(
                "Fresh model",
                [
                    "XGBoostClassifier starts from zero",
                    "model sees TRAIN folds only",
                    "500 boosted trees learn corrections",
                    "after training, model predicts TEST fold",
                ],
                style="orange3",
            )

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.set_params(random_state=args.seed + fold_id)
        model.fit(X_train, y_train_enc, sample_weight=compute_sample_weights(y_train_enc), verbose=False)

        y_pred = le.inverse_transform(model.predict(X_test))
        cm = confusion_matrix(y_test, y_pred, labels=ALGORITHMS)
        report = classification_report(y_test, y_pred, labels=ALGORITHMS, output_dict=True, zero_division=0)
        regret = regret_metrics(y_pred, test_df)

        metrics = {
            "fold": fold_id,
            "train_size_raw": int(len(train_df)),
            "train_size_balanced": int(len(train_bal)),
            "test_size": int(len(test_df)),
            "train_class_counts_raw": {k: int(v) for k, v in train_df["best_algorithm"].value_counts().to_dict().items()},
            "train_class_counts_balanced": {k: int(v) for k, v in train_bal["best_algorithm"].value_counts().to_dict().items()},
            "test_class_counts": {k: int(v) for k, v in test_df["best_algorithm"].value_counts().to_dict().items()},
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, labels=ALGORITHMS, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, y_pred, labels=ALGORITHMS, average="weighted", zero_division=0)),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "regret": regret,
            "elapsed_s": round(time.time() - fold_start, 2),
        }

        pred_df = test_df[["file", "n_elements", "best_algorithm", "time_introsort", "time_heapsort", "time_timsort"]].copy()
        pred_df["predicted_algorithm"] = y_pred
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)
        save_confusion_matrix_csv(cm, fold_dir / "confusion_matrix.csv")
        (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        fold_rows.append(
            {
                "fold": fold_id,
                "train_size_balanced": metrics["train_size_balanced"],
                "test_size": metrics["test_size"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "gap_closed_pct": regret["gap_closed_pct"],
                "zero_regret_pct": regret["zero_regret_pct"],
                "mean_regret_us": regret["mean_regret_us"],
                "elapsed_s": metrics["elapsed_s"],
            }
        )

        print(f"\n[FOLD {fold_id}/{args.folds}]")
        print(f"  train raw:      {len(train_df):,}")
        print(f"  train balanced: {len(train_bal):,}")
        print(f"  test:           {len(test_df):,}")
        print(f"  accuracy:       {metrics['accuracy']*100:6.2f}%")
        print(f"  balanced acc:   {metrics['balanced_accuracy']*100:6.2f}%")
        print(f"  macro F1:       {metrics['macro_f1']*100:6.2f}%")
        print(f"  weighted F1:    {metrics['weighted_f1']*100:6.2f}%")
        print(f"  gap closed:     {regret['gap_closed_pct']:6.2f}%")
        print(f"  zero regret:    {regret['zero_regret_pct']:6.2f}%")
        print(f"  elapsed:        {metrics['elapsed_s']:.1f}s")

        if args.story:
            story.metrics_table(metrics, regret)
            story.panel(
                "Round close",
                [
                    "score saved to fold folder",
                    "temporary fold model discarded",
                    "next round moves TEST role to next fold",
                ],
                style="green",
            )

    summary_df = pd.DataFrame(fold_rows)
    summary_df.to_csv(args.out / "summary.csv", index=False)
    plot_metric_bars(summary_df, figures_dir)

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "gap_closed_pct",
        "zero_regret_pct",
        "mean_regret_us",
    ]
    aggregate = {
        col: {
            "mean": float(summary_df[col].mean()),
            "std": float(summary_df[col].std(ddof=1)),
            "min": float(summary_df[col].min()),
            "max": float(summary_df[col].max()),
        }
        for col in metric_cols
    }
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data": str(args.data),
        "sample_rows": int(len(df)),
        "folds": int(args.folds),
        "seed": int(args.seed),
        "xgb_params": XGB_PARAMS,
        "folds_table": fold_rows,
        "aggregate": aggregate,
        "figures": sorted(str(p.relative_to(args.out)) for p in figures_dir.glob("*.png")),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 78)
    print("  K-FOLD SUMMARY")
    print("=" * 78)
    if args.story:
        story.banner("K-FOLD COMPLETE", "five temporary models -> five scores -> average + deviation")
        story.panel(
            "Meaning",
            [
                "Mean = central performance estimate",
                "Std = how much fold scores move",
                "small Std = stable row-level split behavior",
            ],
            style="green",
        )
        story.summary_table(aggregate)
    for col, label, pct in [
        ("accuracy", "Accuracy", True),
        ("balanced_accuracy", "Balanced accuracy", True),
        ("macro_f1", "Macro F1", True),
        ("weighted_f1", "Weighted F1", True),
        ("gap_closed_pct", "Gap closed", False),
        ("zero_regret_pct", "Zero regret", False),
    ]:
        mean = aggregate[col]["mean"] * (100 if pct else 1)
        std = aggregate[col]["std"] * (100 if pct else 1)
        print(f"  {label:18s}: {mean:6.2f} ± {std:.2f}%")
    print(f"  Total time: {(time.time() - t0)/60:.1f} min")
    print(f"  Saved: {args.out}")


if __name__ == "__main__":
    main()
