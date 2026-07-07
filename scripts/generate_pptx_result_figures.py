#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pptx_real_results_figures_clean"

MAIN = "#df982d"
ACCENT = "#c89116"
BLACK = "#000000"
MAROON = "#9B173D"
WHITE = "#FFFFFF"
GRID = "#E8ECEF"
LIGHT = "#FBF7EF"


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def init() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("*.png"):
        p.unlink()
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=260, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def style_table(ax) -> None:
    ax.axis("off")
    ax.set_facecolor(WHITE)


def draw_benchmark_table(df: pd.DataFrame, name: str, highlight_cols: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    style_table(ax)
    display = df.copy()
    cell_text = display.values.tolist()
    col_labels = list(display.columns)
    row_labels = list(display.index)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        loc="center",
        cellLoc="center",
        rowLoc="center",
        bbox=[0.02, 0.05, 0.96, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(19)
    table.scale(1, 2.1)
    highlight_cols = highlight_cols or []
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(WHITE)
        cell.set_linewidth(2.0)
        txt = cell.get_text()
        txt.set_color(BLACK)
        if r == 0:
            cell.set_facecolor(BLACK)
            txt.set_color(WHITE)
            txt.set_weight("bold")
            txt.set_fontsize(20)
        elif c == -1:
            cell.set_facecolor(MAIN)
            txt.set_color(WHITE)
            txt.set_weight("bold")
            txt.set_fontsize(20)
        else:
            col_name = col_labels[c] if c >= 0 and c < len(col_labels) else ""
            cell.set_facecolor("#F3D7A3" if col_name in highlight_cols else LIGHT)
            txt.set_weight("bold" if col_name in highlight_cols else "normal")
    save(fig, name)


def v5_benchmark_table(v5: dict) -> None:
    rows = {}
    for split, label in [("train", "Train"), ("val", "Validation"), ("test", "Test")]:
        res = v5["results"][split]
        rep = res["classification_report"]
        rows[label] = {
            "Accuracy": f"{res['accuracy']*100:.1f}",
            "Balanced Acc.": f"{res['balanced_accuracy']*100:.1f}",
            "Macro F1": f"{rep['macro avg']['f1-score']*100:.1f}",
            "Weighted F1": f"{rep['weighted avg']['f1-score']*100:.1f}",
        }
    draw_benchmark_table(pd.DataFrame(rows).T, "01_v5_benchmark_table.png", ["Accuracy", "Weighted F1"])


def v5_class_report_matrix(v5: dict) -> None:
    rep = v5["results"]["test"]["classification_report"]
    rows = []
    for cls in ["introsort", "heapsort", "timsort"]:
        rows.append(
            {
                "Class": cls.title(),
                "Precision": rep[cls]["precision"] * 100,
                "Recall": rep[cls]["recall"] * 100,
                "F1": rep[cls]["f1-score"] * 100,
                "Support": int(rep[cls]["support"]),
            }
        )
    df = pd.DataFrame(rows).set_index("Class")

    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    style_table(ax)
    text_df = df.copy()
    for c in ["Precision", "Recall", "F1"]:
        text_df[c] = text_df[c].map(lambda x: f"{x:.1f}")
    text_df["Support"] = text_df["Support"].map(lambda x: f"{x:,}")
    table = ax.table(
        cellText=text_df.values.tolist(),
        colLabels=list(text_df.columns),
        rowLabels=list(text_df.index),
        loc="center",
        cellLoc="center",
        rowLoc="center",
        bbox=[0.02, 0.05, 0.96, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(1, 2.15)
    vals = df[["Precision", "Recall", "F1"]].to_numpy()
    lo, hi = 30, 100
    cmap = sns.light_palette(MAIN, as_cmap=True)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(WHITE)
        cell.set_linewidth(2.0)
        txt = cell.get_text()
        txt.set_color(BLACK)
        if r == 0:
            cell.set_facecolor(BLACK)
            txt.set_color(WHITE)
            txt.set_weight("bold")
        elif c == -1:
            cell.set_facecolor(MAIN)
            txt.set_color(WHITE)
            txt.set_weight("bold")
        elif c in [0, 1, 2] and r >= 1:
            norm = max(0, min(1, (vals[r - 1, c] - lo) / (hi - lo)))
            cell.set_facecolor(cmap(norm))
            txt.set_weight("bold")
        else:
            cell.set_facecolor(LIGHT)
    save(fig, "02_v5_classification_report_table.png")


def v5_confusion_matrix(v5: dict) -> None:
    res = v5["results"]["test"]
    cm = np.array(res["confusion_matrix"], dtype=float)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    ann = np.empty_like(cm, dtype=object)
    for i in range(3):
        for j in range(3):
            ann[i, j] = f"{int(cm[i,j]):,}\n{pct[i,j]:.1f}%"
    labels = ["Introsort", "Heapsort", "Timsort"]
    fig, ax = plt.subplots(figsize=(9.8, 7.6))
    sns.heatmap(
        pct,
        annot=ann,
        fmt="",
        cmap=sns.light_palette(MAIN, as_cmap=True),
        cbar=False,
        linewidths=2.2,
        linecolor=WHITE,
        vmin=0,
        vmax=100,
        annot_kws={"fontsize": 18, "weight": "bold", "color": BLACK},
        ax=ax,
    )
    ax.set_xticklabels(labels, fontsize=18, weight="bold")
    ax.set_yticklabels(labels, fontsize=18, rotation=0, weight="bold")
    ax.set_xlabel("Predicted algorithm", fontsize=20, weight="bold", labelpad=14)
    ax.set_ylabel("True fastest algorithm", fontsize=20, weight="bold", labelpad=14)
    save(fig, "03_v5_confusion_matrix.png")


def f1_kfold_table(kfold: dict) -> None:
    agg = kfold["aggregate"]
    rows = {
        "Accuracy": agg["accuracy"],
        "Balanced Acc.": agg["balanced_accuracy"],
        "Macro F1": agg["macro_f1"],
        "Weighted F1": agg["weighted_f1"],
        "Gap closed": agg["gap_closed_pct"],
        "Zero regret": agg["zero_regret_pct"],
    }
    data = {}
    for k, v in rows.items():
        mul = 100 if k in ["Accuracy", "Balanced Acc.", "Macro F1", "Weighted F1"] else 1
        data[k] = {
            "Mean": f"{v['mean']*mul:.1f}",
            "Std": f"{v['std']*mul:.1f}",
            "Min": f"{v['min']*mul:.1f}",
            "Max": f"{v['max']*mul:.1f}",
        }
    draw_benchmark_table(pd.DataFrame(data).T, "04_f1_kfold_benchmark_table.png", ["Mean", "Std"])


def runtime_gap(regret: dict) -> None:
    labels = ["VBS\noracle", "Model v5\nselector", "SBS\nsingle best"]
    vals = [regret["vbs_total_s"], regret["model_total_s"], regret["sbs_total_s"]]
    colors = [BLACK, MAIN, MAROON]
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    bars = ax.bar(np.arange(3), vals, color=colors, width=0.52)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.35, f"{v:.3f}s", ha="center", fontsize=20, weight="bold")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=18, weight="bold")
    ax.set_ylabel("Total runtime (seconds)", fontsize=20, weight="bold")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", color=GRID, linewidth=1.2)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=16)
    ax.text(
        1.55,
        max(vals) * 1.08,
        f"Gap closed: {regret['gap_closed_pct']:.1f}%\nZero regret: {regret['per_instance_regret_us']['zero_pct']:.1f}%",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc=LIGHT, ec=ACCENT, lw=2),
    )
    save(fig, "05_runtime_gap_benchmark.png")


def main() -> None:
    init()
    v5 = load("results/xgboost_v5/evaluation_results.json")
    regret = load("results/xgboost_v5/regret_analysis.json")
    kfold = load("results/xgboost_v5_f1_kfold/summary.json")
    v5_benchmark_table(v5)
    v5_class_report_matrix(v5)
    v5_confusion_matrix(v5)
    f1_kfold_table(kfold)
    runtime_gap(regret)
    (OUT / "00_sources.txt").write_text(
        "results/xgboost_v5/evaluation_results.json\n"
        "results/xgboost_v5/regret_analysis.json\n"
        "results/xgboost_v5_f1_kfold/summary.json\n"
    )
    print(OUT)


if __name__ == "__main__":
    main()
