#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "f1_dynamic_figures"
OUT.mkdir(parents=True, exist_ok=True)

STRICT_JSON = ROOT / "results" / "f1_9_channel_models_dynamic_v2_strict" / "strict_router_eval.json"
MANIFEST_JSON = ROOT / "results" / "f1_9_channel_models_dynamic_v2" / "manifest.json"


def fig_01_strict_metrics(strict: dict) -> Path:
    labels = ["routed", "channel_sbs", "global_sbs"]
    acc = [strict["metrics"][k]["accuracy"] for k in labels]
    bacc = [strict["metrics"][k]["balanced_accuracy"] for k in labels]
    x = np.arange(len(labels))
    w = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - w / 2, acc, w, label="Accuracy")
    plt.bar(x + w / 2, bacc, w, label="Balanced Accuracy")
    plt.ylim(0, 1.0)
    plt.xticks(x, labels)
    plt.title("Strict Router Evaluation Metrics")
    plt.legend()
    for i, v in enumerate(acc):
        plt.text(i - w / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    for i, v in enumerate(bacc):
        plt.text(i + w / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    p = OUT / "01_strict_metrics.png"
    plt.tight_layout()
    plt.savefig(p, dpi=220)
    plt.close()
    return p


def fig_02_channel_balacc(manifest: dict) -> Path:
    rows = []
    for ch, d in manifest.get("channels", {}).items():
        if d.get("status") == "trained":
            rows.append((ch, d["test_balanced_accuracy"]))
    rows.sort(key=lambda x: x[0])
    ch = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    plt.figure(figsize=(11, 5))
    plt.bar(ch, vals)
    plt.ylim(0, 1.0)
    plt.title("Per-Channel Balanced Accuracy (Dynamic V2)")
    plt.ylabel("Balanced Accuracy")
    plt.xticks(rotation=20)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    p = OUT / "02_channel_balanced_accuracy.png"
    plt.tight_layout()
    plt.savefig(p, dpi=220)
    plt.close()
    return p


def fig_03_pipeline() -> Path:
    plt.figure(figsize=(13, 4.5))
    plt.axis("off")
    boxes = [
        (0.02, "FastF1 Fetch\nscripts/fetch_f1_1m.py"),
        (0.24, "Raw Store\nraw_arrays.h5 + index.csv"),
        (0.46, "Feature Build\ntraining_dataset.csv"),
        (0.68, "Relabel V2\nactual_5 + margin filter"),
        (0.88, "Strict Train/Eval\nrouter vs baselines"),
    ]
    for x, txt in boxes:
        plt.gca().add_patch(plt.Rectangle((x, 0.35), 0.18, 0.3, fill=False, lw=2))
        plt.text(x + 0.09, 0.5, txt, ha="center", va="center", fontsize=10)
    for x in [0.20, 0.42, 0.64, 0.86]:
        plt.arrow(x, 0.5, 0.03, 0, head_width=0.03, head_length=0.01, length_includes_head=True)
    plt.title("Pipeline: Input -> Processing -> Output", fontsize=14, pad=12)
    p = OUT / "03_pipeline.png"
    plt.tight_layout()
    plt.savefig(p, dpi=220)
    plt.close()
    return p


def fig_04_data_sizes(strict: dict) -> Path:
    n_total = strict["split"]["n_total"]
    n_train = strict["split"]["n_train"]
    n_test = strict["split"]["n_test"]
    vals = [n_total, n_train, n_test]
    labels = ["Total", "Train", "Test"]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, vals)
    plt.title("Strict Split Sizes")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.015, f"{v:,}", ha="center")
    p = OUT / "04_strict_split_sizes.png"
    plt.tight_layout()
    plt.savefig(p, dpi=220)
    plt.close()
    return p


def main() -> None:
    strict = json.loads(STRICT_JSON.read_text())
    manifest = json.loads(MANIFEST_JSON.read_text()) if MANIFEST_JSON.exists() else {"channels": {}}
    paths = [
        fig_01_strict_metrics(strict),
        fig_02_channel_balacc(manifest),
        fig_03_pipeline(),
        fig_04_data_sizes(strict),
    ]
    summary = {"figures": [str(p) for p in paths]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved figures:")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()

