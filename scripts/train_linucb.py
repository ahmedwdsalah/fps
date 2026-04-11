#!/usr/bin/env python3
"""
Step 5: LinUCB Contextual Bandit — TRUE Online Adaptive Selector
=================================================================
The MAIN THESIS CONTRIBUTION.

A LinUCB contextual bandit that adapts online by ACTUALLY SORTING
arrays and observing real wall-clock timings — not replaying cached data.

Pipeline (per array, at runtime):
    1. Load raw array
    2. Extract 16 O(n) features
    3. LinUCB picks an algorithm (with exploration bonus)
    4. ACTUALLY SORT the array with the chosen algorithm
    5. Measure real wall-clock time (perf_counter, gc disabled, best-of-N)
    6. Feed reward = −time back to LinUCB
    7. For evaluation only: also time the other 2 algorithms (oracle comparison)

Warm Start:
    Initialise A_a, b_a from training_dataset.csv timings (pre-recorded).
    This gives Tier 1 knowledge without re-sorting all training arrays.

Experiments:
    1. In-distribution: warm vs cold vs v5-frozen vs SBS on test arrays
    2. Domain shift: per-domain breakdown
    3. α sensitivity sweep

Inputs:  data/training_dataset.csv       (warm-start + array paths)
         data/real_world_10k/raw/        (actual array files for real sorting)
         models/xgboost_v5/xgb_v5.json   (v5 frozen baseline)
Outputs: models/linucb/                  (bandit state)
         results/linucb/                 (metrics + figures)

Usage:
    python3 scripts/train_linucb.py                # full run
    python3 scripts/train_linucb.py --limit 5000   # quick test on 5K arrays
    python3 scripts/train_linucb.py --skip-sort     # fallback: use cached timings
                                                     (if external drive unavailable)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time as time_module
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
DATA_CSV    = ROOT / "data" / "training_dataset.csv"
RAW_DIR     = ROOT / "data" / "real_world_10k" / "raw"
V5_MODEL    = ROOT / "models" / "xgboost_v5" / "xgb_v5.json"
MODEL_DIR   = ROOT / "models" / "linucb"
RESULTS_DIR = ROOT / "results" / "linucb"
FIGURES_DIR = RESULTS_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import extract_features, FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
SORT_KINDS = ["quicksort", "heapsort", "stable"]  # numpy kind= for each
TIME_COLS  = ["time_introsort", "time_heapsort", "time_timsort"]
SEED = 42
N_ARMS = 3
D = len(FEATURE_NAMES)  # 16

# Timing config (same as build_training_dataset.py)
TIMING_REPEATS_SMALL = 5   # n <= 10K
TIMING_REPEATS_LARGE = 3   # n > 10K


# ── Real sorting + timing ────────────────────────────────────────────────

def time_sort(arr: np.ndarray, kind: str, repeats: int) -> float:
    """Time a numpy sort. Returns best-of-N time in seconds."""
    best = float("inf")
    for _ in range(repeats):
        copy = arr.copy()
        gc.disable()
        t0 = time_module.perf_counter()
        np.sort(copy, kind=kind)
        t1 = time_module.perf_counter()
        gc.enable()
        best = min(best, t1 - t0)
    return best


def time_all_algorithms(arr: np.ndarray) -> np.ndarray:
    """Time all 3 algorithms on an array. Returns [t_intro, t_heap, t_tim]."""
    n = len(arr)
    repeats = TIMING_REPEATS_SMALL if n <= 10_000 else TIMING_REPEATS_LARGE
    return np.array([time_sort(arr, kind, repeats) for kind in SORT_KINDS])


def load_array(filepath: str) -> np.ndarray | None:
    """Load a raw array CSV file."""
    path = RAW_DIR / filepath
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, header=None).values.ravel().astype(np.float64)
    except Exception:
        return None


# ── LinUCB Implementation ────────────────────────────────────────────────

class LinUCB:
    """
    LinUCB contextual bandit with disjoint linear models per arm.
    Uses Sherman-Morrison incremental inverse for O(d²) updates.
    """

    def __init__(self, d: int, n_arms: int, alpha: float = 1.0,
                 lambda_reg: float = 1.0):
        self.d = d
        self.n_arms = n_arms
        self.alpha = alpha
        self.lambda_reg = lambda_reg

        # Per-arm matrices
        self.A = [lambda_reg * np.eye(d) for _ in range(n_arms)]
        self.A_inv = [(1.0 / lambda_reg) * np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]

        # Tracking
        self.arm_pulls = np.zeros(n_arms, dtype=int)
        self.total_pulls = 0

    def select_arm(self, x: np.ndarray) -> tuple[int, np.ndarray]:
        """Select arm with highest UCB. Returns (chosen_arm, ucb_values)."""
        ucb_values = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            theta_a = self.A_inv[a] @ self.b[a]
            exploitation = theta_a @ x
            exploration = self.alpha * np.sqrt(x @ self.A_inv[a] @ x)
            ucb_values[a] = exploitation + exploration
        return int(np.argmax(ucb_values)), ucb_values

    def update(self, arm: int, x: np.ndarray, reward: float):
        """Update after observing reward. Sherman-Morrison O(d²) inverse."""
        Ainv_x = self.A_inv[arm] @ x
        denom = 1.0 + x @ Ainv_x
        self.A_inv[arm] -= np.outer(Ainv_x, Ainv_x) / denom

        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.arm_pulls[arm] += 1
        self.total_pulls += 1

    def warm_start(self, X: np.ndarray, times: np.ndarray):
        """
        Initialise from pre-recorded training data with FULL information.
        All 3 arm timings available → update all arms per sample.
        """
        n = len(X)
        print(f"  Warm-starting on {n:,} arrays (all 3 arms)...")

        for i in range(n):
            x = X[i]
            for arm in range(self.n_arms):
                reward = -times[i, arm] * 1e6  # negative µs
                self.A[arm] += np.outer(x, x)
                self.b[arm] += reward * x
                self.arm_pulls[arm] += 1

        # Recompute inverse from scratch (numerically stable after bulk load)
        for arm in range(self.n_arms):
            self.A_inv[arm] = np.linalg.inv(self.A[arm])

        self.total_pulls = n * self.n_arms
        print(f"  Done. Pulls per arm: {self.arm_pulls.tolist()}")

    def save(self, path: Path):
        state = {
            "d": self.d, "n_arms": self.n_arms,
            "alpha": self.alpha, "lambda_reg": self.lambda_reg,
            "A": [a.tolist() for a in self.A],
            "A_inv": [a.tolist() for a in self.A_inv],
            "b": [b.tolist() for b in self.b],
            "arm_pulls": self.arm_pulls.tolist(),
            "total_pulls": self.total_pulls,
        }
        with open(path, "w") as f:
            json.dump(state, f)


# ── Regret computation ───────────────────────────────────────────────────

def compute_regret_metrics(picks, times):
    n = len(picks)
    best_times = times.min(axis=1)
    model_times = np.array([times[i, picks[i]] for i in range(n)])
    regret_us = (model_times - best_times) * 1e6

    vbs_total = best_times.sum()
    sbs_idx = np.argmin(times.sum(axis=0))
    sbs_total = times[:, sbs_idx].sum()
    model_total = model_times.sum()

    gap_closed = 1.0 - (model_total - vbs_total) / (sbs_total - vbs_total + 1e-15)

    return {
        "gap_closed_pct": round(float(gap_closed * 100), 4),
        "perfect_picks_pct": round(float((regret_us < 1e-9).mean() * 100), 4),
        "mean_regret_us": round(float(regret_us.mean()), 4),
        "median_regret_us": round(float(np.median(regret_us)), 4),
        "p95_regret_us": round(float(np.percentile(regret_us, 95)), 4),
        "p99_regret_us": round(float(np.percentile(regret_us, 99)), 4),
        "max_regret_us": round(float(regret_us.max()), 4),
        "model_total_s": round(float(model_total), 6),
        "vbs_total_s": round(float(vbs_total), 6),
        "sbs_total_s": round(float(sbs_total), 6),
    }


# ── Online simulation (TRUE sorting) ────────────────────────────────────

def run_online_experiment(bandit: LinUCB, df_stream: pd.DataFrame,
                          use_real_sort: bool = True,
                          window: int = 500) -> dict:
    """
    TRUE online bandit loop:
        For each array in the stream:
        1. Load raw array from disk
        2. Extract features
        3. Bandit picks algorithm
        4. ACTUALLY SORT with chosen algorithm → measure real time
        5. Update bandit with observed reward
        6. (For evaluation) time all 3 algorithms → compute oracle regret

    If use_real_sort=False, falls back to cached CSV timings.
    """
    n = len(df_stream)
    picks = np.zeros(n, dtype=int)
    all_times = np.zeros((n, 3))  # actual timings (real or cached)
    exploration_bonus = np.zeros(n)
    features_used = np.zeros((n, D))

    skipped = 0
    processed = 0

    for i, (_, row) in enumerate(df_stream.iterrows()):
        if i % 2000 == 0 and i > 0:
            print(f"    [{i:,}/{n:,}] processed={processed}, skipped={skipped}")

        if use_real_sort:
            # Load actual array from disk
            arr = load_array(row["file"])
            if arr is None:
                # Can't load → use cached timing as fallback
                x = np.array([row[f] for f in FEATURE_NAMES])
                cached_times = np.array([row[c] for c in TIME_COLS])
                arm, ucb_vals = bandit.select_arm(x)
                reward = -cached_times[arm] * 1e6
                bandit.update(arm, x, reward)

                picks[i] = arm
                all_times[i] = cached_times
                features_used[i] = x
                exploration_bonus[i] = bandit.alpha * np.sqrt(
                    x @ bandit.A_inv[arm] @ x)
                skipped += 1
                continue

            # Extract features from the ACTUAL array (not from CSV cache)
            x = np.array(extract_features(arr))

            # Bandit picks
            arm, ucb_vals = bandit.select_arm(x)

            # ACTUALLY SORT with chosen algorithm and measure time
            repeats = TIMING_REPEATS_SMALL if len(arr) <= 10_000 else TIMING_REPEATS_LARGE
            chosen_time = time_sort(arr, SORT_KINDS[arm], repeats)

            # Reward = negative time in µs
            reward = -chosen_time * 1e6
            bandit.update(arm, x, reward)

            # For evaluation: time ALL algorithms (oracle comparison)
            other_times = np.zeros(3)
            other_times[arm] = chosen_time
            for a in range(3):
                if a != arm:
                    other_times[a] = time_sort(arr, SORT_KINDS[a], repeats)

            picks[i] = arm
            all_times[i] = other_times
            features_used[i] = x
            exploration_bonus[i] = bandit.alpha * np.sqrt(
                x @ bandit.A_inv[arm] @ x)
            processed += 1

        else:
            # Fallback: use cached timings from CSV (no real sorting)
            x = np.array([row[f] for f in FEATURE_NAMES])
            cached_times = np.array([row[c] for c in TIME_COLS])

            arm, ucb_vals = bandit.select_arm(x)
            reward = -cached_times[arm] * 1e6
            bandit.update(arm, x, reward)

            picks[i] = arm
            all_times[i] = cached_times
            features_used[i] = x
            exploration_bonus[i] = bandit.alpha * np.sqrt(
                x @ bandit.A_inv[arm] @ x)
            processed += 1

    print(f"    Final: {processed:,} processed, {skipped:,} skipped (missing files)")

    # Compute per-step and windowed metrics
    best_times = all_times.min(axis=1)
    model_times = np.array([all_times[i, picks[i]] for i in range(n)])
    regrets = (model_times - best_times) * 1e6
    cum_regret = np.cumsum(regrets)

    n_windows = n // window
    windowed_gap_closed = []
    windowed_regret = []
    windowed_accuracy = []
    window_positions = []

    # True best labels from real timings
    y_true = all_times.argmin(axis=1)

    for w in range(n_windows):
        s, e = w * window, (w + 1) * window
        m = compute_regret_metrics(picks[s:e], all_times[s:e])
        windowed_gap_closed.append(m["gap_closed_pct"])
        windowed_regret.append(m["mean_regret_us"])
        windowed_accuracy.append(accuracy_score(y_true[s:e], picks[s:e]) * 100)
        window_positions.append(e)

    return {
        "picks": picks,
        "all_times": all_times,
        "regrets": regrets,
        "cum_regret": cum_regret,
        "exploration_bonus": exploration_bonus,
        "windowed_gap_closed": windowed_gap_closed,
        "windowed_regret": windowed_regret,
        "windowed_accuracy": windowed_accuracy,
        "window_positions": window_positions,
        "processed": processed,
        "skipped": skipped,
    }


# ── V5 baseline (also real sort or cached) ───────────────────────────────

def run_v5_baseline(v5_booster, df_stream: pd.DataFrame,
                    use_real_sort: bool = True) -> dict:
    """Run v5 frozen predictions. If use_real_sort, re-time for fair comparison."""
    n = len(df_stream)
    picks = np.zeros(n, dtype=int)
    all_times = np.zeros((n, 3))

    for i, (_, row) in enumerate(df_stream.iterrows()):
        if use_real_sort:
            arr = load_array(row["file"])
            if arr is None:
                x = np.array([row[f] for f in FEATURE_NAMES])
                all_times[i] = np.array([row[c] for c in TIME_COLS])
            else:
                x = np.array(extract_features(arr))
                all_times[i] = time_all_algorithms(arr)
        else:
            x = np.array([row[f] for f in FEATURE_NAMES])
            all_times[i] = np.array([row[c] for c in TIME_COLS])

        # v5 prediction
        dmat = xgb.DMatrix(x.reshape(1, -1), feature_names=FEATURE_NAMES)
        proba = v5_booster.predict(dmat)
        picks[i] = int(np.argmax(proba[0]))

    regrets = np.array([(all_times[i, picks[i]] - all_times[i].min()) * 1e6
                        for i in range(n)])

    return {"picks": picks, "all_times": all_times,
            "regrets": regrets, "cum_regret": np.cumsum(regrets)}


# ── Figures ──────────────────────────────────────────────────────────────

def plot_cumulative_regret(results_dict, output_path):
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"SBS (heapsort)": "#d62728", "v5 frozen": "#ff7f0e",
              "LinUCB cold": "#1f77b4", "LinUCB warm": "#2ca02c"}
    for name, data in results_dict.items():
        cr = data["cum_regret"]
        ax.plot(range(len(cr)), cr, label=name, linewidth=2,
                color=colors.get(name, None), alpha=0.85)
    ax.set_xlabel("Arrays Sorted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Regret (µs)", fontsize=12, fontweight="bold")
    ax.set_title("Cumulative Regret — Real Online Sorting", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "01_cumulative_regret.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {output_path / '01_cumulative_regret.png'}")


def plot_windowed_gap_closed(results_dict, output_path, window=500):
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"SBS (heapsort)": "#d62728", "v5 frozen": "#ff7f0e",
              "LinUCB cold": "#1f77b4", "LinUCB warm": "#2ca02c"}
    for name, data in results_dict.items():
        if "windowed_gap_closed" not in data:
            continue
        ax.plot(data["window_positions"], data["windowed_gap_closed"],
                label=name, linewidth=2, color=colors.get(name, None), alpha=0.85)
    ax.set_xlabel("Arrays Sorted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Gap Closed (%)", fontsize=12, fontweight="bold")
    ax.set_title(f"Gap Closed Over Time (window={window})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(output_path / "02_windowed_gap_closed.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {output_path / '02_windowed_gap_closed.png'}")


def plot_exploration_decay(results_dict, output_path, window=500):
    fig, ax = plt.subplots(figsize=(14, 6))
    for name in ["LinUCB cold", "LinUCB warm"]:
        if name not in results_dict or "exploration_bonus" not in results_dict[name]:
            continue
        eb = results_dict[name]["exploration_bonus"]
        n_w = len(eb) // window
        means = [np.mean(eb[i*window:(i+1)*window]) for i in range(n_w)]
        positions = [(i+1)*window for i in range(n_w)]
        color = "#1f77b4" if "cold" in name else "#2ca02c"
        ax.plot(positions, means, label=name, linewidth=2, color=color, alpha=0.85)
    ax.set_xlabel("Arrays Sorted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Exploration Bonus", fontsize=12, fontweight="bold")
    ax.set_title("Exploration Decay (Convergence)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "04_exploration_decay.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {output_path / '04_exploration_decay.png'}")


def plot_arm_evolution(results_dict, output_path, window=500):
    names = [n for n in ["LinUCB cold", "LinUCB warm"] if n in results_dict]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(8*len(names), 6))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        picks = results_dict[name]["picks"]
        n_w = len(picks) // window
        colors_alg = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for a in range(N_ARMS):
            props = [(picks[i*window:(i+1)*window] == a).mean() * 100 for i in range(n_w)]
            ax.plot([(i+1)*window for i in range(n_w)], props,
                    label=ALGORITHMS[a], linewidth=2, color=colors_alg[a])
        ax.set_xlabel("Arrays Sorted", fontsize=11, fontweight="bold")
        ax.set_ylabel("Selection %", fontsize=11, fontweight="bold")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    plt.suptitle("Arm Selection Over Time", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / "05_arm_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {output_path / '05_arm_evolution.png'}")


def plot_domain_comparison(domain_results, output_path):
    fig, ax = plt.subplots(figsize=(12, 7))
    domains = list(domain_results.keys())
    v5_gaps = [domain_results[d]["v5"]["gap_closed_pct"] for d in domains]
    warm_gaps = [domain_results[d]["linucb_warm"]["gap_closed_pct"] for d in domains]
    x = np.arange(len(domains))
    width = 0.35
    ax.bar(x - width/2, v5_gaps, width, label="v5 frozen", color="#ff7f0e", alpha=0.8)
    ax.bar(x + width/2, warm_gaps, width, label="LinUCB warm", color="#2ca02c", alpha=0.8)
    ax.set_xlabel("Domain", fontsize=12, fontweight="bold")
    ax.set_ylabel("Gap Closed (%)", fontsize=12, fontweight="bold")
    ax.set_title("Per-Domain: v5 vs LinUCB", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    for i, (v, w) in enumerate(zip(v5_gaps, warm_gaps)):
        ax.annotate(f"{v:.1f}%", xy=(i - width/2, v), ha="center", va="bottom", fontsize=9)
        ax.annotate(f"{w:.1f}%", xy=(i + width/2, w), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path / "06_domain_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {output_path / '06_domain_comparison.png'}")


def plot_long_tail(v5_regrets, warm_regrets, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for label, data, color in [("v5 frozen", v5_regrets, "#ff7f0e"),
                                ("LinUCB warm", warm_regrets, "#2ca02c")]:
        sd = np.sort(data)
        cdf = np.arange(1, len(sd)+1) / len(sd)
        ax1.plot(sd, cdf*100, label=label, linewidth=2, color=color)
    ax1.set_xlabel("Regret (µs)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cumulative %", fontsize=12, fontweight="bold")
    ax1.set_title("Regret CDF", fontsize=13, fontweight="bold")
    ax1.set_xlim(0, np.percentile(np.concatenate([v5_regrets, warm_regrets]), 99.5))
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    pctls = [90, 95, 97, 99, 99.5, 99.9]
    v5_p = [np.percentile(v5_regrets, p) for p in pctls]
    warm_p = [np.percentile(warm_regrets, p) for p in pctls]
    x = np.arange(len(pctls))
    w = 0.35
    ax2.bar(x-w/2, v5_p, w, label="v5 frozen", color="#ff7f0e", alpha=0.8)
    ax2.bar(x+w/2, warm_p, w, label="LinUCB warm", color="#2ca02c", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"P{p}" for p in pctls])
    ax2.set_ylabel("Regret (µs)", fontsize=12, fontweight="bold")
    ax2.set_title("Tail Regret", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Long-Tail: v5 vs LinUCB", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / "07_long_tail.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {output_path / '07_long_tail.png'}")


def plot_alpha_sensitivity(alpha_results, output_path):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    alphas = sorted(alpha_results.keys())
    gaps = [alpha_results[a]["gap_closed_pct"] for a in alphas]
    regs = [alpha_results[a]["mean_regret_us"] for a in alphas]
    ax1.plot(alphas, gaps, "b-o", linewidth=2, markersize=6, label="Gap Closed (%)")
    ax1.set_xlabel("α (exploration weight)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Gap Closed (%)", fontsize=12, fontweight="bold", color="blue")
    ax2 = ax1.twinx()
    ax2.plot(alphas, regs, "r--s", linewidth=2, markersize=6, label="Mean Regret (µs)")
    ax2.set_ylabel("Mean Regret (µs)", fontsize=12, fontweight="bold", color="red")
    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, l1+l2, fontsize=10, loc="center right")
    ax1.set_title("α Sensitivity (LinUCB Warm)", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "08_alpha_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  {output_path / '08_alpha_sensitivity.png'}")


# ── MAIN ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit test stream size (0=all)")
    parser.add_argument("--skip-sort", action="store_true",
                        help="Use cached CSV timings instead of real sorting")
    args = parser.parse_args()

    use_real_sort = not args.skip_sort

    print("=" * 80)
    print("STEP 5: LinUCB — TRUE Online Adaptive Selector")
    print(f"  Mode: {'REAL SORTING' if use_real_sort else 'CACHED TIMINGS (--skip-sort)'}")
    print("=" * 80)

    t0 = time_module.time()
    np.random.seed(SEED)

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"\n[LOAD] {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    print(f"  {len(df):,} rows")

    # Check if raw arrays are accessible
    if use_real_sort and not RAW_DIR.exists():
        print(f"\n  WARNING: {RAW_DIR} not found. Falling back to cached timings.")
        print(f"  Connect external drive or use --skip-sort")
        use_real_sort = False

    # ── Load v5 ───────────────────────────────────────────────────────────
    print(f"\n[V5] Loading {V5_MODEL}")
    le = LabelEncoder().fit(ALGORITHMS)
    v5_booster = xgb.Booster()
    v5_booster.load_model(str(V5_MODEL))

    # ── Split: undersample + 70/15/15 (matching v5 exactly) ──────────────
    print(f"\n[SPLIT] Balanced undersample + 70/15/15")

    def balanced_undersample(df_in, label_col, max_ratio=3.0):
        counts = df_in[label_col].value_counts()
        cap = int(counts.min() * max_ratio)
        parts = []
        for cls in counts.index:
            sub = df_in[df_in[label_col] == cls]
            if len(sub) > cap:
                sub = sub.sample(n=cap, random_state=SEED)
            parts.append(sub)
        return pd.concat(parts, ignore_index=True).sample(
            frac=1.0, random_state=SEED).reset_index(drop=True)

    df_bal = balanced_undersample(df, "best_algorithm", max_ratio=3.0)

    # Split keeping the DataFrame intact (need 'file' column for loading)
    idx_train, idx_temp = train_test_split(
        np.arange(len(df_bal)), test_size=0.30,
        stratify=df_bal["best_algorithm"], random_state=SEED
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50,
        stratify=df_bal["best_algorithm"].iloc[idx_temp], random_state=SEED
    )

    df_train = df_bal.iloc[idx_train].reset_index(drop=True)
    df_val   = df_bal.iloc[idx_val].reset_index(drop=True)
    df_test  = df_bal.iloc[idx_test].reset_index(drop=True)

    if args.limit > 0:
        df_test = df_test.head(args.limit)

    # Shuffle test stream (random online arrival order)
    df_test = df_test.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # Warm-start data (use cached timings — these are pre-recorded)
    X_train = df_train[FEATURE_NAMES].values
    t_train = df_train[TIME_COLS].values

    print(f"  Train (warm-start): {len(df_train):,}")
    print(f"  Test stream:        {len(df_test):,}")

    WINDOW = max(100, len(df_test) // 100)  # ~100 windows

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: In-distribution — 4 strategies
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1: In-Distribution (stream={len(df_test):,}, window={WINDOW})")
    print(f"{'='*60}")

    all_results = {}

    # ── SBS (always heapsort) ─────────────────────────────────────────────
    print(f"\n[SBS] Always heapsort...")
    sbs_times = df_test[TIME_COLS].values
    sbs_picks = np.ones(len(df_test), dtype=int)
    sbs_regrets = np.array([(sbs_times[i, 1] - sbs_times[i].min()) * 1e6
                            for i in range(len(df_test))])
    n_w = len(df_test) // WINDOW
    sbs_wgc, sbs_wr, sbs_wpos = [], [], []
    for w in range(n_w):
        s, e = w*WINDOW, (w+1)*WINDOW
        m = compute_regret_metrics(sbs_picks[s:e], sbs_times[s:e])
        sbs_wgc.append(m["gap_closed_pct"])
        sbs_wr.append(m["mean_regret_us"])
        sbs_wpos.append(e)
    all_results["SBS (heapsort)"] = {
        "picks": sbs_picks, "regrets": sbs_regrets,
        "cum_regret": np.cumsum(sbs_regrets),
        "windowed_gap_closed": sbs_wgc, "windowed_regret": sbs_wr,
        "window_positions": sbs_wpos,
    }
    sbs_metrics = compute_regret_metrics(sbs_picks, sbs_times)
    print(f"  Gap closed: {sbs_metrics['gap_closed_pct']:.2f}%")

    # ── V5 frozen ─────────────────────────────────────────────────────────
    print(f"\n[V5] Frozen classifier...")
    X_test_cached = df_test[FEATURE_NAMES].values
    t_test_cached = df_test[TIME_COLS].values
    dtest = xgb.DMatrix(X_test_cached, feature_names=FEATURE_NAMES)
    v5_proba = v5_booster.predict(dtest)
    v5_picks = np.argmax(v5_proba, axis=1).astype(int)
    v5_regrets = np.array([(t_test_cached[i, v5_picks[i]] - t_test_cached[i].min()) * 1e6
                           for i in range(len(df_test))])
    v5_wgc, v5_wr, v5_wpos = [], [], []
    for w in range(n_w):
        s, e = w*WINDOW, (w+1)*WINDOW
        m = compute_regret_metrics(v5_picks[s:e], t_test_cached[s:e])
        v5_wgc.append(m["gap_closed_pct"])
        v5_wr.append(m["mean_regret_us"])
        v5_wpos.append(e)
    all_results["v5 frozen"] = {
        "picks": v5_picks, "regrets": v5_regrets,
        "cum_regret": np.cumsum(v5_regrets),
        "windowed_gap_closed": v5_wgc, "windowed_regret": v5_wr,
        "window_positions": v5_wpos,
    }
    v5_metrics = compute_regret_metrics(v5_picks, t_test_cached)
    print(f"  Gap closed: {v5_metrics['gap_closed_pct']:.2f}%  "
          f"Perfect: {v5_metrics['perfect_picks_pct']:.2f}%")

    # ── LinUCB cold start ─────────────────────────────────────────────────
    print(f"\n[COLD] LinUCB cold start (α=1.0)...")
    bandit_cold = LinUCB(d=D, n_arms=N_ARMS, alpha=1.0, lambda_reg=1.0)
    cold_data = run_online_experiment(bandit_cold, df_test,
                                      use_real_sort=use_real_sort, window=WINDOW)
    all_results["LinUCB cold"] = cold_data
    cold_metrics = compute_regret_metrics(cold_data["picks"], cold_data["all_times"])
    print(f"  Gap closed: {cold_metrics['gap_closed_pct']:.2f}%  "
          f"Perfect: {cold_metrics['perfect_picks_pct']:.2f}%  "
          f"Mean regret: {cold_metrics['mean_regret_us']:.3f}µs")

    # ── LinUCB warm start ─────────────────────────────────────────────────
    print(f"\n[WARM] LinUCB warm start (α=1.0, warm from {len(X_train):,} arrays)...")
    bandit_warm = LinUCB(d=D, n_arms=N_ARMS, alpha=1.0, lambda_reg=1.0)
    bandit_warm.warm_start(X_train, t_train)
    warm_data = run_online_experiment(bandit_warm, df_test,
                                      use_real_sort=use_real_sort, window=WINDOW)
    all_results["LinUCB warm"] = warm_data
    warm_metrics = compute_regret_metrics(warm_data["picks"], warm_data["all_times"])
    print(f"  Gap closed: {warm_metrics['gap_closed_pct']:.2f}%  "
          f"Perfect: {warm_metrics['perfect_picks_pct']:.2f}%  "
          f"Mean regret: {warm_metrics['mean_regret_us']:.3f}µs")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Strategy':<20} {'Gap%':>7} {'Perfect%':>9} {'MeanReg':>9} {'P99':>8} {'Max':>8}")
    print(f"{'-'*70}")
    for name, metrics in [("SBS (heapsort)", sbs_metrics),
                           ("v5 frozen", v5_metrics),
                           ("LinUCB cold", cold_metrics),
                           ("LinUCB warm", warm_metrics)]:
        print(f"{name:<20} {metrics['gap_closed_pct']:>6.2f}% "
              f"{metrics['perfect_picks_pct']:>8.2f}% "
              f"{metrics['mean_regret_us']:>8.3f}µ "
              f"{metrics['p99_regret_us']:>7.2f}µ "
              f"{metrics['max_regret_us']:>7.1f}µ")

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: Per-Domain
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 2: Per-Domain Analysis")
    print(f"{'='*60}")

    domain_results = {}
    for domain in df_test["domain"].unique():
        mask = df_test["domain"] == domain
        if mask.sum() < 50:
            continue
        dom_t = t_test_cached[mask.values]
        dom_v5 = v5_picks[mask.values]
        dom_warm = warm_data["picks"][mask.values]
        # Use the same timing source for both (fair comparison)
        dom_times = warm_data["all_times"][mask.values] if use_real_sort else dom_t

        v5_dom = compute_regret_metrics(dom_v5, dom_t)
        warm_dom = compute_regret_metrics(dom_warm, dom_times)

        domain_results[domain] = {
            "n": int(mask.sum()),
            "v5": v5_dom,
            "linucb_warm": warm_dom,
        }
        delta = warm_dom["gap_closed_pct"] - v5_dom["gap_closed_pct"]
        print(f"  {domain:15s} (n={mask.sum():>5,}): "
              f"v5={v5_dom['gap_closed_pct']:>6.2f}%  "
              f"LinUCB={warm_dom['gap_closed_pct']:>6.2f}%  "
              f"Δ={delta:>+6.2f}pp")

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: α Sensitivity
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 3: α Sensitivity")
    print(f"{'='*60}")

    alpha_results = {}
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        b = LinUCB(d=D, n_arms=N_ARMS, alpha=alpha, lambda_reg=1.0)
        b.warm_start(X_train, t_train)
        a_data = run_online_experiment(b, df_test, use_real_sort=use_real_sort,
                                       window=WINDOW)
        a_metrics = compute_regret_metrics(a_data["picks"], a_data["all_times"])
        alpha_results[alpha] = a_metrics
        print(f"  α={alpha:<5.2f}: gap={a_metrics['gap_closed_pct']:.2f}%  "
              f"regret={a_metrics['mean_regret_us']:.3f}µs  "
              f"perfect={a_metrics['perfect_picks_pct']:.2f}%")

    # ══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[FIGURES]")
    plot_cumulative_regret(all_results, FIGURES_DIR)
    plot_windowed_gap_closed(all_results, FIGURES_DIR, WINDOW)
    plot_exploration_decay(all_results, FIGURES_DIR, WINDOW)
    plot_arm_evolution(all_results, FIGURES_DIR, WINDOW)
    if domain_results:
        plot_domain_comparison(domain_results, FIGURES_DIR)
    plot_long_tail(v5_regrets, warm_data["regrets"], FIGURES_DIR)
    plot_alpha_sensitivity(alpha_results, FIGURES_DIR)

    # ══════════════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[SAVE]")
    bandit_warm.save(MODEL_DIR / "linucb_warm.json")
    print(f"  Bandit → {MODEL_DIR / 'linucb_warm.json'}")

    output = {
        "timestamp": datetime.now().isoformat(),
        "version": "linucb_step5",
        "mode": "real_sorting" if use_real_sort else "cached_timings",
        "params": {"d": D, "n_arms": N_ARMS, "alpha": 1.0, "lambda_reg": 1.0},
        "dataset": {
            "total_raw": len(df),
            "warm_start": len(X_train),
            "test_stream": len(df_test),
        },
        "experiment1": {
            "sbs": sbs_metrics,
            "v5_frozen": v5_metrics,
            "linucb_cold": cold_metrics,
            "linucb_warm": warm_metrics,
        },
        "experiment2_domains": {
            d: {"n": dr["n"],
                "v5_gap": dr["v5"]["gap_closed_pct"],
                "linucb_gap": dr["linucb_warm"]["gap_closed_pct"],
                "delta_pp": round(dr["linucb_warm"]["gap_closed_pct"] -
                                  dr["v5"]["gap_closed_pct"], 4)}
            for d, dr in domain_results.items()
        },
        "experiment3_alpha": {
            str(a): {"gap": r["gap_closed_pct"], "regret": r["mean_regret_us"],
                     "perfect": r["perfect_picks_pct"]}
            for a, r in alpha_results.items()
        },
    }

    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results → {RESULTS_DIR / 'evaluation_results.json'}")

    elapsed = time_module.time() - t0
    print(f"\n{'='*80}")
    print(f"STEP 5 COMPLETE. {elapsed:.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
