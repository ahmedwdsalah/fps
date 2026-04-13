"""
Benchmark XGBoost v5 against 5 baseline models on the same training data and features.
Outputs: benchmarks/benchmark_results.png + benchmarks/benchmark_results.json
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

# ── Paths ──
ROOT = Path("/Users/ahmed/Desktop/My-Master-thesis")
DATA = ROOT / "data" / "training_dataset.csv"
MODEL_PATH = ROOT / "models" / "xgboost_v5" / "xgb_v5.json"
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
ALGORITHMS = ["introsort", "heapsort", "timsort"]
FEATURES = [
    "length_norm", "adj_sorted_ratio", "duplicate_ratio", "dispersion_ratio",
    "runs_ratio", "inversion_ratio", "entropy_ratio", "skewness_t",
    "kurtosis_excess_t", "longest_run_ratio", "iqr_norm", "mad_norm",
    "top1_freq_ratio", "top5_freq_ratio", "outlier_ratio", "mean_abs_diff_norm",
]

print("[1/5] Loading data...")
df = pd.read_csv(DATA)
print(f"  Loaded {len(df):,} rows")

# ── Same balancing as v5 ──
margin = df[["time_introsort","time_heapsort","time_timsort"]].values
best_time = margin.min(axis=1)
second_best = np.partition(margin, 1, axis=1)[:, 1]
pct_margin = (second_best - best_time) / (best_time + 1e-15)
mask = (pct_margin >= 0.05) | (df["n_elements"] >= 2000)
df_clean = df[mask].copy()
print(f"  After margin filter: {len(df_clean):,}")

# Undersample
counts = df_clean["best_algorithm"].value_counts()
min_count = counts.min()
cap = int(min_count * 3.0)
parts = []
for cls in counts.index:
    subset = df_clean[df_clean["best_algorithm"] == cls]
    if len(subset) > cap:
        subset = subset.sample(n=cap, random_state=SEED)
    parts.append(subset)
df_bal = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED)
print(f"  After undersample: {len(df_bal):,}")

X = df_bal[FEATURES].values
y = df_bal["best_algorithm"].values
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.15, random_state=SEED, stratify=y_enc
)
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

# ── Also need timing columns for regret ──
df_bal_test_idx = df_bal.iloc[
    train_test_split(range(len(df_bal)), test_size=0.15, random_state=SEED, stratify=y_enc)[1]
]
time_cols = df_bal_test_idx[["time_introsort","time_heapsort","time_timsort"]].values

# ── Models ──
print("\n[2/5] Training models...")
models = {}

# 1. Random baseline
class RandomBaseline:
    def fit(self, X, y): self.classes_ = np.unique(y); return self
    def predict(self, X): return np.random.RandomState(SEED).choice(self.classes_, size=len(X))

# 2. Always SBS (heapsort)
class AlwaysSBS:
    def fit(self, X, y): self.label = 1; return self  # heapsort=1 after encoding
    def predict(self, X): return np.full(len(X), self.label)

models["Random"] = RandomBaseline()
models["Always Heapsort (SBS)"] = AlwaysSBS()
models["Decision Tree"] = DecisionTreeClassifier(max_depth=10, random_state=SEED)
models["Random Forest"] = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1)
models["KNN (k=15)"] = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
models["MLP"] = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=SEED, early_stopping=True)

# Train all
for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    dt = time.time() - t0
    print(f"  {name:30s} trained in {dt:.1f}s")

# 3. XGBoost v5 (pre-trained, just load)
print("  Loading XGBoost v5...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(MODEL_PATH))
models["XGBoost v5"] = xgb_model

# ── Evaluate ──
print("\n[3/5] Evaluating...")
results = {}
for name, model in models.items():
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)

    # Regret calculation
    pred_labels = le.inverse_transform(preds)
    algo_to_col = {"introsort": 0, "heapsort": 1, "timsort": 2}
    pred_times = np.array([time_cols[i, algo_to_col[pred_labels[i]]] for i in range(len(preds))])
    vbs_times = time_cols.min(axis=1)
    sbs_times = time_cols[:, 1]  # heapsort

    model_total = pred_times.sum()
    vbs_total = vbs_times.sum()
    sbs_total = sbs_times.sum()

    gap_closed = (sbs_total - model_total) / (sbs_total - vbs_total + 1e-15) * 100
    mean_regret = (pred_times - vbs_times).mean() * 1e6  # microseconds
    perfect_picks = (pred_times == vbs_times).mean() * 100

    results[name] = {
        "accuracy": round(acc * 100, 1),
        "balanced_accuracy": round(bal_acc * 100, 1),
        "gap_closed": round(gap_closed, 1),
        "mean_regret_us": round(mean_regret, 2),
        "perfect_picks": round(perfect_picks, 1),
    }
    print(f"  {name:30s}  acc={acc*100:.1f}%  bal_acc={bal_acc*100:.1f}%  gap={gap_closed:.1f}%  regret={mean_regret:.2f}μs")

# Save JSON
with open(OUT_DIR / "benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ── Plot ──
print("\n[4/5] Generating figures...")
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
gap_closed = [results[n]["gap_closed"] for n in names]
regrets = [results[n]["mean_regret_us"] for n in names]
perfect = [results[n]["perfect_picks"] for n in names]

# Color: ours = green, rest = gray shades
colors = ["#aaaaaa" if "XGBoost" not in n else "#2ecc71" for n in names]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Model Benchmark: XGBoost v5 vs Baselines", fontsize=16, fontweight="bold")

# 1. Accuracy
ax = axes[0, 0]
bars = ax.barh(names, accs, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Accuracy (%)")
ax.set_title("Test Accuracy")
for bar, val in zip(bars, accs):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{val}%", va="center", fontsize=10)
ax.set_xlim(0, 105)

# 2. Gap Closed
ax = axes[0, 1]
bars = ax.barh(names, gap_closed, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Gap Closed (%)")
ax.set_title("VBS-SBS Gap Closed (higher = better)")
for bar, val in zip(bars, gap_closed):
    ax.text(max(bar.get_width() + 0.5, 2), bar.get_y() + bar.get_height()/2, f"{val}%", va="center", fontsize=10)

# 3. Mean Regret
ax = axes[1, 0]
bars = ax.barh(names, regrets, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Mean Regret (μs)")
ax.set_title("Mean Regret vs Oracle (lower = better)")
for bar, val in zip(bars, regrets):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"{val}μs", va="center", fontsize=10)

# 4. Perfect Picks
ax = axes[1, 1]
bars = ax.barh(names, perfect, color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Perfect Picks (%)")
ax.set_title("Arrays Where Model Picks True Best (%)")
for bar, val in zip(bars, perfect):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{val}%", va="center", fontsize=10)
ax.set_xlim(0, 105)

plt.tight_layout()
plt.savefig(OUT_DIR / "benchmark_results.png", dpi=200, bbox_inches="tight")
print(f"\n[5/5] Saved: {OUT_DIR / 'benchmark_results.png'}")
print(f"       Saved: {OUT_DIR / 'benchmark_results.json'}")
print("\nDone!")
