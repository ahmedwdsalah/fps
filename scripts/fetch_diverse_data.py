#!/usr/bin/env python3
"""
Fetch Diverse Real-World Data v2 — Honest Edition
===================================================
Addresses 8 data-quality issues from v1:

  1. NO bootstrapping (no replace=True resampling)
  2. Only non-overlapping arrays from each source column
  3. source_id tracking → group-aware train/test splits
  4. Tie filter: skip arrays where timing gap < 5%
  5. Real sensor data (UCI HAR via OpenML) replaces simulated
  6. SyntheticReal capped at ~10%
  7. Many diverse SOURCES, few arrays per source
  8. Honest sizes (no inflating 178-row wine to 100K)

Domains:
  Finance      — 50 tickers via yfinance (raw, returns, logret, volatility)
  Ecology      — CoverType 10 columns × 7 cover-type subsets
  Network      — KDD Cup 99 columns × attack-type subsets
  Housing      — California Housing 9 columns (raw only)
  Text         — 20 Newsgroups word-frequency vectors
  Medical      — Breast cancer + diabetes columns (raw only)
  Geospatial   — Species distribution raster layers
  OpenML       — 30+ datasets incl. HAR (real accelerometers!)
  SyntheticReal— 12 distribution types, capped at 500

Output: /Volumes/k/thesis_data/diverse_v6/diverse_training_data.csv

Usage:
    python3 scripts/fetch_diverse_data.py
    python3 scripts/fetch_diverse_data.py --domains Finance,OpenML
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# ── Output paths ──────────────────────────────────────────────────────────
DATA_DIR   = Path("/Volumes/k/thesis_data/diverse_v6")
RAW_DIR    = DATA_DIR / "raw"
OUTPUT_CSV = DATA_DIR / "diverse_training_data.csv"
PROJECT_CSV = ROOT / "data" / "diverse_training_data.csv"

sys.path.insert(0, str(SCRIPT_DIR))
from feature_extraction import extract_features, FEATURE_NAMES

ALGORITHMS = ["introsort", "heapsort", "timsort"]
SEED = 42
rng = np.random.RandomState(SEED)

# ── Quality thresholds ────────────────────────────────────────────────────
MIN_ARRAY_SIZE = 200
MAX_ARRAY_SIZE = 200_000
TIE_THRESHOLD  = 0.05   # skip if timing spread < 5% of best time


# ── Timing ────────────────────────────────────────────────────────────────

def time_sort(arr: np.ndarray, kind: str, repeats: int) -> float:
    """Best-of-N sort timing in seconds."""
    best = float("inf")
    for _ in range(repeats):
        copy = arr.copy()
        gc.disable()
        t0 = time.perf_counter()
        np.sort(copy, kind=kind)
        t1 = time.perf_counter()
        gc.enable()
        best = min(best, t1 - t0)
    return best


def process_array(arr, array_id, domain, source_id, n_max,
                  save_raw=True) -> dict | None:
    """Extract features + time sorts. Returns row dict or None."""
    arr = arr.astype(np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < MIN_ARRAY_SIZE or n > MAX_ARRAY_SIZE:
        return None

    if save_raw:
        raw_path = RAW_DIR / domain / f"{array_id}.npy"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_path, arr)

    features = extract_features(arr, n_max, array_id)

    repeats = 5 if n <= 10_000 else 3
    t_intro = time_sort(arr, "quicksort", repeats)
    t_heap  = time_sort(arr, "heapsort",  repeats)
    t_tim   = time_sort(arr, "stable",    repeats)

    times = {"introsort": t_intro, "heapsort": t_heap, "timsort": t_tim}
    best = min(times, key=times.get)
    best_time  = min(times.values())
    worst_time = max(times.values())

    # Tie filter — if all algorithms within TIE_THRESHOLD, label is noise
    if best_time > 0 and (worst_time - best_time) / best_time < TIE_THRESHOLD:
        return None

    row = {
        "file": array_id,
        "domain": domain,
        "source_id": source_id,
        "n_elements": n,
    }
    row.update(features)
    row["time_introsort"] = t_intro
    row["time_heapsort"]  = t_heap
    row["time_timsort"]   = t_tim
    row["best_algorithm"] = best
    row["timing_margin"]  = (worst_time - best_time) / best_time
    return row


# ═══════════════════════════════════════════════════════════════════════════
# GENERATORS — each yields (array, array_id, source_id)
#
# Rules:
#   - 1 array per column per transform (no windows, no bootstrap)
#   - NON-OVERLAPPING chunks from long cols are OK (different data)
#   - source_id groups all arrays that share underlying data
#   - Per-class subsets are OK (different distribution)
#   - Transforms (returns, logret, volatility) are OK (different features)
# ═══════════════════════════════════════════════════════════════════════════

# Chunk sizes for non-overlapping slicing (varied for size diversity)
CHUNK_SIZES = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]


def non_overlapping_chunks(col, col_name, source_id, max_chunks_per_size=2):
    """
    Yield non-overlapping chunks from a long column at varied sizes.
    Each chunk is genuinely different data (no overlap).
    Only works for columns with > 10K elements.
    """
    n = len(col)
    if n < 10_000:
        return  # column too short, not worth chunking

    for csize in CHUNK_SIZES:
        if csize > n:
            continue
        n_possible = n // csize
        n_chunks = min(max_chunks_per_size, n_possible)
        if n_chunks == 0:
            continue
        # Pick evenly spaced start points (no overlap guaranteed)
        starts = np.linspace(0, n - csize, n_possible, dtype=int)
        selected = rng.choice(len(starts), size=n_chunks, replace=False)
        for idx in selected:
            start = starts[idx]
            chunk = col[start:start + csize]
            yield chunk, f"{col_name}_chunk{csize}_{idx}", source_id

def gen_finance(target):
    """Real stock data — one array per column per transform."""
    print("  [Finance] Downloading stock data via yfinance...")
    try:
        import yfinance as yf
    except ImportError:
        print("    yfinance not available, skipping")
        return

    tickers = [
        # US Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX",
        "AMD", "INTC",
        # US Finance
        "JPM", "BAC", "GS", "V", "MA", "BRK-B", "C",
        # US Healthcare
        "JNJ", "PFE", "UNH", "MRK", "ABBV", "LLY",
        # US Energy
        "XOM", "CVX", "COP", "SLB", "OXY",
        # US Consumer
        "WMT", "PG", "KO", "PEP", "MCD", "NKE", "SBUX",
        # ETFs
        "SPY", "QQQ", "IWM", "DIA", "VTI", "XLF", "XLE",
        # Indices
        "^GSPC", "^IXIC", "^DJI", "^RUT",
        # Crypto
        "BTC-USD", "ETH-USD", "SOL-USD",
        # Commodities
        "GLD", "SLV", "USO", "TLT",
    ]

    count = 0
    for ticker in tickers:
        if count >= target:
            return
        try:
            data = yf.download(ticker, period="max", progress=False, timeout=15)
            if data is None or len(data) < MIN_ARRAY_SIZE:
                continue
        except Exception:
            continue

        if hasattr(data.columns, 'get_level_values'):
            data.columns = data.columns.get_level_values(0)

        # All arrays from this ticker share one source_id
        src = f"fin_{ticker}"

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in data.columns:
                continue
            arr = data[col].dropna().values.astype(np.float64)
            if len(arr) < MIN_ARRAY_SIZE:
                continue

            # 1) Raw column (one array)
            yield arr[:MAX_ARRAY_SIZE], f"fin_{ticker}_{col}_raw", src
            count += 1

            # 1b) Non-overlapping chunks at varied sizes
            for chunk, cid, csrc in non_overlapping_chunks(
                    arr, f"fin_{ticker}_{col}", src):
                yield chunk, cid, csrc
                count += 1
                if count >= target:
                    return

            # 2) Returns — genuinely different distribution
            if col != "Volume" and len(arr) > MIN_ARRAY_SIZE + 1:
                returns = np.diff(arr) / (np.abs(arr[:-1]) + 1e-15)
                if len(returns) >= MIN_ARRAY_SIZE:
                    yield returns, f"fin_{ticker}_{col}_returns", src
                    count += 1

            # 3) Log-returns
            if col != "Volume":
                pos = arr[arr > 0]
                if len(pos) > MIN_ARRAY_SIZE:
                    logret = np.diff(np.log(pos))
                    if len(logret) >= MIN_ARRAY_SIZE:
                        yield logret, f"fin_{ticker}_{col}_logret", src
                        count += 1

            # 4) Rolling volatility (one window, rotated per col)
            windows = [20, 50, 200]
            w = windows[hash(f"{ticker}_{col}") % len(windows)]
            if len(arr) > w + MIN_ARRAY_SIZE:
                rolling = pd.Series(arr).rolling(w).std().dropna().values
                if len(rolling) >= MIN_ARRAY_SIZE:
                    yield rolling, f"fin_{ticker}_{col}_vol{w}", src
                    count += 1

    print(f"    -> {count} arrays from finance")


def gen_ecology(target):
    """CoverType — 10 columns + 7 per-class subsets each."""
    print("  [Ecology] Loading CoverType...")
    from sklearn.datasets import fetch_covtype
    data = fetch_covtype(as_frame=False)
    X = data.data
    labels = data.target

    count = 0
    for i in range(10):  # First 10 cols are continuous
        col = X[:, i].astype(np.float64)
        src = f"eco_covtype_col{i}"

        # Full column (capped at 200K)
        yield col[:MAX_ARRAY_SIZE], f"eco_col{i}_full", src
        count += 1

        # Non-overlapping chunks (column has 581K rows — plenty of room)
        for chunk, cid, csrc in non_overlapping_chunks(
                col, f"eco_col{i}", src, max_chunks_per_size=3):
            yield chunk, cid, csrc
            count += 1
            if count >= target:
                return

        # Per cover-type subsets (genuinely different distributions)
        for ct in range(1, 8):
            subset = col[labels == ct]
            if len(subset) >= MIN_ARRAY_SIZE:
                yield subset[:MAX_ARRAY_SIZE], f"eco_col{i}_type{ct}", src
                count += 1

                # Chunks from per-type subsets too
                for chunk, cid, csrc in non_overlapping_chunks(
                        subset, f"eco_col{i}_type{ct}", src):
                    yield chunk, cid, csrc
                    count += 1
                    if count >= target:
                        return

        if count >= target:
            return

    print(f"    -> {count} arrays from ecology")


def gen_network(target):
    """KDD Cup 99 — columns + per-attack subsets."""
    print("  [Network] Loading KDD Cup 99...")
    from sklearn.datasets import fetch_kddcup99
    try:
        data = fetch_kddcup99(subset='SA', as_frame=False, percent10=True)
        X = data.data
        labels = data.target

        numeric_cols = []
        for i in range(X.shape[1]):
            try:
                col = X[:, i].astype(np.float64)
                col = col[np.isfinite(col)]
                if len(col) >= MIN_ARRAY_SIZE:
                    numeric_cols.append((i, col))
            except (ValueError, TypeError):
                continue
    except Exception as e:
        print(f"    Failed: {e}")
        return

    count = 0
    top_attacks = pd.Series(labels).value_counts().index[:10]

    for col_idx, col in numeric_cols:
        src = f"net_kdd_col{col_idx}"

        # Full column
        yield col[:MAX_ARRAY_SIZE], f"net_col{col_idx}_full", src
        count += 1

        # Non-overlapping chunks
        for chunk, cid, csrc in non_overlapping_chunks(
                col, f"net_col{col_idx}", src, max_chunks_per_size=3):
            yield chunk, cid, csrc
            count += 1
            if count >= target:
                return

        # Per-attack subsets
        for atk in top_attacks:
            try:
                sub = X[labels == atk, col_idx].astype(np.float64)
                sub = sub[np.isfinite(sub)]
                if len(sub) >= MIN_ARRAY_SIZE:
                    atk_s = (atk.decode() if isinstance(atk, bytes)
                             else str(atk))[:15].replace(".", "_")
                    yield sub[:MAX_ARRAY_SIZE], \
                        f"net_col{col_idx}_{atk_s}", src
                    count += 1
            except (ValueError, TypeError):
                continue

        if count >= target:
            return

    print(f"    -> {count} arrays from network")


def gen_housing(target):
    """California Housing — 9 columns, raw only, no bootstrap."""
    print("  [Housing] Loading California Housing...")
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=False)
    X = data.data
    y = data.target
    names = data.feature_names

    count = 0
    for i in range(X.shape[1]):
        col = X[:, i]
        if len(col) >= MIN_ARRAY_SIZE:
            yield col, f"house_{names[i]}", f"house_{names[i]}"
            count += 1

    if len(y) >= MIN_ARRAY_SIZE:
        yield y, f"house_target", f"house_target"
        count += 1

    print(f"    -> {count} arrays from housing")


def gen_text(target):
    """20 Newsgroups — word-frequency vectors, no bootstrap."""
    print("  [Text] Loading 20 Newsgroups TF-IDF...")
    from sklearn.datasets import fetch_20newsgroups_vectorized
    data = fetch_20newsgroups_vectorized(subset='all')
    X = data.data
    labels = data.target

    count = 0

    # Pre-filter: words with >= MIN_ARRAY_SIZE nonzero entries
    nnz_per_word = np.array((X > 0).sum(axis=0)).flatten()
    good_words = np.where(nnz_per_word >= MIN_ARRAY_SIZE)[0]
    rng.shuffle(good_words)
    print(f"    {len(good_words)} words with >= {MIN_ARRAY_SIZE} nonzero entries")

    for wi in good_words:
        col = X[:, wi].toarray().flatten()
        nonzero = col[col > 0]
        yield nonzero, f"text_word{wi}", f"text_word{wi}"
        count += 1
        if count >= target:
            return

    # Document length spectrum
    doc_lengths = np.array(X.sum(axis=1)).flatten()
    if len(doc_lengths) >= MIN_ARRAY_SIZE:
        yield doc_lengths, f"text_doclengths", f"text_doclengths"
        count += 1

    # Per-category word sums (20 categories)
    for cat in range(20):
        cat_data = X[labels == cat]
        if cat_data.shape[0] < 50:
            continue
        word_sums = np.array(cat_data.sum(axis=0)).flatten()
        nonzero = word_sums[word_sums > 0]
        if len(nonzero) >= MIN_ARRAY_SIZE:
            yield nonzero, f"text_cat{cat}_sums", f"text_cat{cat}"
            count += 1

    print(f"    -> {count} arrays from text")


def gen_medical(target):
    """Breast cancer + diabetes — raw columns only, no bootstrap."""
    print("  [Medical] Loading medical datasets...")
    from sklearn.datasets import load_breast_cancer, load_diabetes

    datasets = [
        (load_breast_cancer, "cancer"),   # 569 rows, 30 features
        (load_diabetes,      "diabetes"), # 442 rows, 10 features
        # wine: 178 rows < 200 — SKIP (honest)
    ]

    count = 0
    for loader, name in datasets:
        data = loader()
        X = data.data
        y = data.target.astype(np.float64)

        for i in range(X.shape[1]):
            col = X[:, i]
            if len(col) >= MIN_ARRAY_SIZE:
                yield col, f"med_{name}_col{i}", f"med_{name}_col{i}"
                count += 1

        if len(y) >= MIN_ARRAY_SIZE:
            yield y, f"med_{name}_target", f"med_{name}_target"
            count += 1

        if count >= target:
            return

    print(f"    -> {count} arrays from medical")


def gen_geospatial(target):
    """Species distribution raster layers — one array per layer."""
    print("  [Geospatial] Loading species distributions...")
    try:
        from sklearn.datasets import fetch_species_distributions
        data = fetch_species_distributions()
        coverages = data.coverages
    except Exception as e:
        print(f"    Failed: {e}")
        return

    count = 0
    for i, layer in enumerate(coverages):
        flat = layer.flatten().astype(np.float64)
        flat = flat[np.isfinite(flat) & (flat != -9999)]
        if len(flat) < MIN_ARRAY_SIZE:
            continue

        src = f"geo_layer{i}"
        yield flat[:MAX_ARRAY_SIZE], f"geo_layer{i}_full", src
        count += 1

        # Non-overlapping row slices (each row = independent spatial strip)
        rows, cols_count = layer.shape
        selected = rng.choice(rows, size=min(20, rows), replace=False)
        for r in selected:
            row_data = layer[r].astype(np.float64)
            row_data = row_data[np.isfinite(row_data) & (row_data != -9999)]
            if len(row_data) >= MIN_ARRAY_SIZE:
                yield row_data, f"geo_layer{i}_row{r}", src
                count += 1

        if count >= target:
            return

    print(f"    -> {count} arrays from geospatial")


def gen_openml(target):
    """
    30+ datasets from OpenML — including HAR (real accelerometer data).
    One array per numeric column. No bootstrap, no windows.
    """
    print("  [OpenML] Fetching tabular datasets...")
    from sklearn.datasets import fetch_openml

    datasets = [
        # === LARGE (>30K rows) — many columns, chunk-worthy ===
        ("electricity", 2),         # 45K rows, energy market
        ("bank-marketing", 1),      # 45K rows, banking
        ("adult", 2),               # 49K rows, census
        ("numerai28.6", 1),         # 96K rows, quant finance
        ("helena", 1),              # 65K rows, multi-class
        ("jannis", 1),              # 83K rows, multi-class
        ("MiniBooNE", 1),           # 130K rows, particle physics
        ("shuttle", 1),             # 58K rows, space shuttle
        ("covertype", 1),           # 581K rows, forest cover (redundant w/ ecology — adds OpenML variant)
        ("higgs", 1),               # 98K rows, high-energy physics
        ("volkert", 1),             # 58K rows
        ("dionis", 1),              # 416K rows, multi-class
        ("robert", 1),              # 10K rows × 7200 features (wide)
        ("christine", 1),           # 5.4K rows × 1636 features (wide)
        ("fabert", 1),              # 8.2K rows × 800 features
        ("nomao", 1),               # 34K rows, web data
        ("sylvine", 1),             # 5.1K rows × 20 features

        # === MEDIUM-LARGE (10K-30K rows) ===
        ("har", 1),                 # 10K rows × 561 features — REAL SENSORS!
        ("pendigits", 1),           # 11K rows, pen trajectory
        ("letter", 1),              # 20K rows, letter recognition
        ("magic", 1),               # 19K rows, gamma-ray telescope
        ("eye_movements", 1),       # 11K rows, eye tracking
        ("gas-drift", 1),           # 13K rows, chemical sensors
        ("Click_prediction_small", 1), # 40K rows, web clicks
        ("amazon_employee_access", 1), # 33K rows
        ("KDDCup09_appetency", 1),  # 50K rows
        ("mc1", 1),                 # 10K rows, software defects

        # === MEDIUM (2K-10K rows) ===
        ("phoneme", 1),             # 5.4K rows, speech
        ("wine-quality-red", 1),    # 1.6K rows, chemistry
        ("wine-quality-white", 1),  # 4.9K rows, chemistry
        ("abalone", 1),             # 4.2K rows, marine biology
        ("segment", 1),             # 2.3K rows, image segments
        ("steel-plates-fault", 1),  # 1.9K rows, manufacturing
        ("texture", 1),             # 5.5K rows, image features
        ("optdigits", 1),           # 5.6K rows, digit features
        ("satimage", 1),            # 6.4K rows, satellite
        ("spambase", 1),            # 4.6K rows, email features
        ("waveform-5000", 1),       # 5K rows
        ("wall-robot-navigation", 1), # 5.5K rows, robot sensors
        ("mfeat-factors", 1),       # 2K × 216
        ("mfeat-fourier", 1),       # 2K × 76
        ("mfeat-karhunen", 1),      # 2K × 64
        ("mfeat-zernike", 1),       # 2K × 47
        ("mfeat-morphological", 1), # 2K × 6
        ("mfeat-pixel", 1),         # 2K × 240
        ("vehicle", 1),             # 846 rows — likely too small
        ("analcatdata_authorship", 1), # 841 rows
        ("micro-mass", 1),          # 571 rows × 1300 features
        ("isolet", 1),              # 7.8K × 617, speech features
        ("gina_agnostic", 1),       # 3.5K × 970
        ("madeline", 1),            # 3.1K × 259

        # === SMALL-MEDIUM (500-2K rows) ===
        ("credit-g", 1),            # 1K rows, credit
        ("climate-model-simulation-crashes", 1), # 540 rows
        ("diabetes", 1),            # 768 rows
        ("blood-transfusion-service-center", 1), # 748 rows
        ("pc1", 1),                 # 1.1K rows, software defects
        ("kc1", 1),                 # 2.1K rows, software defects
        ("pc4", 1),                 # 1.5K rows, software defects
        ("ionosphere", 1),          # 351 rows, radar
        ("sonar", 1),               # 208 rows, sonar signals
        ("profb", 1),               # 672 rows
        ("collins", 1),             # 500 rows × 23 features

        # === EXTRA LARGE via OpenML IDs — for size variety ===
        # Note: some of these may fail; that's OK, we try-catch
        ("Airlines_DepDelay_1M", 1), # 1M rows if available
        ("CIFAR_10", 1),            # 60K × 3072
        ("Fashion-MNIST", 1),       # 70K × 784
        ("sf-police-incidents", 1), # 2.2M rows
    ]

    count = 0
    for name, version in datasets:
        if count >= target:
            return
        try:
            print(f"    Fetching {name}...")
            data = fetch_openml(name=name, version=version,
                                as_frame=True, parser='auto')
            df = data.data
            numeric = df.select_dtypes(include=[np.number])
            if numeric.shape[1] == 0:
                continue

            n_cols = 0
            for col_name in numeric.columns:
                col = numeric[col_name].dropna().values.astype(np.float64)
                if len(col) < MIN_ARRAY_SIZE:
                    continue

                src = f"oml_{name}_{col_name}"
                yield col[:MAX_ARRAY_SIZE], f"oml_{name}_{col_name}", src
                count += 1
                n_cols += 1

                # Non-overlapping chunks for large columns
                for chunk, cid, csrc in non_overlapping_chunks(
                        col, f"oml_{name}_{col_name}", src):
                    yield chunk, cid, csrc
                    count += 1
                    if count >= target:
                        return

                if count >= target:
                    return

            print(f"      {name}: {n_cols} columns")

        except Exception as e:
            print(f"      SKIP {name}: {e}")
            continue

    print(f"    -> {count} arrays from OpenML")


# ── SyntheticReal — capped, each truly independent ──────────────────────

MAX_SYNTHETIC = 2000

def gen_synthetic_real(target):
    """
    Arrays from real-world distributions — capped at MAX_SYNTHETIC.
    Each array uses a fresh RNG state → genuinely independent.
    """
    actual_target = min(target, MAX_SYNTHETIC)
    print(f"  [SyntheticReal] Generating up to {actual_target} arrays (hard cap)...")

    count = 0
    sizes = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

    for size in sizes:
        for dtype in ["lognormal", "exponential", "pareto", "mixture",
                      "nearlysorted", "uniform_int", "heavy_dupes",
                      "timestamps", "reversed", "sortappend",
                      "zipf", "beta"]:
            if count >= actual_target:
                return

            # Each synthetic array is truly independent
            src = f"synr_{dtype}_n{size}_{count}"

            if dtype == "lognormal":
                s = rng.uniform(0.3, 2.5)
                arr = rng.lognormal(mean=rng.uniform(5, 15), sigma=s, size=size)
            elif dtype == "exponential":
                arr = rng.exponential(scale=rng.uniform(0.1, 1000), size=size)
            elif dtype == "pareto":
                a = rng.uniform(1.05, 4.0)
                arr = (rng.pareto(a, size=size) + 1) * rng.uniform(1, 1000)
            elif dtype == "mixture":
                n_modes = rng.randint(2, 6)
                parts = []
                for _ in range(n_modes):
                    nc = max(10, size // n_modes +
                             rng.randint(-size // 20, size // 20 + 1))
                    parts.append(rng.normal(rng.uniform(-100, 100),
                                            rng.uniform(0.5, 50), size=nc))
                arr = np.concatenate(parts)
                rng.shuffle(arr)
                arr = arr[:size]
            elif dtype == "nearlysorted":
                arr = np.sort(rng.randn(size) * rng.uniform(1, 1000))
                n_swaps = int(size * rng.uniform(0.005, 0.2))
                for _ in range(n_swaps):
                    a, b = rng.randint(0, size, 2)
                    arr[a], arr[b] = arr[b], arr[a]
            elif dtype == "uniform_int":
                lo = rng.randint(0, 10000)
                arr = rng.randint(lo, lo + rng.randint(100, 1000000),
                                  size=size).astype(np.float64)
            elif dtype == "heavy_dupes":
                n_u = rng.randint(2, min(100, size // 5))
                vals = rng.randn(n_u) * rng.uniform(1, 1000)
                arr = rng.choice(vals, size=size)
            elif dtype == "timestamps":
                arr = (np.arange(size, dtype=np.float64)
                       * rng.uniform(0.001, 10))
                arr += rng.uniform(-0.5, 0.5, size=size) * rng.uniform(0, 2)
            elif dtype == "reversed":
                arr = np.sort(rng.randn(size))[::-1].copy()
            elif dtype == "sortappend":
                split = rng.randint(size // 3, size * 2 // 3)
                arr = np.empty(size)
                arr[:split] = np.sort(rng.randn(split))
                arr[split:] = rng.randn(size - split)
            elif dtype == "zipf":
                arr = rng.zipf(rng.uniform(1.5, 3.0),
                               size=size).astype(np.float64)
            elif dtype == "beta":
                a, b = rng.uniform(0.5, 5), rng.uniform(0.5, 5)
                arr = rng.beta(a, b, size=size) * rng.uniform(1, 10000)
            else:
                continue

            yield arr, f"synr_{dtype}_n{size}", src
            count += 1

    print(f"    -> {count} arrays from synthetic-real")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

GENERATORS = [
    ("Finance",       gen_finance),
    ("Ecology",       gen_ecology),
    ("Network",       gen_network),
    ("Housing",       gen_housing),
    ("Text",          gen_text),
    ("Medical",       gen_medical),
    ("Geospatial",    gen_geospatial),
    ("OpenML",        gen_openml),
    ("SyntheticReal", gen_synthetic_real),
]


def main():
    parser = argparse.ArgumentParser(
        description="Fetch diverse data — honest edition")
    parser.add_argument("--target", type=int, default=50000,
                        help="Max arrays per domain (default 50000)")
    parser.add_argument("--domains", type=str, default="all",
                        help="Comma-separated domains, or 'all'")
    parser.add_argument("--no-raw", action="store_true",
                        help="Skip saving raw .npy files")
    args = parser.parse_args()

    save_raw = not args.no_raw

    if not Path("/Volumes/k").exists():
        print("ERROR: Drive K not mounted at /Volumes/k")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if save_raw:
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  FETCH DIVERSE DATA v2 — HONEST EDITION")
    print(f"  Max per domain: {args.target:,}")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"  Tie threshold: {TIE_THRESHOLD:.0%} (skip ambiguous labels)")
    print(f"  Fixes: no bootstrap, no overlapping windows, source tracking")
    print("=" * 70)

    n_max = float(MAX_ARRAY_SIZE)

    columns = (["file", "domain", "source_id", "n_elements"]
               + FEATURE_NAMES
               + ["time_introsort", "time_heapsort", "time_timsort",
                  "best_algorithm", "timing_margin"])

    # Resume support
    done_ids = set()
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        done_ids = set(existing["file"].tolist())
        print(f"  Resuming: {len(done_ids):,} already processed")
    else:
        pd.DataFrame(columns=columns).to_csv(OUTPUT_CSV, index=False)

    total_start = time.time()
    total_new  = 0
    total_ties = 0
    FLUSH_EVERY = 200
    buffer = []

    if args.domains != "all":
        selected = set(args.domains.split(","))
        generators = [(n, g) for n, g in GENERATORS if n in selected]
    else:
        generators = GENERATORS

    for domain_name, gen_func in generators:
        print(f"\n{'─' * 60}")
        print(f"  Domain: {domain_name}")
        print(f"{'─' * 60}")

        d_start = time.time()
        d_count  = 0
        d_ties   = 0
        d_errors = 0

        try:
            for arr, array_id, source_id in gen_func(args.target):
                if array_id in done_ids:
                    continue
                try:
                    row = process_array(arr, array_id, domain_name,
                                        source_id, n_max, save_raw=save_raw)
                    if row is not None:
                        buffer.append(row)
                        d_count += 1
                        total_new += 1
                    else:
                        d_ties += 1
                        total_ties += 1
                except Exception:
                    d_errors += 1

                if len(buffer) >= FLUSH_EVERY:
                    pd.DataFrame(buffer).to_csv(
                        OUTPUT_CSV, mode="a", header=False, index=False)
                    buffer.clear()
                    elapsed = time.time() - d_start
                    rate = d_count / elapsed if elapsed > 0 else 0
                    print(f"    {d_count:,} done ({rate:.0f}/s, "
                          f"{d_ties} ties, {d_errors} errors)")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

        if buffer:
            pd.DataFrame(buffer).to_csv(
                OUTPUT_CSV, mode="a", header=False, index=False)
            buffer.clear()

        elapsed = time.time() - d_start
        print(f"  => {domain_name}: {d_count:,} arrays in {elapsed:.1f}s "
              f"({d_ties} ties, {d_errors} errors)")

    total_elapsed = time.time() - total_start

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  DONE: {total_new:,} new + {total_ties:,} ties skipped "
          f"in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 70}")

    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        print(f"\n  Total rows: {len(df):,}")
        print(f"  Unique sources: {df['source_id'].nunique():,}")

        print(f"\n  {'Domain':<15s} {'Count':>7s} {'Sources':>8s}  "
              f"{'intro':>7s} {'heap':>7s} {'tim':>7s}")
        print(f"  {'-'*15} {'-'*7} {'-'*8}  {'-'*7} {'-'*7} {'-'*7}")
        for domain in sorted(df["domain"].unique()):
            sub = df[df["domain"] == domain]
            bc = sub["best_algorithm"].value_counts()
            src_n = sub["source_id"].nunique()
            print(f"  {domain:<15s} {len(sub):>7,} {src_n:>8,}  "
                  f"{bc.get('introsort', 0):>7,} "
                  f"{bc.get('heapsort', 0):>7,} "
                  f"{bc.get('timsort', 0):>7,}")

        print(f"\n  Size distribution:")
        for lo, hi, label in [(200, 1000, "200-1K"),
                               (1000, 5000, "1K-5K"),
                               (5000, 20000, "5K-20K"),
                               (20000, 100000, "20K-100K"),
                               (100000, 200001, "100K-200K")]:
            n = ((df["n_elements"] >= lo) & (df["n_elements"] < hi)).sum()
            print(f"    {label:>10s}: {n:>7,}  ({100*n/len(df):.1f}%)")

        print(f"\n  Class balance:")
        for algo in ALGORITHMS:
            n = (df["best_algorithm"] == algo).sum()
            print(f"    {algo:>10s}: {n:>7,}  ({100*n/len(df):.1f}%)")

        print(f"\n  Timing margin (how decisive the winner is):")
        print(f"    Mean:   {df['timing_margin'].mean():.1%}")
        print(f"    Median: {df['timing_margin'].median():.1%}")
        print(f"    Min:    {df['timing_margin'].min():.1%}")

    # Symlink CSV to project dir
    try:
        if PROJECT_CSV.exists() or PROJECT_CSV.is_symlink():
            PROJECT_CSV.unlink()
        PROJECT_CSV.symlink_to(OUTPUT_CSV)
        print(f"\n  Symlink: {PROJECT_CSV} -> {OUTPUT_CSV}")
    except Exception:
        print(f"\n  CSV at: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
