#!/usr/bin/env python3
"""
fetch_stock.py  ·  Pure Stock Data Fetcher (Step 1 only)
=========================================================
Fetches raw stock market arrays via yfinance and saves them as plain .csv files.
NO feature extraction, NO timing — just raw data capture.

Design principles (same as fetch_f1_10k.py):
  • Crash-safe: index CSV appended after EVERY array save.
  • Resumable: on restart, reads existing index to skip already-fetched files.
  • Saves into data/real_world_10k/ alongside F1 data.

Data sources per ticker:
  - OHLCV columns: Open, High, Low, Close, Volume (5 arrays)
  - Derived: log_returns, pct_change, daily_range (3 arrays)
  - Rolling windows on Close: rolling_5d_mean, rolling_20d_mean,
    rolling_5d_std, rolling_20d_std (4 arrays)
  → Up to 12 arrays per ticker per period

Array diversity vs F1:
  - Close prices:   trending random walk  (adj_sorted ≈ 0.53, low duplicates)
  - Log returns:    symmetric noise       (adj_sorted ≈ 0.49, near-random)
  - Volume:         lognormal spikes      (right-skewed, high outlier_ratio)
  - Rolling means:  very smooth           (high adj_sorted, low entropy)
  - Daily range:    positive noise        (right-skewed, no trend)

Anti-bias: Structural transforms for balanced feature-space coverage
  Problem with raw-only data:
    • Close/Open/High/Low/rolling → mostly sorted → timsort ALWAYS wins
    • log_returns/pct_change      → random        → introsort ALWAYS wins
    • Nothing has high duplicates → heapsort NEVER gets a fair chance
    → Model learns "pick timsort or introsort", ignores heapsort.  Useless.
  Fix — 4 transforms per array push data into under-represented regions:
    • REV      → reversed copy:          flips sortedness  high→low
    • SHUF     → random shuffle:         pure random, kills all runs
    • QBIN50   → quantize to 50 bins:    high duplicate_ratio → heapsort region
    • PSORT10  → sort then perturb 10%:  nearly-sorted from random base
  Result: ×5 data, roughly equal coverage of timsort / introsort / heapsort.
  Run with --no-transforms to skip.

Output (shared with F1):
  data/real_world_10k/raw/       → .csv files (one number per line)
  data/real_world_10k/index.csv  → metadata index (appended incrementally)

Usage:
    source venv/bin/activate
    python scripts/fetch_stock.py                    # default 100K target
    python scripts/fetch_stock.py --target 50000     # custom target
    python scripts/fetch_stock.py --dry-run           # estimate only
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

import yfinance as yf

# ── Paths (SHARED with F1 data) ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = Path("/Users/ahmed/Desktop/thesis_data/balanced_data/raw")
INDEX_CSV    = Path("/Users/ahmed/Desktop/thesis_data/balanced_data/index.csv")

# ── Config ────────────────────────────────────────────────────────────────
MIN_ARRAY_LEN = 50

# Time periods to fetch per ticker (more periods = more arrays with
# different sizes and characteristics)
PERIODS = [
    ("1y",  "1d"),    # 1 year  daily   → ~252 points
    ("2y",  "1d"),    # 2 years daily   → ~504 points
    ("5y",  "1d"),    # 5 years daily   → ~1260 points
    ("10y", "1d"),    # 10 years daily  → ~2520 points
    ("max", "1d"),    # max history     → varies, up to 10K+
    ("2y",  "1h"),    # 2 years hourly  → ~3276 points (yfinance limit)
    ("5d",  "5m"),    # 5 days 5-min    → ~390 points
    ("1mo", "5m"),    # 1 month 5-min   → ~1560 points
]

# Raw OHLCV columns from yfinance
RAW_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Derived columns we compute
DERIVED_COLUMNS = [
    "log_returns",       # log(close[t] / close[t-1]) — symmetric noise
    "pct_change",        # (close[t] - close[t-1]) / close[t-1]
    "daily_range",       # high - low — positive, right-skewed
    "rolling_5d_mean",   # 5-period rolling mean of close — very smooth
    "rolling_20d_mean",  # 20-period rolling mean of close — even smoother
    "rolling_5d_std",    # 5-period rolling std — volatility clusters
    "rolling_20d_std",   # 20-period rolling std — smooth volatility
]

# S&P 500 + NASDAQ-100 + global tickers for maximum coverage
# We use a broad list to get thousands of tickers
# yfinance can handle any ticker symbol
TICKER_LISTS = {
    # Major US indices members (representative sample)
    "sp500_sample": [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "UNH",
        "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
        "LLY", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "WMT", "CSCO",
        "ACN", "ABT", "DHR", "NEE", "LIN", "TXN", "PM", "UNP", "RTX",
        "AMGN", "LOW", "HON", "IBM", "COP", "SPGI", "CAT", "GE", "BA",
        "AMAT", "BKNG", "SBUX", "PLD", "ADP", "MDLZ", "GILD", "MMC",
        "ISRG", "ADI", "VRTX", "REGN", "TJX", "SYK", "LRCX", "CB",
        "ZTS", "BDX", "CI", "MMM", "SO", "DUK", "PGR", "MO", "CME",
        "CL", "ITW", "SLB", "USB", "BSX", "TMUS", "EQIX", "AON", "WM",
        "SCHW", "FIS", "HUM", "ICE", "GD", "EMR", "NOC", "MCK", "PNC",
        "CCI", "PSA", "APD", "ORLY", "NSC", "ATVI", "AZO", "KLAC", "SNPS",
        "SHW", "ROP", "ADSK", "AJG", "CMG", "TDG", "MSCI", "MCHP", "CDNS",
    ],
    # Tech & growth
    "tech": [
        "TSLA", "AMD", "INTC", "QCOM", "MU", "NOW", "PANW", "CRWD",
        "SNOW", "DDOG", "ZS", "NET", "FTNT", "WDAY", "TEAM", "SPLK",
        "OKTA", "TTD", "BILL", "HUBS", "TWLO", "SQ", "SHOP", "SPOT",
        "ROKU", "SNAP", "PINS", "UBER", "LYFT", "DASH", "ABNB", "COIN",
        "PLTR", "PATH", "U", "RBLX", "DKNG", "HOOD", "SOFI", "AFRM",
    ],
    # Finance & banking
    "finance": [
        "GS", "MS", "C", "BAC", "WFC", "AXP", "BLK", "PYPL", "T",
        "VZ", "CMCSA", "DIS", "NFLX", "PARA", "WBD", "LYV", "MTCH",
    ],
    # International / ETFs (different market dynamics)
    "global": [
        "EWJ", "FXI", "EWZ", "EWG", "EWU", "EWA", "EWC", "EWY", "EWT",
        "INDA", "VWO", "EEM", "IEMG", "SPY", "QQQ", "IWM", "DIA", "VTI",
        "EFA", "VEA", "GLD", "SLV", "USO", "UNG", "TLT", "HYG", "LQD",
    ],
    # Crypto ETFs & volatile stocks
    "volatile": [
        "BITO", "GBTC", "MSTR", "MARA", "RIOT", "CLSK", "HUT",
        "GME", "AMC", "BBBY", "BB", "NOK", "SPCE", "WKHS", "CLOV",
    ],
    # Commodities & resources
    "commodities": [
        "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB", "XLY",
        "XLRE", "BHP", "RIO", "VALE", "FCX", "NEM", "GOLD", "WPM",
    ],
}

# Rate limiting
SLEEP_BETWEEN_TICKERS = 0.3   # seconds between tickers (yfinance is generous)
SLEEP_ON_ERROR        = 10.0  # seconds to wait on error

# ── CSV columns (shared with F1 index) ───────────────────────────────────
INDEX_COLUMNS = [
    "array_id", "file", "year", "event", "round", "session",
    "driver", "lap", "channel", "n_elements", "dtype", "size_bytes",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Index CSV — incremental append (reuses F1 index)
# ═══════════════════════════════════════════════════════════════════════════

def init_output() -> tuple[int, set[str]]:
    """
    Create output dirs + CSV header if they don't exist.
    Returns (current_count, set_of_existing_filenames) for resume.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)

    if INDEX_CSV.exists():
        try:
            df = pd.read_csv(INDEX_CSV)
            existing = set(df["file"].tolist())
            count = len(df)
            return count, existing
        except Exception:
            return 0, set()
    else:
        # Write header
        with open(INDEX_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
            writer.writeheader()
        return 0, set()


def append_index_row(row: dict) -> None:
    """Append a single row to the index CSV immediately."""
    with open(INDEX_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════
#  Core: compute derived columns from OHLCV data
# ═══════════════════════════════════════════════════════════════════════════

def compute_derived(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Compute derived arrays from OHLCV data. Returns {name: array}."""
    derived = {}

    if "Close" in df.columns:
        close = df["Close"].dropna().values.astype(np.float64)

        if len(close) >= MIN_ARRAY_LEN:
            # Log returns: log(p[t] / p[t-1])
            if len(close) > 1:
                lr = np.diff(np.log(np.maximum(close, 1e-10)))
                if len(lr) >= MIN_ARRAY_LEN:
                    derived["log_returns"] = lr

            # Percent change
            if len(close) > 1:
                pct = np.diff(close) / np.maximum(close[:-1], 1e-10)
                if len(pct) >= MIN_ARRAY_LEN:
                    derived["pct_change"] = pct

            # Rolling means (pandas for simplicity, extract numpy)
            s = pd.Series(close)
            for window, name in [(5, "rolling_5d_mean"), (20, "rolling_20d_mean")]:
                r = s.rolling(window).mean().dropna().values
                if len(r) >= MIN_ARRAY_LEN:
                    derived[name] = r.astype(np.float64)

            # Rolling stds
            for window, name in [(5, "rolling_5d_std"), (20, "rolling_20d_std")]:
                r = s.rolling(window).std().dropna().values
                if len(r) >= MIN_ARRAY_LEN:
                    derived[name] = r.astype(np.float64)

    if "High" in df.columns and "Low" in df.columns:
        high = df["High"].dropna().values.astype(np.float64)
        low = df["Low"].dropna().values.astype(np.float64)
        min_len = min(len(high), len(low))
        if min_len >= MIN_ARRAY_LEN:
            daily_range = high[:min_len] - low[:min_len]
            derived["daily_range"] = daily_range

    return derived


# ═══════════════════════════════════════════════════════════════════════════
#  Structural transforms — anti-bias
# ═══════════════════════════════════════════════════════════════════════════

def apply_transforms(arr: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """
    Generate 4 structural transforms of an array to cover the full 16-feature space.

    Without transforms, stock data clusters in 2 narrow regions:
      • Sorted/smooth → timsort always wins
      • Random noise  → introsort always wins
      • Heapsort never gets a fair chance (no high-duplicate arrays)

    Each transform targets an under-represented region of the feature space:
      REV      — reverses sortedness direction (high→low or low→high)
      SHUF     — full random permutation → adj_sorted ≈ 0.5, low runs
      QBIN50   — quantize to 50 bins → high duplicate_ratio, heapsort territory
      PSORT10  — sort then perturb 10% → nearly sorted, strong timsort territory

    Returns dict {transform_name: transformed_array}.
    """
    rng = np.random.default_rng(seed)
    out = {}

    # 1. Reversed — O(n), flips adj_sorted_ratio from X to (1−X)
    out["REV"] = arr[::-1].copy()

    # 2. Fully shuffled — destroys all structure, pure randomness
    shuf = arr.copy()
    rng.shuffle(shuf)
    out["SHUF"] = shuf

    # 3. Quantized to 50 bins — creates many duplicates (high duplicate_ratio)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        edges = np.linspace(vmin, vmax, 51)
        idx = np.clip(np.digitize(arr, edges) - 1, 0, 49)
        centers = (edges[:-1] + edges[1:]) / 2.0
        out["QBIN50"] = centers[idx]

    # 4. Nearly sorted — sort then randomly swap 10% of positions
    sorted_arr = np.sort(arr.copy())
    n = len(sorted_arr)
    n_swap = max(2, int(n * 0.10))
    swap_idx = rng.choice(n, size=n_swap, replace=False)
    swap_vals = sorted_arr[swap_idx].copy()
    rng.shuffle(swap_vals)
    sorted_arr[swap_idx] = swap_vals
    out["PSORT10"] = sorted_arr

    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Save one array
# ═══════════════════════════════════════════════════════════════════════════

def save_array(
    arr: np.ndarray,
    ticker: str,
    period: str,
    interval: str,
    column: str,
    existing_files: set[str],
    counter: int,
    total_bytes: int,
) -> tuple[int, int]:
    """Save a single array as .csv. Returns (updated_counter, updated_total_bytes)."""

    # Remove NaN/Inf
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_ARRAY_LEN:
        return counter, total_bytes

    # Build filename — reuse F1 naming convention adapted for stock
    # stock_{ticker}_{period}_{interval}_{column}.csv
    fname = f"stock_{ticker}_{period}_{interval}_{column}.csv"

    # Skip if already saved (resume support)
    if fname in existing_files:
        return counter, total_bytes

    # Save as plain CSV (one number per line)
    np.savetxt(OUTPUT_DIR / fname, arr, fmt="%.10g")
    nbytes = (OUTPUT_DIR / fname).stat().st_size

    # Append to shared index CSV — map stock fields to F1 column names
    row = {
        "array_id": counter,
        "file": fname,
        "year": 0,                    # not applicable for stock
        "event": ticker,              # ticker goes in event column
        "round": 0,
        "session": period,            # period goes in session column
        "driver": interval,           # interval goes in driver column
        "lap": "STOCK",               # domain tag
        "channel": column,            # column name = channel
        "n_elements": arr.size,
        "dtype": str(arr.dtype),
        "size_bytes": nbytes,
    }
    append_index_row(row)
    existing_files.add(fname)

    counter += 1
    total_bytes += nbytes

    if counter % 500 == 0:
        print(f"      [{counter:6,} stock arrays | {total_bytes / (1024**2):,.1f} MB]")

    return counter, total_bytes


def save_array_and_transforms(
    arr: np.ndarray,
    ticker: str,
    period: str,
    interval: str,
    column: str,
    existing_files: set[str],
    counter: int,
    total_bytes: int,
) -> tuple[int, int]:
    """Save original array + 4 structural transforms (REV/SHUF/QBIN50/PSORT10)."""
    # Save the original
    counter, total_bytes = save_array(
        arr, ticker, period, interval, column,
        existing_files, counter, total_bytes,
    )

    # Save transforms — deterministic seed from the array identity
    seed = hash(f"{ticker}_{period}_{interval}_{column}") & 0xFFFFFFFF
    transforms = apply_transforms(arr, seed)
    for tname, tarr in transforms.items():
        counter, total_bytes = save_array(
            tarr, ticker, period, interval, f"{column}_{tname}",
            existing_files, counter, total_bytes,
        )

    return counter, total_bytes


# ═══════════════════════════════════════════════════════════════════════════
#  Main fetch loop
# ═══════════════════════════════════════════════════════════════════════════

def fetch_all(target: int = 100_000, transforms: bool = True) -> None:
    """Fetch stock data for all tickers across all periods."""

    # Resume support — count ONLY stock arrays already in shared index
    all_counter, all_existing = init_output()

    # Count stock-specific arrays for progress
    stock_existing = {f for f in all_existing if f.startswith("stock_")}
    stock_counter = len(stock_existing)
    total_bytes = 0
    if INDEX_CSV.exists() and stock_counter > 0:
        try:
            df = pd.read_csv(INDEX_CSV)
            stock_df = df[df["file"].str.startswith("stock_")]
            total_bytes = int(stock_df["size_bytes"].sum())
        except Exception:
            pass

    # Build full ticker list
    all_tickers = []
    for group, tickers in TICKER_LISTS.items():
        all_tickers.extend(tickers)
    # Remove duplicates, preserve order
    seen = set()
    unique_tickers = []
    for t in all_tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    print("=" * 70)
    print("Stock Data Fetcher  —  Step 1 (data collection only)")
    print("=" * 70)
    print(f"  Tickers:        {len(unique_tickers)}")
    print(f"  Periods:        {len(PERIODS)}")
    print(f"  Columns/ticker: up to {len(RAW_COLUMNS) + len(DERIVED_COLUMNS)} "
          f"({len(RAW_COLUMNS)} raw + {len(DERIVED_COLUMNS)} derived)")
    print(f"  Target:         {target:,} stock arrays")
    print(f"  Already have:   {stock_counter:,} stock arrays ({total_bytes / (1024**2):,.1f} MB)")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  Index CSV:      {INDEX_CSV} (shared with F1)")
    if transforms:
        print(f"  Transforms:     ENABLED (REV, SHUF, QBIN50, PSORT10 → ×5 arrays)")
    else:
        print(f"  Transforms:     DISABLED (raw arrays only)")
    print("=" * 70)

    if stock_counter >= target:
        print(f"\nAlready have {stock_counter:,} stock arrays >= target {target:,}. Done!")
        return

    def reached_target() -> bool:
        return stock_counter >= target

    save_fn = save_array_and_transforms if transforms else save_array

    for i, ticker in enumerate(unique_tickers):
        if reached_target():
            break

        print(f"\n  [{i+1}/{len(unique_tickers)}] {ticker}")

        for period, interval in PERIODS:
            if reached_target():
                break

            try:
                df = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    progress=False,
                    timeout=15,
                )

                if df is None or df.empty:
                    continue

                # Flatten multi-level columns if present (yfinance sometimes
                # returns MultiIndex columns with ticker as second level)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Save raw OHLCV columns
                for col in RAW_COLUMNS:
                    if col in df.columns:
                        arr = df[col].dropna().values.astype(np.float64)
                        stock_counter, total_bytes = save_fn(
                            arr, ticker, period, interval, col,
                            all_existing, stock_counter, total_bytes,
                        )
                        if reached_target():
                            break

                if reached_target():
                    break

                # Save derived columns
                derived = compute_derived(df)
                for col_name, arr in derived.items():
                    stock_counter, total_bytes = save_fn(
                        arr, ticker, period, interval, col_name,
                        all_existing, stock_counter, total_bytes,
                    )
                    if reached_target():
                        break

            except Exception as e:
                print(f"    {period}/{interval} failed: {e}")
                time.sleep(SLEEP_ON_ERROR)
                continue

        # Progress per ticker
        print(f"    => {stock_counter:,} stock arrays | {total_bytes / (1024**2):,.1f} MB")
        time.sleep(SLEEP_BETWEEN_TICKERS)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STOCK FETCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Stock arrays:  {stock_counter:,}")
    print(f"  Stock size:    {total_bytes / (1024**2):,.1f} MB")
    print(f"  Output:        {OUTPUT_DIR}")
    print(f"  Index:         {INDEX_CSV}")

    if INDEX_CSV.exists():
        df = pd.read_csv(INDEX_CSV)
        stock_df = df[df["file"].str.startswith("stock_")]
        if len(stock_df) > 0:
            print(f"\n  Stock breakdown:")
            print(f"    Tickers:     {stock_df['event'].nunique()}")
            print(f"    Periods:     {stock_df['session'].value_counts().to_dict()}")
            print(f"    Columns:     {stock_df['channel'].value_counts().to_dict()}")
            print(f"    Array sizes: min={stock_df['n_elements'].min():,}  "
                  f"max={stock_df['n_elements'].max():,}  "
                  f"mean={stock_df['n_elements'].mean():,.0f}")

        # Also show combined totals
        print(f"\n  COMBINED (F1 + Stock):")
        print(f"    Total arrays: {len(df):,}")
        print(f"    Total size:   {df['size_bytes'].sum() / (1024**2):,.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch raw stock market arrays (Step 1 — data collection only)"
    )
    p.add_argument(
        "--target", type=int, default=100_000,
        help="Stop after collecting this many stock arrays (default: 100000)"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print estimated array count without fetching"
    )
    p.add_argument(
        "--no-transforms", action="store_true",
        help="Save only raw arrays, skip structural transforms (REV/SHUF/QBIN50/PSORT10)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        all_tickers = set()
        for tickers in TICKER_LISTS.values():
            all_tickers.update(tickers)
        n_tickers = len(all_tickers)
        n_cols = len(RAW_COLUMNS) + len(DERIVED_COLUMNS)  # 12
        n_periods = len(PERIODS)                           # 8
        est_raw = n_tickers * n_periods * n_cols
        est_with_t = est_raw * 5  # original + 4 transforms
        print("DRY RUN — estimating data volume\n")
        print(f"  Unique tickers:      {n_tickers}")
        print(f"  Periods:             {n_periods}")
        print(f"  Columns per period:  up to {n_cols}")
        print(f"  Max raw arrays:      {est_raw:,}")
        print(f"  With transforms:     {est_with_t:,}  (×5: original + REV/SHUF/QBIN50/PSORT10)")
        print(f"  Realistic estimate:  ~{int(est_with_t * 0.7):,}  (some tickers lack history)")
        print(f"  Est. disk usage:     ~{est_with_t * 0.003:.0f} MB  (avg ~3KB per array)")
        return

    fetch_all(target=args.target, transforms=not args.no_transforms)


if __name__ == "__main__":
    main()
