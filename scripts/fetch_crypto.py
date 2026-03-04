#!/usr/bin/env python3
"""
fetch_crypto.py  ·  Pure Crypto Data Fetcher (Step 1 only)
===========================================================
Fetches raw cryptocurrency arrays via yfinance and saves them as plain .csv files.
NO feature extraction, NO timing — just raw data capture.

Design principles (same as fetch_f1_10k.py, fetch_stock.py, fetch_weather.py):
  • Crash-safe: index CSV appended after EVERY array save.
  • Resumable: on restart, reads existing index to skip already-fetched files.
  • Saves into data/real_world_10k/ alongside F1, Stock, Weather data.

Why crypto adds unique structural patterns:
  • 24/7 trading — no market-close gaps (stock has daily gaps → different run patterns)
  • Extreme kurtosis — fat tails, much heavier than stock or F1
  • Flash crashes / pumps — sharp spikes both directions (high outlier_ratio)
  • Young coins — very short history = small arrays
  • Stablecoins — near-constant price + micro noise (like pressure but different)
  • Meme coins — extreme volatility + pump-and-dump patterns
  • Volume patterns — 24/7 with massive weekend/weekday differences

Anti-bias: Structural transforms for balanced feature-space coverage:
  REV, SHUF, QBIN50, PSORT10 → ×5 arrays per raw array.

Output (shared with F1 + Stock + Weather):
  data/real_world_10k/raw/       → .csv files (one number per line)
  data/real_world_10k/index.csv  → metadata index (appended incrementally)

Usage:
    source venv/bin/activate
    python scripts/fetch_crypto.py                    # default 50K target
    python scripts/fetch_crypto.py --target 30000     # custom target
    python scripts/fetch_crypto.py --dry-run           # estimate only
    python scripts/fetch_crypto.py --no-transforms     # raw only
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

import yfinance as yf

# ── Paths (SHARED with all domains) ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "real_world_10k" / "raw"
INDEX_CSV    = PROJECT_ROOT / "data" / "real_world_10k" / "index.csv"

# ── Config ────────────────────────────────────────────────────────────────
MIN_ARRAY_LEN = 50

# Time periods — crypto is 24/7 so hourly data is continuous (no gaps!)
PERIODS = [
    ("1y",  "1d"),    # 1 year daily    → ~365 points
    ("2y",  "1d"),    # 2 years daily   → ~730 points
    ("5y",  "1d"),    # 5 years daily   → ~1825 points
    ("max", "1d"),    # full history    → varies, BTC since 2014 = ~4000
    ("3mo", "1d"),    # 3 months daily  → ~90 points
    ("6mo", "1d"),    # 6 months daily  → ~180 points
    ("10y", "1d"),    # 10 years daily  → ~3650 points (old coins only)
    ("2y",  "1h"),    # 2 years hourly  → ~17520 points (24/7!)
    ("1y",  "1h"),    # 1 year hourly   → ~8760 points (24/7!)
    ("3mo", "1h"),    # 3 months hourly → ~2160 points
    ("5d",  "5m"),    # 5 days 5-min    → ~1440 points (24/7!)
    ("1mo", "5m"),    # 1 month 5-min   → ~8640 points (24/7!)
    ("1mo", "1h"),    # 1 month hourly  → ~720 points
]

# Raw OHLCV columns
RAW_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Derived columns — same logic as stock but crypto has unique patterns
DERIVED_COLUMNS = [
    "log_returns",
    "pct_change",
    "daily_range",
    "rolling_5d_mean",
    "rolling_20d_mean",
    "rolling_5d_std",
    "rolling_20d_std",
    "typical_price",
    "momentum_5d",
    "volume_change",
]

# ── Crypto tickers ───────────────────────────────────────────────────────
# yfinance uses -USD suffix for crypto pairs
CRYPTO_TICKERS = {
    # Major — long history, high liquidity
    "major": [
        "BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD", "BCH-USD",
        "ADA-USD", "DOT-USD", "LINK-USD", "BNB-USD", "SOL-USD",
        "DOGE-USD", "AVAX-USD", "MATIC-USD", "UNI-USD", "ATOM-USD",
        "XLM-USD", "ALGO-USD", "FIL-USD", "VET-USD", "NEAR-USD",
    ],
    # DeFi — different trading patterns (governance tokens, yield farming)
    "defi": [
        "AAVE-USD", "MKR-USD", "COMP-USD", "SNX-USD", "CRV-USD",
        "SUSHI-USD", "YFI-USD", "UMA-USD", "BAL-USD", "LDO-USD",
        "FXS-USD", "LQTY-USD", "RPL-USD", "PENDLE-USD",
    ],
    # Stablecoins — near-constant, tight peg (unique: very low entropy,
    # ultra-high adj_sorted, micro-noise)
    "stablecoins": [
        "USDT-USD", "USDC-USD", "DAI-USD", "BUSD-USD", "TUSD-USD",
        "USDP-USD", "FRAX-USD", "GUSD-USD",
    ],
    # Meme / volatile — pump-and-dump, extreme outliers
    "meme": [
        "SHIB-USD", "PEPE24478-USD", "FLOKI-USD", "BONK-USD",
        "WIF-USD", "TURBO-USD", "BABYDOGE-USD",
    ],
    # Layer 2 / scaling — newer, shorter history
    "layer2": [
        "ARB11841-USD", "OP-USD", "IMX-USD", "STRK-USD",
        "MANTA-USD", "BLAST-USD", "ZK-USD",
    ],
    # Older / mid-cap — different eras of crypto
    "midcap": [
        "EOS-USD", "TRX-USD", "XTZ-USD", "NEO-USD", "DASH-USD",
        "ZEC-USD", "XMR-USD", "ETC-USD", "IOTA-USD", "THETA-USD",
        "FTM-USD", "HBAR-USD", "EGLD-USD", "SAND-USD", "MANA-USD",
        "AXS-USD", "ENJ-USD", "CHZ-USD", "GALA-USD", "ICP-USD",
    ],
    # Exchange tokens — platform-native, correlated with exchange volume
    "exchange": [
        "CRO-USD", "OKB-USD", "LEO-USD", "KCS-USD", "GT-USD", "HT-USD",
    ],
    # AI / DePIN — newer narrative, unique trading patterns
    "ai_depin": [
        "FET-USD", "RNDR-USD", "WLD-USD", "ARKM-USD", "AR-USD",
        "GRT-USD", "OCEAN-USD", "AKT-USD", "TAO22974-USD",
    ],
    # Gaming / Metaverse — play-to-earn dynamics
    "gaming": [
        "RONIN-USD", "PYR-USD", "MAGIC-USD", "ALICE-USD",
        "SUPER8290-USD", "GODS-USD", "PIXEL-USD", "PORTAL-USD",
    ],
    # Newer L1s (2023-2024 launches)
    "newer_l1": [
        "SUI20947-USD", "APT21794-USD", "SEI-USD", "INJ-USD",
        "TIA22861-USD", "PYTH-USD", "JUP29210-USD", "ONDO-USD",
        "ENA-USD", "W-USD",
    ],
    # Infrastructure & Storage
    "infra": [
        "STORJ-USD", "SC-USD", "HNT-USD", "IOTX-USD", "ANKR-USD",
        "CELO-USD", "RLC-USD", "BAND-USD", "CTSI-USD", "NKN-USD",
    ],
    # Classic alts — long history = lots of data, diverse market cycles
    "classic": [
        "LRC-USD", "ZIL-USD", "ONE-USD", "QTUM-USD", "ICX-USD",
        "ONT-USD", "WAVES-USD", "KSM-USD", "KAVA-USD", "CKB-USD",
        "HOT-USD", "SKL-USD", "CELR-USD", "MTL-USD", "DENT-USD",
        "STMX-USD", "SLP-USD", "C98-USD", "SPELL-USD", "JOE-USD",
    ],
}

# Rate limiting — yfinance is generous
SLEEP_BETWEEN_TICKERS = 0.3
SLEEP_ON_ERROR        = 10.0

# ── CSV columns (shared index) ───────────────────────────────────────────
INDEX_COLUMNS = [
    "array_id", "file", "year", "event", "round", "session",
    "driver", "lap", "channel", "n_elements", "dtype", "size_bytes",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Index CSV — incremental append
# ═══════════════════════════════════════════════════════════════════════════

def init_output() -> tuple[int, set[str]]:
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
        with open(INDEX_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
            writer.writeheader()
        return 0, set()


def append_index_row(row: dict) -> None:
    with open(INDEX_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════
#  Structural transforms — anti-bias
# ═══════════════════════════════════════════════════════════════════════════

def apply_transforms(arr: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}

    out["REV"] = arr[::-1].copy()

    shuf = arr.copy()
    rng.shuffle(shuf)
    out["SHUF"] = shuf

    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        edges = np.linspace(vmin, vmax, 51)
        idx = np.clip(np.digitize(arr, edges) - 1, 0, 49)
        centers = (edges[:-1] + edges[1:]) / 2.0
        out["QBIN50"] = centers[idx]

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
#  Sub-array windowing — extract more training arrays from large sequences
# ═══════════════════════════════════════════════════════════════════════════

WINDOW_SIZES = [250, 500, 1000, 2500, 5000]
MAX_WINDOWS_PER_SIZE = 3


def generate_windows(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Split large arrays into non-overlapping sub-arrays of various sizes.
    Each window captures a different temporal segment with different structural
    properties (volatility regime, trend direction, etc.)."""
    windows = {}
    n = len(arr)
    for ws in WINDOW_SIZES:
        if n < ws * 2:  # need at least 2× window size
            continue
        n_chunks = min(n // ws, MAX_WINDOWS_PER_SIZE)
        for i in range(n_chunks):
            chunk = arr[i * ws : (i + 1) * ws]
            if len(chunk) >= MIN_ARRAY_LEN:
                windows[f"w{ws}c{i}"] = chunk
    return windows


# ═══════════════════════════════════════════════════════════════════════════
#  Derived columns
# ═══════════════════════════════════════════════════════════════════════════

def compute_derived(df: pd.DataFrame) -> dict[str, np.ndarray]:
    derived = {}

    if "Close" in df.columns:
        close = df["Close"].dropna().values.astype(np.float64)
        if len(close) >= MIN_ARRAY_LEN:
            if len(close) > 1:
                lr = np.diff(np.log(np.maximum(close, 1e-10)))
                if len(lr) >= MIN_ARRAY_LEN:
                    derived["log_returns"] = lr

            if len(close) > 1:
                pct = np.diff(close) / np.maximum(close[:-1], 1e-10)
                if len(pct) >= MIN_ARRAY_LEN:
                    derived["pct_change"] = pct

            s = pd.Series(close)
            for window, name in [(5, "rolling_5d_mean"), (20, "rolling_20d_mean")]:
                r = s.rolling(window).mean().dropna().values
                if len(r) >= MIN_ARRAY_LEN:
                    derived[name] = r.astype(np.float64)

            for window, name in [(5, "rolling_5d_std"), (20, "rolling_20d_std")]:
                r = s.rolling(window).std().dropna().values
                if len(r) >= MIN_ARRAY_LEN:
                    derived[name] = r.astype(np.float64)

    if "High" in df.columns and "Low" in df.columns:
        high = df["High"].dropna().values.astype(np.float64)
        low = df["Low"].dropna().values.astype(np.float64)
        min_len = min(len(high), len(low))
        if min_len >= MIN_ARRAY_LEN:
            derived["daily_range"] = high[:min_len] - low[:min_len]

    # Typical price: (H+L+C)/3 — smoother representation
    if all(c in df.columns for c in ["High", "Low", "Close"]):
        h = df["High"].dropna().values.astype(np.float64)
        l = df["Low"].dropna().values.astype(np.float64)
        c = df["Close"].dropna().values.astype(np.float64)
        min3 = min(len(h), len(l), len(c))
        if min3 >= MIN_ARRAY_LEN:
            derived["typical_price"] = (h[:min3] + l[:min3] + c[:min3]) / 3.0

    # Momentum: Close(t) - Close(t-5)
    if "Close" in df.columns:
        close = df["Close"].dropna().values.astype(np.float64)
        if len(close) > 5:
            mom = close[5:] - close[:-5]
            if len(mom) >= MIN_ARRAY_LEN:
                derived["momentum_5d"] = mom

    # Volume change
    if "Volume" in df.columns:
        vol = df["Volume"].dropna().values.astype(np.float64)
        if len(vol) > 1:
            vc = np.diff(vol)
            if len(vc) >= MIN_ARRAY_LEN:
                derived["volume_change"] = vc

    return derived


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
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_ARRAY_LEN:
        return counter, total_bytes

    fname = f"crypto_{ticker}_{period}_{interval}_{column}.csv"

    if fname in existing_files:
        return counter, total_bytes

    np.savetxt(OUTPUT_DIR / fname, arr, fmt="%.10g")
    nbytes = (OUTPUT_DIR / fname).stat().st_size

    row = {
        "array_id": counter,
        "file": fname,
        "year": 0,
        "event": ticker,
        "round": 0,
        "session": period,
        "driver": interval,
        "lap": "CRYPTO",
        "channel": column,
        "n_elements": arr.size,
        "dtype": str(arr.dtype),
        "size_bytes": nbytes,
    }
    append_index_row(row)
    existing_files.add(fname)

    counter += 1
    total_bytes += nbytes

    if counter % 500 == 0:
        print(f"      [{counter:6,} crypto arrays | {total_bytes / (1024**2):,.1f} MB]")

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
    counter, total_bytes = save_array(
        arr, ticker, period, interval, column,
        existing_files, counter, total_bytes,
    )

    seed = hash(f"crypto_{ticker}_{period}_{interval}_{column}") & 0xFFFFFFFF
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
    all_counter, all_existing = init_output()

    crypto_existing = {f for f in all_existing if f.startswith("crypto_")}
    crypto_counter = len(crypto_existing)
    total_bytes = 0
    if INDEX_CSV.exists() and crypto_counter > 0:
        try:
            df = pd.read_csv(INDEX_CSV)
            crypto_df = df[df["file"].str.startswith("crypto_")]
            total_bytes = int(crypto_df["size_bytes"].sum())
        except Exception:
            pass

    # Build unique ticker list
    all_tickers = []
    for group, tickers in CRYPTO_TICKERS.items():
        all_tickers.extend(tickers)
    seen = set()
    unique_tickers = []
    for t in all_tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    print("=" * 70)
    print("Crypto Data Fetcher  —  Step 1 (data collection only)")
    print("=" * 70)
    print(f"  Tickers:        {len(unique_tickers)} (major/defi/stablecoins/meme/L2/midcap)")
    print(f"  Periods:        {len(PERIODS)}")
    print(f"  Columns/ticker: up to {len(RAW_COLUMNS) + len(DERIVED_COLUMNS)} "
          f"({len(RAW_COLUMNS)} raw + {len(DERIVED_COLUMNS)} derived)")
    print(f"  Target:         {target:,} crypto arrays")
    print(f"  Already have:   {crypto_counter:,} crypto arrays ({total_bytes / (1024**2):,.1f} MB)")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  Index CSV:      {INDEX_CSV} (shared)")
    if transforms:
        print(f"  Transforms:     ENABLED (REV, SHUF, QBIN50, PSORT10 → ×5 arrays)")
    else:
        print(f"  Transforms:     DISABLED")
    print("=" * 70)

    if crypto_counter >= target:
        print(f"\nAlready have {crypto_counter:,} crypto arrays >= target {target:,}. Done!")
        return

    def reached_target() -> bool:
        return crypto_counter >= target

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

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                for col in RAW_COLUMNS:
                    if col in df.columns:
                        arr = df[col].dropna().values.astype(np.float64)
                        crypto_counter, total_bytes = save_fn(
                            arr, ticker, period, interval, col,
                            all_existing, crypto_counter, total_bytes,
                        )
                        if reached_target():
                            break
                        # Window large arrays into sub-arrays
                        for wname, warr in generate_windows(arr).items():
                            crypto_counter, total_bytes = save_fn(
                                warr, ticker, period, interval,
                                f"{col}_{wname}",
                                all_existing, crypto_counter, total_bytes,
                            )
                            if reached_target():
                                break
                        if reached_target():
                            break

                if reached_target():
                    break

                derived = compute_derived(df)
                for col_name, arr in derived.items():
                    crypto_counter, total_bytes = save_fn(
                        arr, ticker, period, interval, col_name,
                        all_existing, crypto_counter, total_bytes,
                    )
                    if reached_target():
                        break
                    # Window large derived arrays
                    for wname, warr in generate_windows(arr).items():
                        crypto_counter, total_bytes = save_fn(
                            warr, ticker, period, interval,
                            f"{col_name}_{wname}",
                            all_existing, crypto_counter, total_bytes,
                        )
                        if reached_target():
                            break
                    if reached_target():
                        break

            except Exception as e:
                print(f"    {period}/{interval} failed: {e}")
                time.sleep(SLEEP_ON_ERROR)
                continue

        print(f"    => {crypto_counter:,} crypto arrays | {total_bytes / (1024**2):,.1f} MB")
        time.sleep(SLEEP_BETWEEN_TICKERS)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CRYPTO FETCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Crypto arrays:  {crypto_counter:,}")
    print(f"  Crypto size:    {total_bytes / (1024**2):,.1f} MB")
    print(f"  Output:         {OUTPUT_DIR}")
    print(f"  Index:          {INDEX_CSV}")

    if INDEX_CSV.exists():
        df = pd.read_csv(INDEX_CSV)
        crypto_df = df[df["file"].str.startswith("crypto_")]
        if len(crypto_df) > 0:
            print(f"\n  Crypto breakdown:")
            print(f"    Tickers:     {crypto_df['event'].nunique()}")
            print(f"    Periods:     {crypto_df['session'].value_counts().to_dict()}")
            print(f"    Array sizes: min={crypto_df['n_elements'].min():,}  "
                  f"max={crypto_df['n_elements'].max():,}  "
                  f"mean={crypto_df['n_elements'].mean():,.0f}")

        # Combined totals
        for prefix, label in [("f1_", "F1"), ("stock_", "Stock"),
                               ("weather_", "Weather"), ("crypto_", "Crypto")]:
            c = len(df[df["file"].str.startswith(prefix)])
            if c > 0:
                print(f"    {label:12s} {c:>10,}")
        print(f"    {'TOTAL':12s} {len(df):>10,}")
        print(f"    Total size:  {df['size_bytes'].sum() / (1024**2):,.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch raw crypto arrays via yfinance (Step 1 — data collection only)"
    )
    p.add_argument("--target", type=int, default=100_000,
                   help="Stop after this many crypto arrays (default: 100000)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print estimated count without fetching")
    p.add_argument("--no-transforms", action="store_true",
                   help="Save only raw arrays, skip structural transforms")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        all_t = set()
        for tickers in CRYPTO_TICKERS.values():
            all_t.update(tickers)
        n = len(all_t)
        n_cols = len(RAW_COLUMNS) + len(DERIVED_COLUMNS)
        n_per = len(PERIODS)
        est_raw = n * n_per * n_cols
        est = est_raw * 5
        print("DRY RUN — estimating data volume\n")
        print(f"  Unique tickers:      {n}")
        print(f"  Periods:             {n_per}")
        print(f"  Columns per period:  up to {n_cols}")
        print(f"  Max raw arrays:      {est_raw:,}")
        print(f"  With transforms:     {est:,}  (×5)")
        print(f"  Realistic estimate:  ~{int(est * 0.5):,}  (many young coins lack history)")
        print(f"  Est. disk usage:     ~{est * 0.005:.0f} MB")
        return

    fetch_all(target=args.target, transforms=not args.no_transforms)


if __name__ == "__main__":
    main()
