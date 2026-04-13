#!/usr/bin/env python3
"""
fetch_earthquake.py  ·  Pure Earthquake Data Fetcher (Step 1 only)
===================================================================
Fetches raw seismic data arrays from USGS Earthquake API (free, no key)
and saves them as plain .csv files.
NO feature extraction, NO timing — just raw data capture.

Design principles (same as all other fetchers):
  • Crash-safe: index CSV appended after EVERY array save.
  • Resumable: on restart, reads existing index to skip already-fetched files.
  • Saves into data/real_world_10k/ alongside all other domain data.

Why earthquake data adds unique structural patterns:
  • Magnitudes:       power-law distributed (many small, few large)
                      → very right-skewed, high outlier_ratio
  • Depths:           bimodal (shallow <70km + deep subduction 300-700km)
                      → unusual distribution shape
  • Inter-event times: exponential-like with clustering (aftershocks)
                      → heavily right-skewed, some near-zero runs
  • Latitude/Longitude sequences: NOT random — follow plate boundaries
                      → clustered, unusual run patterns
  • Gap (time between events): Poisson-like but with bursts
  • Cumulative energy: monotone increasing with jumps (big quakes)
  • Event counts per day: integer-valued, Poisson-like

These patterns are COMPLETELY different from F1 (smooth telemetry),
Stock (random walk), Weather (sinusoidal), and Crypto (fat-tailed walk).

USGS API details:
  - URL: https://earthquake.usgs.gov/fdsnws/event/1/query
  - Free, no key, generous limits
  - Max 20,000 events per query → use time windows to get more
  - Returns GeoJSON with magnitude, depth, location, timestamp

Structural transforms: REV, SHUF, QBIN50, PSORT10 → ×5 arrays.

Output (shared):
  data/real_world_10k/raw/       → .csv files (one number per line)
  data/real_world_10k/index.csv  → metadata index (appended incrementally)

Usage:
    source venv/bin/activate
    python scripts/fetch_earthquake.py                    # default 30K target
    python scripts/fetch_earthquake.py --target 20000     # custom
    python scripts/fetch_earthquake.py --dry-run           # estimate only
    python scripts/fetch_earthquake.py --no-transforms     # raw only
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import warnings

warnings.filterwarnings("ignore")

# ── Paths (SHARED) ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = Path("/Users/ahmed/Desktop/thesis_data/balanced_data/raw")
INDEX_CSV    = Path("/Users/ahmed/Desktop/thesis_data/balanced_data/index.csv")

# ── Config ────────────────────────────────────────────────────────────────
MIN_ARRAY_LEN = 50
API_BASE      = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# ── Query parameters ─────────────────────────────────────────────────────
# We fetch different magnitude ranges to get diverse arrays
# minmagnitude filters help get enough events in each query
QUERIES = [
    # Large global earthquakes — few per year, high magnitude variability
    {"label": "global_M5plus",    "minmagnitude": 5.0, "maxmagnitude": 10.0},
    # Moderate global — thousands per month
    {"label": "global_M4plus",    "minmagnitude": 4.0, "maxmagnitude": 5.0},
    # Smaller — very many events, rapid-fire aftershock sequences
    {"label": "global_M2.5plus",  "minmagnitude": 2.5, "maxmagnitude": 4.0},
    # All events in seismic regions — captures micro-earthquakes
    {"label": "global_M1plus",    "minmagnitude": 1.0, "maxmagnitude": 2.5},
]

# Time windows — different lengths give different array sizes and patterns
# (month-long windows during aftershock sequences vs multi-year for trends)
TIME_WINDOWS = [
    # Recent full years
    ("2020-01-01", "2020-12-31"),
    ("2021-01-01", "2021-12-31"),
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
    # Half-year windows — finer temporal resolution
    ("2020-01-01", "2020-06-30"),
    ("2020-07-01", "2020-12-31"),
    ("2021-01-01", "2021-06-30"),
    ("2021-07-01", "2021-12-31"),
    ("2022-01-01", "2022-06-30"),
    ("2022-07-01", "2022-12-31"),
    ("2023-01-01", "2023-06-30"),
    ("2023-07-01", "2023-12-31"),
    ("2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
    # Multi-year spans → larger arrays
    ("2015-01-01", "2019-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2000-01-01", "2009-12-31"),
    ("2005-01-01", "2014-12-31"),
    # Short windows during notable seismic events (aftershock sequences)
    ("2011-03-01", "2011-06-30"),  # Tohoku M9.1
    ("2023-02-01", "2023-05-31"),  # Turkey-Syria M7.8
    ("2010-01-01", "2010-03-31"),  # Haiti M7.0 + Chile M8.8
    ("2015-04-01", "2015-07-31"),  # Nepal M7.8
    ("2004-12-01", "2005-03-31"),  # Indian Ocean tsunami M9.1
    ("2016-04-01", "2016-07-31"),  # Ecuador M7.8 + Kumamoto M7.0
    ("2018-09-01", "2018-12-31"),  # Indonesia M7.5 + tsunami
    ("2019-07-01", "2019-09-30"),  # California Ridgecrest M7.1
    ("2023-09-01", "2023-12-31"),  # Morocco M6.8
    ("2017-09-01", "2017-12-31"),  # Mexico M8.2 + M7.1
]

# Regional bounding boxes for location-specific sequences
# (lat_min, lat_max, lon_min, lon_max, label)
REGIONS = [
    (None, None, None, None, "global"),          # No bounds = worldwide
    (30.0, 50.0, 125.0, 150.0, "japan"),         # Japan/Pacific Ring
    (35.0, 42.0, 25.0, 45.0, "turkey"),          # Turkey-Middle East
    (-5.0, 25.0, 95.0, 140.0, "indonesia"),      # Indonesia arc
    (32.0, 42.0, -125.0, -114.0, "california"),  # California
    (-40.0, -15.0, -75.0, -65.0, "chile"),       # Chile subduction
    (25.0, 50.0, 65.0, 100.0, "himalaya"),       # Himalayan collision
    (55.0, 65.0, -165.0, -145.0, "alaska"),      # Alaska-Aleutian arc
    (5.0, 20.0, 118.0, 128.0, "philippines"),    # Philippine trench
    (-48.0, -34.0, 165.0, 178.0, "newzealand"),  # New Zealand
    (36.0, 44.0, -10.0, 5.0, "iberia_maghreb"),  # Spain/Morocco
    (25.0, 40.0, -120.0, -100.0, "uswest"),      # Broader western US
]

# Raw columns extracted from each event set
RAW_ARRAY_NAMES = [
    "magnitude",       # power-law distributed
    "depth",           # bimodal: shallow (<70km) + deep (300-700km)
    "latitude",        # follows plate boundaries
    "longitude",       # follows plate boundaries
]

# Derived columns computed from raw event data
DERIVED_NAMES = [
    "inter_event_time",    # seconds between consecutive events
    "cumulative_energy",   # cumulative seismic energy release (log scale)
    "mag_diff",            # magnitude change between consecutive events
    "depth_diff",          # depth change between consecutive events
    "events_per_day",      # daily event count (integer-valued)
    "log_inter_event",     # log of inter-event time (more normal)
]

# Rate limiting — USGS is generous but let's be respectful
SLEEP_BETWEEN_REQUESTS = 1.5
SLEEP_ON_ERROR         = 10.0

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
#  Fetch earthquake events from USGS
# ═══════════════════════════════════════════════════════════════════════════

def fetch_events(
    start_date: str,
    end_date: str,
    minmag: float,
    maxmag: float,
    region: tuple | None = None,
) -> pd.DataFrame | None:
    """
    Fetch earthquake events from USGS API.
    Returns DataFrame with columns: time, latitude, longitude, depth, mag
    or None on failure.
    """
    params = {
        "format": "csv",
        "starttime": start_date,
        "endtime": end_date,
        "minmagnitude": minmag,
        "maxmagnitude": maxmag,
        "orderby": "time-asc",
        "limit": 20000,
    }

    if region is not None:
        lat_min, lat_max, lon_min, lon_max, _ = region
        if lat_min is not None:
            params["minlatitude"] = lat_min
            params["maxlatitude"] = lat_max
            params["minlongitude"] = lon_min
            params["maxlongitude"] = lon_max

    try:
        resp = requests.get(API_BASE, params=params, timeout=60)
        resp.raise_for_status()

        # USGS returns CSV directly
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))

        if df.empty or len(df) < MIN_ARRAY_LEN:
            return None

        return df

    except Exception as e:
        print(f"      API error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Extract arrays + derived columns from event DataFrame
# ═══════════════════════════════════════════════════════════════════════════

def extract_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract raw + derived arrays from USGS event DataFrame."""
    arrays = {}

    # Raw columns
    if "mag" in df.columns:
        mag = df["mag"].dropna().values.astype(np.float64)
        if len(mag) >= MIN_ARRAY_LEN:
            arrays["magnitude"] = mag

    if "depth" in df.columns:
        depth = df["depth"].dropna().values.astype(np.float64)
        if len(depth) >= MIN_ARRAY_LEN:
            arrays["depth"] = depth

    if "latitude" in df.columns:
        lat = df["latitude"].dropna().values.astype(np.float64)
        if len(lat) >= MIN_ARRAY_LEN:
            arrays["latitude"] = lat

    if "longitude" in df.columns:
        lon = df["longitude"].dropna().values.astype(np.float64)
        if len(lon) >= MIN_ARRAY_LEN:
            arrays["longitude"] = lon

    # Derived: inter-event time (seconds)
    if "time" in df.columns:
        try:
            times = pd.to_datetime(df["time"]).sort_values()
            dt = times.diff().dt.total_seconds().dropna().values.astype(np.float64)
            dt = dt[np.isfinite(dt) & (dt >= 0)]
            if len(dt) >= MIN_ARRAY_LEN:
                arrays["inter_event_time"] = dt
                # Log inter-event time (more normal-like)
                log_dt = np.log1p(dt)
                if len(log_dt) >= MIN_ARRAY_LEN:
                    arrays["log_inter_event"] = log_dt
        except Exception:
            pass

    # Derived: cumulative seismic energy (Gutenberg-Richter: E = 10^(1.5*M + 4.8))
    if "magnitude" in arrays:
        mag = arrays["magnitude"]
        energy = np.power(10.0, 1.5 * mag + 4.8)
        cum_energy = np.cumsum(energy)
        if len(cum_energy) >= MIN_ARRAY_LEN:
            arrays["cumulative_energy"] = np.log10(cum_energy)  # log scale

    # Derived: magnitude and depth differences
    if "magnitude" in arrays and len(arrays["magnitude"]) > 1:
        md = np.diff(arrays["magnitude"])
        if len(md) >= MIN_ARRAY_LEN:
            arrays["mag_diff"] = md

    if "depth" in arrays and len(arrays["depth"]) > 1:
        dd = np.diff(arrays["depth"])
        if len(dd) >= MIN_ARRAY_LEN:
            arrays["depth_diff"] = dd

    # Derived: events per day
    if "time" in df.columns:
        try:
            times = pd.to_datetime(df["time"])
            daily_counts = times.dt.date.value_counts().sort_index().values.astype(np.float64)
            if len(daily_counts) >= MIN_ARRAY_LEN:
                arrays["events_per_day"] = daily_counts
        except Exception:
            pass

    return arrays


# ═══════════════════════════════════════════════════════════════════════════
#  Event DataFrame chunking — split large event sets for more arrays
# ═══════════════════════════════════════════════════════════════════════════

EVENT_CHUNK_SIZES = [200, 500, 1000, 3000, 5000]
MAX_CHUNKS_PER_SIZE = 3


def chunk_events(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split a large event DataFrame into time-ordered chunks.
    Each chunk captures a different temporal segment (e.g., aftershock-rich
    period vs quiet period) and produces arrays with different structural
    properties."""
    chunks = []
    n = len(df)
    for cs in EVENT_CHUNK_SIZES:
        if n < cs * 2:  # need at least 2× chunk size
            continue
        n_full = min(n // cs, MAX_CHUNKS_PER_SIZE)
        for i in range(n_full):
            chunk = df.iloc[i * cs : (i + 1) * cs]
            if len(chunk) >= MIN_ARRAY_LEN:
                chunks.append((f"c{cs}_{i}", chunk))
    return chunks


# ═══════════════════════════════════════════════════════════════════════════
#  Save one array
# ═══════════════════════════════════════════════════════════════════════════

def save_array(
    arr: np.ndarray,
    query_label: str,
    period_tag: str,
    region_label: str,
    channel: str,
    existing_files: set[str],
    counter: int,
    total_bytes: int,
) -> tuple[int, int]:
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_ARRAY_LEN:
        return counter, total_bytes

    fname = f"quake_{region_label}_{query_label}_{period_tag}_{channel}.csv"

    if fname in existing_files:
        return counter, total_bytes

    np.savetxt(OUTPUT_DIR / fname, arr, fmt="%.10g")
    nbytes = (OUTPUT_DIR / fname).stat().st_size

    row = {
        "array_id": counter,
        "file": fname,
        "year": 0,
        "event": region_label,
        "round": 0,
        "session": period_tag,
        "driver": query_label,
        "lap": "EARTHQUAKE",
        "channel": channel,
        "n_elements": arr.size,
        "dtype": str(arr.dtype),
        "size_bytes": nbytes,
    }
    append_index_row(row)
    existing_files.add(fname)

    counter += 1
    total_bytes += nbytes

    if counter % 500 == 0:
        print(f"      [{counter:6,} quake arrays | {total_bytes / (1024**2):,.1f} MB]")

    return counter, total_bytes


def save_array_and_transforms(
    arr: np.ndarray,
    query_label: str,
    period_tag: str,
    region_label: str,
    channel: str,
    existing_files: set[str],
    counter: int,
    total_bytes: int,
) -> tuple[int, int]:
    counter, total_bytes = save_array(
        arr, query_label, period_tag, region_label, channel,
        existing_files, counter, total_bytes,
    )

    seed = hash(f"quake_{region_label}_{query_label}_{period_tag}_{channel}") & 0xFFFFFFFF
    transforms = apply_transforms(arr, seed)
    for tname, tarr in transforms.items():
        counter, total_bytes = save_array(
            tarr, query_label, period_tag, region_label, f"{channel}_{tname}",
            existing_files, counter, total_bytes,
        )

    return counter, total_bytes


# ═══════════════════════════════════════════════════════════════════════════
#  Main fetch loop
# ═══════════════════════════════════════════════════════════════════════════

def fetch_all(target: int = 100_000, transforms: bool = True) -> None:
    all_counter, all_existing = init_output()

    quake_existing = {f for f in all_existing if f.startswith("quake_")}
    quake_counter = len(quake_existing)
    total_bytes = 0
    if INDEX_CSV.exists() and quake_counter > 0:
        try:
            df = pd.read_csv(INDEX_CSV)
            quake_df = df[df["file"].str.startswith("quake_")]
            total_bytes = int(quake_df["size_bytes"].sum())
        except Exception:
            pass

    total_combos = len(QUERIES) * len(TIME_WINDOWS) * len(REGIONS)

    print("=" * 70)
    print("Earthquake Data Fetcher  —  Step 1 (data collection only)")
    print("=" * 70)
    print(f"  Magnitude ranges: {len(QUERIES)}")
    print(f"  Time windows:     {len(TIME_WINDOWS)}")
    print(f"  Regions:          {len(REGIONS)} (global + 6 seismic zones)")
    print(f"  API calls:        up to {total_combos}")
    print(f"  Target:           {target:,} earthquake arrays")
    print(f"  Already have:     {quake_counter:,} quake arrays ({total_bytes / (1024**2):,.1f} MB)")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"  Index CSV:        {INDEX_CSV} (shared)")
    if transforms:
        print(f"  Transforms:       ENABLED (REV, SHUF, QBIN50, PSORT10 → ×5)")
    else:
        print(f"  Transforms:       DISABLED")
    print(f"  API:              USGS Earthquake (free, no key)")
    print("=" * 70)

    if quake_counter >= target:
        print(f"\nAlready have {quake_counter:,} quake arrays >= target {target:,}. Done!")
        return

    def reached_target() -> bool:
        return quake_counter >= target

    save_fn = save_array_and_transforms if transforms else save_array
    api_calls = 0
    combo_num = 0

    for region in REGIONS:
        if reached_target():
            break

        region_label = region[4]
        print(f"\n  Region: {region_label}")

        for query in QUERIES:
            if reached_target():
                break

            q_label = query["label"]
            minmag = query["minmagnitude"]
            maxmag = query["maxmagnitude"]

            for start_date, end_date in TIME_WINDOWS:
                if reached_target():
                    break

                combo_num += 1
                period_tag = f"{start_date.replace('-','')}_{end_date.replace('-','')}"

                print(f"    [{combo_num}/{total_combos}] {q_label} | {start_date}→{end_date} | {region_label}")

                events_df = fetch_events(start_date, end_date, minmag, maxmag, region)
                api_calls += 1
                time.sleep(SLEEP_BETWEEN_REQUESTS)

                if events_df is None:
                    print(f"      (no data / too few events)")
                    continue

                print(f"      {len(events_df):,} events")

                # Full event set
                arrays = extract_arrays(events_df)

                for arr_name, arr in arrays.items():
                    if reached_target():
                        break
                    quake_counter, total_bytes = save_fn(
                        arr, q_label, period_tag, region_label, arr_name,
                        all_existing, quake_counter, total_bytes,
                    )

                # Chunked event subsets — different temporal segments
                if not reached_target():
                    for chunk_tag, chunk_df in chunk_events(events_df):
                        if reached_target():
                            break
                        chunk_arrays = extract_arrays(chunk_df)
                        for arr_name, arr in chunk_arrays.items():
                            if reached_target():
                                break
                            quake_counter, total_bytes = save_fn(
                                arr, q_label, period_tag, region_label,
                                f"{arr_name}_{chunk_tag}",
                                all_existing, quake_counter, total_bytes,
                            )

        print(f"    => {quake_counter:,} quake arrays | {total_bytes / (1024**2):,.1f} MB")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EARTHQUAKE FETCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Quake arrays:  {quake_counter:,}")
    print(f"  Quake size:    {total_bytes / (1024**2):,.1f} MB")
    print(f"  API calls:     {api_calls}")
    print(f"  Output:        {OUTPUT_DIR}")
    print(f"  Index:         {INDEX_CSV}")

    if INDEX_CSV.exists():
        df = pd.read_csv(INDEX_CSV)
        quake_df = df[df["file"].str.startswith("quake_")]
        if len(quake_df) > 0:
            print(f"\n  Quake breakdown:")
            print(f"    Regions:     {quake_df['event'].nunique()}")
            print(f"    Mag ranges:  {quake_df['driver'].nunique()}")
            print(f"    Variables:   {quake_df['channel'].nunique()}")
            print(f"    Array sizes: min={quake_df['n_elements'].min():,}  "
                  f"max={quake_df['n_elements'].max():,}  "
                  f"mean={quake_df['n_elements'].mean():,.0f}")

        # Combined totals
        print(f"\n  COMBINED DATASET:")
        for prefix, label in [("f1_", "F1"), ("stock_", "Stock"),
                               ("weather_", "Weather"), ("crypto_", "Crypto"),
                               ("quake_", "Earthquake")]:
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
        description="Fetch raw earthquake arrays via USGS API (Step 1 — data collection only)"
    )
    p.add_argument("--target", type=int, default=100_000,
                   help="Stop after this many earthquake arrays (default: 100000)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print estimated count without fetching")
    p.add_argument("--no-transforms", action="store_true",
                   help="Save only raw arrays, skip structural transforms")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        n_combos = len(QUERIES) * len(TIME_WINDOWS) * len(REGIONS)
        n_vars = len(RAW_ARRAY_NAMES) + len(DERIVED_NAMES)
        est_raw = n_combos * n_vars
        est = est_raw * 5
        print("DRY RUN — estimating data volume\n")
        print(f"  Magnitude ranges:    {len(QUERIES)}")
        print(f"  Time windows:        {len(TIME_WINDOWS)}")
        print(f"  Regions:             {len(REGIONS)}")
        print(f"  Total API calls:     {n_combos}")
        print(f"  Variables per query: up to {n_vars} ({len(RAW_ARRAY_NAMES)} raw + {len(DERIVED_NAMES)} derived)")
        print(f"  Max raw arrays:      {est_raw:,}")
        print(f"  With transforms:     {est:,}  (×5)")
        print(f"  Realistic estimate:  ~{int(est * 0.7):,}  (some combos too few events)")
        print(f"  Est. disk usage:     ~{est * 0.003:.0f} MB")
        return

    fetch_all(target=args.target, transforms=not args.no_transforms)


if __name__ == "__main__":
    main()
