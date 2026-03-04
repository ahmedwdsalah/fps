#!/usr/bin/env python3
"""
fetch_weather.py  ·  Pure Weather Data Fetcher (Step 1 only)
=============================================================
Fetches raw weather arrays via Open-Meteo Historical API (free, no key needed)
and saves them as plain .csv files.
NO feature extraction, NO timing — just raw data capture.

Design principles (same as fetch_f1_10k.py and fetch_stock.py):
  • Crash-safe: index CSV appended after EVERY array save.
  • Resumable: on restart, reads existing index to skip already-fetched files.
  • Saves into data/real_world_10k/ alongside F1 and Stock data.

Why weather data adds value to the training set:
  • Temperature:      smooth sinusoidal annual cycle — NEW pattern (high adj_sorted,
                      long runs, low disorder). Neither F1 telemetry nor stock has this.
  • Precipitation:    sparse/zero-inflated — MANY zeros (natural high duplicate_ratio).
                      This is exactly the heapsort territory we needed transforms for
                      in stock. Weather gives it for FREE from raw data.
  • Pressure:         near-constant series — extremely narrow value range
                      (1013 ± 30 hPa), high adj_sorted, near-zero entropy_ratio.
  • Wind speed:       oscillating positive noise — moderate structure,
                      right-skewed, some outlier spikes.
  • Humidity:         bounded [0,100%] — creates natural plateaus, moderate
                      duplicates at 0% and 100% boundaries.
  • Derived columns:  daily temp range, rolling means/stds add even more variety.

Structural transforms (same as stock) for anti-bias:
  REV, SHUF, QBIN50, PSORT10 → ×5 arrays per raw array.

Open-Meteo API details:
  - URL: https://archive-api.open-meteo.com/v1/archive
  - Free, no API key, no registration
  - Historical hourly data from 1940 to present
  - Rate limit: ~600 requests/minute (generous)
  - Returns JSON with hourly arrays

Output (shared with F1 + Stock):
  data/real_world_10k/raw/       → .csv files (one number per line)
  data/real_world_10k/index.csv  → metadata index (appended incrementally)

Usage:
    source venv/bin/activate
    python scripts/fetch_weather.py                    # default 50K target
    python scripts/fetch_weather.py --target 30000     # custom target
    python scripts/fetch_weather.py --dry-run           # estimate only
    python scripts/fetch_weather.py --no-transforms     # raw only
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import warnings

warnings.filterwarnings("ignore")

# ── Paths (SHARED with F1 + Stock data) ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "real_world_10k" / "raw"
INDEX_CSV    = PROJECT_ROOT / "data" / "real_world_10k" / "index.csv"

# ── Config ────────────────────────────────────────────────────────────────
MIN_ARRAY_LEN = 50
API_BASE      = "https://archive-api.open-meteo.com/v1/archive"

# ── Weather variables to fetch (hourly) ──────────────────────────────────
# These map to Open-Meteo parameter names
HOURLY_VARIABLES = [
    "temperature_2m",         # °C, smooth sinusoidal
    "relative_humidity_2m",   # %, bounded [0,100], natural plateaus
    "precipitation",          # mm, zero-inflated / sparse spiky
    "surface_pressure",       # hPa, near-constant ~1013±30
    "wind_speed_10m",         # km/h, oscillating positive noise
    "cloud_cover",            # %, bounded [0,100], many 0s and 100s
    "shortwave_radiation",    # W/m², zero at night, peaks at noon → sawtooth
]

# Derived columns computed from raw data
DERIVED_NAMES = [
    "temp_daily_range",       # max−min temperature per day (positive, right-skewed)
    "temp_rolling_24h_mean",  # 24h rolling mean (very smooth)
    "temp_rolling_168h_mean", # 7-day rolling mean (extremely smooth)
    "temp_diff",              # hour-to-hour temp change (symmetric noise)
    "precip_cumsum",          # cumulative precipitation (monotone increasing)
    "pressure_diff",          # hour-to-hour pressure change (near-zero noise)
    "wind_rolling_24h_std",   # daily wind volatility
]

# ── Cities: diverse climates + locations for maximum array variety ────────
# Format: (city_name, latitude, longitude)
# Chosen to cover: tropical, arid, temperate, continental, polar, monsoon,
# maritime, Mediterranean — different temp/precip patterns
CITIES = [
    # Tropical — hot, humid, heavy rain
    ("Singapore",        1.29,  103.85),
    ("Bangkok",         13.75,  100.52),
    ("Lagos",            6.45,    3.40),
    ("Manaus",          -3.12,  -60.02),
    ("Jakarta",         -6.21,  106.85),
    ("Mumbai",          19.08,   72.88),
    ("Havana",          23.13,  -82.38),
    ("Nairobi",         -1.29,   36.82),

    # Arid / desert — extreme heat, near-zero rain
    ("Riyadh",          24.69,   46.72),
    ("Phoenix",         33.45, -112.07),
    ("Cairo",           30.04,   31.24),
    ("Dubai",           25.20,   55.27),
    ("Lima",           -12.05,  -77.04),
    ("Karachi",         24.86,   67.01),
    ("Las_Vegas",       36.17, -115.14),
    ("Alice_Springs",  -23.70,  133.88),

    # Temperate / Mediterranean — seasonal, moderate rain
    ("London",          51.51,   -0.13),
    ("Paris",           48.86,    2.35),
    ("Berlin",          52.52,   13.41),
    ("Rome",            41.90,   12.50),
    ("Madrid",          40.42,   -3.70),
    ("Tokyo",           35.68,  139.69),
    ("Sydney",         -33.87,  151.21),
    ("San_Francisco",   37.77, -122.42),
    ("New_York",        40.71,  -74.01),
    ("Buenos_Aires",   -34.60,  -58.38),
    ("Istanbul",        41.01,   28.98),
    ("Cape_Town",      -33.92,   18.42),

    # Continental — huge temp swings summer/winter
    ("Moscow",          55.76,   37.62),
    ("Chicago",         41.88,  -87.63),
    ("Beijing",         39.90,  116.41),
    ("Toronto",         43.65,  -79.38),
    ("Ulaanbaatar",     47.92,  106.91),
    ("Astana",          51.17,   71.43),
    ("Minneapolis",     44.98,  -93.27),
    ("Novosibirsk",     55.03,   82.92),

    # Polar / sub-arctic — extreme cold, low precipitation
    ("Reykjavik",       64.15,  -21.94),
    ("Anchorage",       61.22, -149.90),
    ("Tromsoe",         69.65,   18.96),
    ("Yakutsk",         62.04,  129.73),
    ("Murmansk",        68.97,   33.09),
    ("Fairbanks",       64.84, -147.72),

    # High-altitude — thin air, big temp swings
    ("La_Paz",         -16.50,  -68.15),
    ("Quito",           -0.18,  -78.47),
    ("Denver",          39.74, -104.99),
    ("Addis_Ababa",      9.02,   38.75),
    ("Lhasa",           29.65,   91.10),
    ("Bogota",           4.71,  -74.07),

    # Monsoon / wet-dry — extreme seasonal precipitation
    ("Chennai",         13.08,   80.27),
    ("Hanoi",           21.03,  105.85),
    ("Manila",          14.60,  120.98),
    ("Dhaka",           23.81,   90.41),
    ("Yangon",          16.87,   96.20),
    ("Ho_Chi_Minh",     10.82,  106.63),
]

# Time periods — different year ranges give different array lengths
# and different climate episodes (El Niño, La Niña, etc.)
TIME_RANGES = [
    ("2020-01-01", "2020-12-31"),   # 1 year  → ~8760 hourly points
    ("2019-01-01", "2020-12-31"),   # 2 years → ~17520 points
    ("2015-01-01", "2020-12-31"),   # 6 years → ~52560 points
    ("2000-01-01", "2020-12-31"),   # 21 years → ~184K points (large!)
    ("2022-06-01", "2022-08-31"),   # 1 summer → ~2208 points (short)
    ("2021-12-01", "2022-02-28"),   # 1 winter → ~2160 points (short)
]

# Rate limiting — Open-Meteo free tier is stricter than documented
SLEEP_BETWEEN_REQUESTS = 2.0   # seconds between successful requests
SLEEP_ON_ERROR         = 10.0  # base wait on non-429 errors
MAX_RETRIES            = 5     # retries per request on 429
RETRY_BASE_WAIT        = 15.0  # base seconds to wait on 429 (doubles each retry)

# ── CSV columns (shared with F1 + Stock index) ───────────────────────────
INDEX_COLUMNS = [
    "array_id", "file", "year", "event", "round", "session",
    "driver", "lap", "channel", "n_elements", "dtype", "size_bytes",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Index CSV — incremental append (reuses shared index)
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
        # Write header — only if index doesn't exist at all
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
#  Structural transforms — anti-bias (same as stock fetcher)
# ═══════════════════════════════════════════════════════════════════════════

def apply_transforms(arr: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """
    Generate 4 structural transforms of an array.
    See fetch_stock.py docstring for full rationale.
    """
    rng = np.random.default_rng(seed)
    out = {}

    # 1. Reversed
    out["REV"] = arr[::-1].copy()

    # 2. Fully shuffled
    shuf = arr.copy()
    rng.shuffle(shuf)
    out["SHUF"] = shuf

    # 3. Quantized to 50 bins — high duplicate_ratio
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        edges = np.linspace(vmin, vmax, 51)
        idx = np.clip(np.digitize(arr, edges) - 1, 0, 49)
        centers = (edges[:-1] + edges[1:]) / 2.0
        out["QBIN50"] = centers[idx]

    # 4. Nearly sorted — sort then perturb 10%
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
#  Compute derived columns from weather data
# ═══════════════════════════════════════════════════════════════════════════

def compute_derived(hourly_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute derived weather arrays from raw hourly data."""
    derived = {}

    temp = hourly_data.get("temperature_2m")
    if temp is not None and len(temp) >= MIN_ARRAY_LEN:
        # Hour-to-hour temperature change — symmetric noise
        if len(temp) > 1:
            td = np.diff(temp)
            if len(td) >= MIN_ARRAY_LEN:
                derived["temp_diff"] = td

        # Rolling means — very smooth
        s = pd.Series(temp)
        for window, name in [(24, "temp_rolling_24h_mean"),
                              (168, "temp_rolling_168h_mean")]:
            r = s.rolling(window).mean().dropna().values
            if len(r) >= MIN_ARRAY_LEN:
                derived[name] = r.astype(np.float64)

        # Daily temperature range: group by 24h blocks, take max-min
        n_days = len(temp) // 24
        if n_days >= MIN_ARRAY_LEN:
            daily_temp = temp[:n_days * 24].reshape(n_days, 24)
            daily_range = daily_temp.max(axis=1) - daily_temp.min(axis=1)
            derived["temp_daily_range"] = daily_range

    precip = hourly_data.get("precipitation")
    if precip is not None and len(precip) >= MIN_ARRAY_LEN:
        # Cumulative precipitation — monotone increasing
        cs = np.cumsum(precip)
        if len(cs) >= MIN_ARRAY_LEN:
            derived["precip_cumsum"] = cs

    pressure = hourly_data.get("surface_pressure")
    if pressure is not None and len(pressure) > 1:
        pd_diff = np.diff(pressure)
        if len(pd_diff) >= MIN_ARRAY_LEN:
            derived["pressure_diff"] = pd_diff

    wind = hourly_data.get("wind_speed_10m")
    if wind is not None and len(wind) >= MIN_ARRAY_LEN:
        s = pd.Series(wind)
        r = s.rolling(24).std().dropna().values
        if len(r) >= MIN_ARRAY_LEN:
            derived["wind_rolling_24h_std"] = r.astype(np.float64)

    return derived


# ═══════════════════════════════════════════════════════════════════════════
#  Save one array
# ═══════════════════════════════════════════════════════════════════════════

def save_array(
    arr: np.ndarray,
    city: str,
    period_tag: str,
    channel: str,
    existing_files: set[str],
    counter: int,
    total_bytes: int,
) -> tuple[int, int]:
    """Save a single array as .csv. Returns (updated_counter, updated_total_bytes)."""

    # Remove NaN/Inf
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_ARRAY_LEN:
        return counter, total_bytes

    # Build filename: weather_{city}_{period}_{channel}.csv
    fname = f"weather_{city}_{period_tag}_{channel}.csv"

    # Skip if already saved (resume support)
    if fname in existing_files:
        return counter, total_bytes

    # Save as plain CSV (one number per line)
    np.savetxt(OUTPUT_DIR / fname, arr, fmt="%.10g")
    nbytes = (OUTPUT_DIR / fname).stat().st_size

    # Append to shared index CSV
    row = {
        "array_id": counter,
        "file": fname,
        "year": 0,
        "event": city,              # city name in event column
        "round": 0,
        "session": period_tag,      # date range tag in session column
        "driver": "hourly",         # resolution in driver column
        "lap": "WEATHER",           # domain tag
        "channel": channel,         # variable name
        "n_elements": arr.size,
        "dtype": str(arr.dtype),
        "size_bytes": nbytes,
    }
    append_index_row(row)
    existing_files.add(fname)

    counter += 1
    total_bytes += nbytes

    if counter % 500 == 0:
        print(f"      [{counter:6,} weather arrays | {total_bytes / (1024**2):,.1f} MB]")

    return counter, total_bytes


def save_array_and_transforms(
    arr: np.ndarray,
    city: str,
    period_tag: str,
    channel: str,
    existing_files: set[str],
    counter: int,
    total_bytes: int,
) -> tuple[int, int]:
    """Save original array + 4 structural transforms."""
    # Save the original
    counter, total_bytes = save_array(
        arr, city, period_tag, channel,
        existing_files, counter, total_bytes,
    )

    # Save transforms
    seed = hash(f"weather_{city}_{period_tag}_{channel}") & 0xFFFFFFFF
    transforms = apply_transforms(arr, seed)
    for tname, tarr in transforms.items():
        counter, total_bytes = save_array(
            tarr, city, period_tag, f"{channel}_{tname}",
            existing_files, counter, total_bytes,
        )

    return counter, total_bytes


# ═══════════════════════════════════════════════════════════════════════════
#  Fetch one city/period from Open-Meteo
# ═══════════════════════════════════════════════════════════════════════════

def fetch_city_period(
    city: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> dict[str, np.ndarray] | None:
    """
    Fetch hourly weather data for one city and date range.
    Returns dict of {variable_name: numpy_array} or None on failure.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(API_BASE, params=params, timeout=30)
            if resp.status_code == 429:
                wait = RETRY_BASE_WAIT * (2 ** attempt)
                print(f"      429 rate-limited → waiting {wait:.0f}s (attempt {attempt+1}/{MAX_RETRIES+1})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break  # success
        except requests.exceptions.HTTPError as e:
            if "429" not in str(e):
                print(f"      API error: {e}")
                return None
            # 429 already handled above, but just in case
            wait = RETRY_BASE_WAIT * (2 ** attempt)
            print(f"      429 rate-limited → waiting {wait:.0f}s (attempt {attempt+1}/{MAX_RETRIES+1})")
            time.sleep(wait)
            continue
        except Exception as e:
            print(f"      API error: {e}")
            time.sleep(SLEEP_ON_ERROR)
            return None
    else:
        print(f"      Gave up after {MAX_RETRIES+1} retries")
        return None

    hourly = data.get("hourly", {})
    if not hourly:
        return None

    arrays = {}
    for var in HOURLY_VARIABLES:
        values = hourly.get(var)
        if values is not None:
            # Convert to float64, handle None values as NaN
            arr = np.array([float(v) if v is not None else np.nan
                           for v in values], dtype=np.float64)
            # Remove NaNs
            arr = arr[np.isfinite(arr)]
            if arr.size >= MIN_ARRAY_LEN:
                arrays[var] = arr

    return arrays if arrays else None


# ═══════════════════════════════════════════════════════════════════════════
#  Main fetch loop
# ═══════════════════════════════════════════════════════════════════════════

def fetch_all(target: int = 50_000, transforms: bool = True) -> None:
    """Fetch weather data for all cities across all time periods."""

    # Resume support
    all_counter, all_existing = init_output()

    # Count weather-specific arrays for progress
    weather_existing = {f for f in all_existing if f.startswith("weather_")}
    weather_counter = len(weather_existing)
    total_bytes = 0
    if INDEX_CSV.exists() and weather_counter > 0:
        try:
            df = pd.read_csv(INDEX_CSV)
            weather_df = df[df["file"].str.startswith("weather_")]
            total_bytes = int(weather_df["size_bytes"].sum())
        except Exception:
            pass

    n_cities = len(CITIES)
    n_periods = len(TIME_RANGES)
    n_vars = len(HOURLY_VARIABLES) + len(DERIVED_NAMES)

    print("=" * 70)
    print("Weather Data Fetcher  —  Step 1 (data collection only)")
    print("=" * 70)
    print(f"  Cities:         {n_cities} (tropical/arid/temperate/continental/polar/monsoon)")
    print(f"  Time ranges:    {n_periods}")
    print(f"  Variables:      up to {n_vars} ({len(HOURLY_VARIABLES)} raw + {len(DERIVED_NAMES)} derived)")
    print(f"  Target:         {target:,} weather arrays")
    print(f"  Already have:   {weather_counter:,} weather arrays ({total_bytes / (1024**2):,.1f} MB)")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  Index CSV:      {INDEX_CSV} (shared with F1 + Stock)")
    if transforms:
        print(f"  Transforms:     ENABLED (REV, SHUF, QBIN50, PSORT10 → ×5 arrays)")
    else:
        print(f"  Transforms:     DISABLED (raw arrays only)")
    print(f"  API:            Open-Meteo Archive (free, no key)")
    print("=" * 70)

    if weather_counter >= target:
        print(f"\nAlready have {weather_counter:,} weather arrays >= target {target:,}. Done!")
        return

    def reached_target() -> bool:
        return weather_counter >= target

    save_fn = save_array_and_transforms if transforms else save_array

    total_requests = 0

    for i, (city, lat, lon) in enumerate(CITIES):
        if reached_target():
            break

        print(f"\n  [{i+1}/{n_cities}] {city} ({lat:.1f}°, {lon:.1f}°)")

        for start_date, end_date in TIME_RANGES:
            if reached_target():
                break

            # Period tag for filename
            period_tag = f"{start_date[:4]}_{end_date[:4]}"
            if start_date[5:] != "01-01":
                # Short seasonal period — include month
                period_tag = f"{start_date.replace('-','')[:8]}_{end_date.replace('-','')[:8]}"

            # Fetch hourly data from API
            arrays = fetch_city_period(city, lat, lon, start_date, end_date)
            total_requests += 1
            time.sleep(SLEEP_BETWEEN_REQUESTS)

            if arrays is None:
                continue

            # Save raw variables
            for var_name, arr in arrays.items():
                if reached_target():
                    break
                weather_counter, total_bytes = save_fn(
                    arr, city, period_tag, var_name,
                    all_existing, weather_counter, total_bytes,
                )

            if reached_target():
                break

            # Compute and save derived variables
            derived = compute_derived(arrays)
            for var_name, arr in derived.items():
                if reached_target():
                    break
                weather_counter, total_bytes = save_fn(
                    arr, city, period_tag, var_name,
                    all_existing, weather_counter, total_bytes,
                )

        # Progress per city
        print(f"    => {weather_counter:,} weather arrays | {total_bytes / (1024**2):,.1f} MB | {total_requests} API calls")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  WEATHER FETCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Weather arrays:  {weather_counter:,}")
    print(f"  Weather size:    {total_bytes / (1024**2):,.1f} MB")
    print(f"  API calls:       {total_requests}")
    print(f"  Output:          {OUTPUT_DIR}")
    print(f"  Index:           {INDEX_CSV}")

    if INDEX_CSV.exists():
        df = pd.read_csv(INDEX_CSV)
        weather_df = df[df["file"].str.startswith("weather_")]
        if len(weather_df) > 0:
            print(f"\n  Weather breakdown:")
            print(f"    Cities:      {weather_df['event'].nunique()}")
            print(f"    Periods:     {weather_df['session'].nunique()}")
            print(f"    Variables:   {weather_df['channel'].nunique()}")
            print(f"    Array sizes: min={weather_df['n_elements'].min():,}  "
                  f"max={weather_df['n_elements'].max():,}  "
                  f"mean={weather_df['n_elements'].mean():,.0f}")

        # Combined totals
        f1_count = len(df[df["file"].str.startswith("f1_")])
        stock_count = len(df[df["file"].str.startswith("stock_")])
        w_count = len(weather_df)
        print(f"\n  COMBINED DATASET:")
        print(f"    F1 arrays:      {f1_count:,}")
        print(f"    Stock arrays:   {stock_count:,}")
        print(f"    Weather arrays: {w_count:,}")
        print(f"    TOTAL:          {len(df):,}")
        print(f"    Total size:     {df['size_bytes'].sum() / (1024**2):,.1f} MB")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch raw weather arrays via Open-Meteo (Step 1 — data collection only)"
    )
    p.add_argument(
        "--target", type=int, default=50_000,
        help="Stop after collecting this many weather arrays (default: 50000)"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print estimated array count without fetching"
    )
    p.add_argument(
        "--no-transforms", action="store_true",
        help="Save only raw arrays, skip structural transforms"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        n_cities = len(CITIES)
        n_periods = len(TIME_RANGES)
        n_vars = len(HOURLY_VARIABLES) + len(DERIVED_NAMES)
        est_raw = n_cities * n_periods * n_vars
        est_with_t = est_raw * 5
        print("DRY RUN — estimating data volume\n")
        print(f"  Cities:              {n_cities}")
        print(f"  Time ranges:         {n_periods}")
        print(f"  Variables per range: up to {n_vars} ({len(HOURLY_VARIABLES)} raw + {len(DERIVED_NAMES)} derived)")
        print(f"  API calls needed:    {n_cities * n_periods}")
        print(f"  Max raw arrays:      {est_raw:,}")
        print(f"  With transforms:     {est_with_t:,}  (×5: original + REV/SHUF/QBIN50/PSORT10)")
        print(f"  Realistic estimate:  ~{int(est_with_t * 0.85):,}  (some short periods drop)")
        print(f"  Est. disk usage:     ~{est_with_t * 0.010:.0f} MB  (avg ~10KB per array, weather is longer)")
        return

    fetch_all(target=args.target, transforms=not args.no_transforms)


if __name__ == "__main__":
    main()
