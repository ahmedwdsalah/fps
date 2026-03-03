#!/usr/bin/env python3
"""
fetch_f1_10k.py  ·  Pure F1 Data Fetcher (Step 1 only)
=======================================================
Fetches raw F1 telemetry arrays via FastF1 and saves them as plain .csv files.
NO feature extraction, NO timing, NO filtering — just raw data capture.

Design principles:
  • Bring ALL data: ALL drivers, ALL laps, ALL numeric channels.
    No cherry-picking. The model pipeline (Step 2) decides what to use.
  • Crash-safe: index CSV is appended after EVERY array save.
    If the session disconnects, you keep everything collected so far.
  • Resumable: on restart, reads existing index to skip already-fetched arrays.
  • Rate-limit friendly: configurable sleeps + retry on API errors.
  • Disk-aware: prints running total so you can monitor usage.

Output:
  data/real_world_10k/raw/       → .csv files (one number per line, plain text)
  data/real_world_10k/index.csv  → metadata index (appended incrementally)

Pipeline:
  Step 1: fetch_f1_10k.py          → raw .csv + index.csv     (THIS SCRIPT)
  Step 2: build_training_dataset.py → features + timings CSV   (separate script)
  Step 3: train model               → XGBoost on real data     (separate script)

Usage:
    source venv/bin/activate
    python scripts/fetch_f1_10k.py                          # full run (2018-2024)
    python scripts/fetch_f1_10k.py --seasons 2022 2023      # specific seasons
    python scripts/fetch_f1_10k.py --dry-run                # estimate without fetching
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

import logging

import fastf1

# Suppress noisy FastF1 logging (keep only errors)
logging.getLogger("fastf1").setLevel(logging.ERROR)

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "real_world_10k" / "raw"
INDEX_CSV    = PROJECT_ROOT / "data" / "real_world_10k" / "index.csv"
CACHE_DIR    = PROJECT_ROOT / "data" / "f1_cache"

# ── Config ────────────────────────────────────────────────────────────────
DEFAULT_SEASONS  = list(range(2018, 2025))  # 2018-2024 (7 seasons)
DEFAULT_SESSIONS = ["FP1", "FP2", "FP3", "Q", "R", "S", "SS"]  # every session type

# Only filter: arrays shorter than this are noise (< 3 seconds of 10-20Hz telemetry)
MIN_ARRAY_LEN = 50

# Channels to skip (low-quality data that adds noise, not signal)
# DRS: 34% all-zeros, rest binary 0/1 — no sorting diversity
# nGear: only 7 unique values (gears 2-8) — too low diversity
SKIP_CHANNELS = {"DRS", "nGear"}

# Rate limiting
SLEEP_BETWEEN_SESSIONS = 1.0   # seconds between session loads
SLEEP_BETWEEN_GPS      = 3.0   # seconds between different GPs
SLEEP_ON_ERROR         = 30.0  # seconds to wait on API error before retry
MAX_RETRIES            = 2

# ── CSV columns ───────────────────────────────────────────────────────────
INDEX_COLUMNS = [
    "array_id", "file", "year", "event", "round", "session",
    "driver", "lap", "channel", "n_elements", "dtype", "size_bytes",
]


# ═══════════════════════════════════════════════════════════════════════════
#  Index CSV — incremental append
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
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that are actually numeric (int/float),
    excluding channels in SKIP_CHANNELS."""
    cols = []
    for col in df.columns:
        if col in SKIP_CHANNELS:
            continue
        if df[col].dtype.kind in ("i", "u", "f"):  # int, unsigned int, float
            cols.append(col)
    return cols


def load_session_safe(year: int, roundnum: int, sess: str):
    """Load a FastF1 session with retry on error."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            session = fastf1.get_session(year, roundnum, sess)
            session.load(telemetry=True, laps=True, weather=False)
            return session
        except ValueError:
            # Session type doesn't exist for this event (e.g. Sprint on a
            # non-sprint weekend). Not a transient error — skip immediately.
            return None
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = SLEEP_ON_ERROR * (attempt + 1)
                print(f"      ERROR (attempt {attempt+1}/{MAX_RETRIES+1}): {e}")
                print(f"      Retrying in {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"      FAILED after {MAX_RETRIES+1} attempts: {e}")
                return None


# ═══════════════════════════════════════════════════════════════════════════
#  Core: save all numeric channels from a telemetry DataFrame
# ═══════════════════════════════════════════════════════════════════════════

def save_telemetry_arrays(
    telemetry_df: pd.DataFrame,
    year: int,
    event_name: str,
    roundnum: int,
    sess: str,
    driver: str,
    lap: str,
    existing_files: set[str],
    counter: int,
    total_bytes: int,
) -> tuple[int, int]:
    """
    Save every numeric channel from a telemetry DataFrame as a plain .csv file.
    Appends to index CSV after each save.

    Returns (updated_counter, updated_total_bytes).
    """
    if telemetry_df is None or telemetry_df.empty:
        return counter, total_bytes

    numeric_cols = get_numeric_columns(telemetry_df)

    for ch in numeric_cols:
        try:
            raw = telemetry_df[ch].to_numpy()
            # Remove NaN but keep everything else as-is
            arr = raw[~pd.isna(raw)]

            if arr.size < MIN_ARRAY_LEN:
                continue

            # Build filename
            fname = f"f1_{year}_R{roundnum}_{sess}_{driver}_{lap}_{ch}.csv"

            # Skip if already saved (resume support)
            if fname in existing_files:
                continue

            # Save as plain CSV (one number per line — universal format)
            np.savetxt(OUTPUT_DIR / fname, arr, fmt="%.10g")
            nbytes = (OUTPUT_DIR / fname).stat().st_size

            # Append to index CSV immediately
            row = {
                "array_id": counter,
                "file": fname,
                "year": year,
                "event": event_name,
                "round": roundnum,
                "session": sess,
                "driver": driver,
                "lap": lap,
                "channel": ch,
                "n_elements": arr.size,
                "dtype": str(arr.dtype),
                "size_bytes": nbytes,
            }
            append_index_row(row)
            existing_files.add(fname)

            counter += 1
            total_bytes += nbytes

            if counter % 100 == 0:
                print(
                    f"        [{counter:6,} arrays | "
                    f"{total_bytes / (1024**2):,.1f} MB on disk]"
                )

        except Exception as e:
            # Don't crash on one bad channel — skip and continue
            continue

    return counter, total_bytes


# ═══════════════════════════════════════════════════════════════════════════
#  Main fetch loop — brings EVERYTHING
# ═══════════════════════════════════════════════════════════════════════════

def fetch_all(seasons: list[int], sessions: list[str], target: int = 0) -> None:
    """Fetch ALL drivers, ALL laps, ALL numeric channels. No filtering.
    If target > 0, stop after collecting that many arrays."""

    # Setup cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    # Resume support
    counter, existing_files = init_output()
    total_bytes = 0
    if counter > 0:
        # Estimate existing disk usage from index
        try:
            df = pd.read_csv(INDEX_CSV)
            total_bytes = int(df["size_bytes"].sum())
        except Exception:
            pass

    print("=" * 70)
    print("F1 Raw Data Fetcher  —  Step 1 (data collection only)")
    print("=" * 70)
    print(f"  Seasons:        {seasons}")
    print(f"  Session types:  {sessions}")
    print(f"  Min array len:  {MIN_ARRAY_LEN}")
    print(f"  Target:         {'unlimited' if target <= 0 else f'{target:,} arrays'}")
    print(f"  Already have:   {counter:,} arrays ({total_bytes / (1024**2):,.1f} MB)")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  Index CSV:      {INDEX_CSV}")
    print(f"  F1 cache:       {CACHE_DIR}")
    print("=" * 70)

    if target > 0 and counter >= target:
        print(f"\nAlready have {counter:,} arrays >= target {target:,}. Done!")
        return

    def reached_target() -> bool:
        return target > 0 and counter >= target

    for year in seasons:
        if reached_target():
            break
        print(f"\n{'='*70}")
        print(f"  SEASON {year}")
        print(f"{'='*70}")

        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"  Failed to get schedule for {year}: {e}")
            continue

        for _, event in schedule.iterrows():
            if reached_target():
                break
            event_name = event["EventName"]
            roundnum = event["RoundNumber"]

            # Skip testing events (pre-season tests, not real GPs)
            if event.get("EventFormat", "") == "testing":
                continue

            print(f"\n  GP: {event_name} (Round {roundnum})")

            for sess in sessions:
                if reached_target():
                    break
                print(f"    Session: {sess}")

                session = load_session_safe(year, roundnum, sess)
                if session is None:
                    continue

                # ── ALL drivers, ALL laps ─────────────────────────────────
                # We use get_car_data() and get_pos_data() instead of
                # get_telemetry() because get_telemetry() computes expensive
                # derived fields (DriverAhead, Distance) that we don't need
                # and that cause hangs / errors.
                try:
                    drivers = list(session.drivers)  # ALL drivers
                except Exception:
                    drivers = []

                for drv in drivers:
                    if reached_target():
                        break
                    try:
                        drv_laps = session.laps.pick_drivers([drv])
                        if drv_laps.empty:
                            continue

                        # Full driver session: car data (Speed, RPM, etc.)
                        try:
                            drv_car = drv_laps.get_car_data()
                            if drv_car is not None and not drv_car.empty:
                                counter, total_bytes = save_telemetry_arrays(
                                    drv_car, year, event_name, roundnum, sess,
                                    driver=str(drv), lap="FULL",
                                    existing_files=existing_files,
                                    counter=counter, total_bytes=total_bytes,
                                )
                        except Exception:
                            pass

                        # Full driver session: position data (X, Y, Z)
                        try:
                            drv_pos = drv_laps.get_pos_data()
                            if drv_pos is not None and not drv_pos.empty:
                                counter, total_bytes = save_telemetry_arrays(
                                    drv_pos, year, event_name, roundnum, sess,
                                    driver=str(drv), lap="FULLPOS",
                                    existing_files=existing_files,
                                    counter=counter, total_bytes=total_bytes,
                                )
                        except Exception:
                            pass

                        # Every lap for this driver
                        for lap_num, lap_data in drv_laps.iterlaps():
                            if reached_target():
                                break
                            try:
                                lap_car = lap_data.get_car_data()
                                if lap_car is not None and not lap_car.empty:
                                    counter, total_bytes = save_telemetry_arrays(
                                        lap_car, year, event_name, roundnum, sess,
                                        driver=str(drv), lap=f"L{lap_num}",
                                        existing_files=existing_files,
                                        counter=counter, total_bytes=total_bytes,
                                    )
                            except Exception:
                                continue

                    except Exception as e:
                        print(f"      Driver {drv} failed: {e}")
                        continue

                # ── Progress ──────────────────────────────────────────────
                print(
                    f"    => {counter:,} arrays total  |  "
                    f"{total_bytes / (1024**2):,.1f} MB on disk"
                )

                time.sleep(SLEEP_BETWEEN_SESSIONS)

            time.sleep(SLEEP_BETWEEN_GPS)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FETCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Total arrays:  {counter:,}")
    print(f"  Total size:    {total_bytes / (1024**2):,.1f} MB "
          f"({total_bytes / (1024**3):,.2f} GB)")
    print(f"  Output:        {OUTPUT_DIR}")
    print(f"  Index:         {INDEX_CSV}")

    if INDEX_CSV.exists():
        df = pd.read_csv(INDEX_CSV)
        print(f"\n  Breakdown:")
        print(f"    Years:       {sorted(df['year'].unique())}")
        print(f"    GPs:         {df['event'].nunique()}")
        print(f"    Sessions:    {df['session'].value_counts().to_dict()}")
        print(f"    Channels:    {df['channel'].nunique()} unique")
        print(f"    Drivers:     {df['driver'].nunique()}")
        print(f"    Array sizes: min={df['n_elements'].min():,}  "
              f"max={df['n_elements'].max():,}  "
              f"mean={df['n_elements'].mean():,.0f}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch raw F1 telemetry arrays (Step 1 — data collection only)"
    )
    p.add_argument(
        "--seasons", type=int, nargs="+", default=DEFAULT_SEASONS,
        help="Which seasons to fetch (default: 2018-2024)"
    )
    p.add_argument(
        "--sessions", nargs="+", default=DEFAULT_SESSIONS,
        help="Which session types (default: all — FP1 FP2 FP3 Q R S SS)"
    )
    p.add_argument(
        "--target", type=int, default=10_000,
        help="Stop after collecting this many arrays (default: 10000, 0=unlimited)"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print estimated array count without fetching"
    )
    return p.parse_args()


def dry_run(seasons: list[int], sessions: list[str]) -> None:
    """Estimate how many arrays we'd get without fetching anything."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    print("DRY RUN — estimating data volume\n")
    total_gps = 0
    for yr in seasons:
        try:
            s = fastf1.get_event_schedule(yr)
            n_gps = len(s[s.get("EventFormat", "") != "testing"])
            total_gps += n_gps
            print(f"  {yr}: {n_gps} GPs")
        except Exception as e:
            print(f"  {yr}: error — {e}")

    # Conservative estimate:
    # ~20 drivers × ~50 laps each × ~5 numeric channels = ~5,000 per session
    # Plus full session + per-driver aggregates = ~5,200 per session
    # But many laps/channels will be short or missing → estimate ~2,000 usable
    est_per_session = 2000
    total_est = total_gps * len(sessions) * est_per_session

    print(f"\n  Total GPs:           {total_gps}")
    print(f"  Session types:       {len(sessions)}")
    print(f"  Est. arrays/session: ~{est_per_session:,}")
    print(f"  Est. total arrays:   ~{total_est:,}")
    print(f"\n  Estimated disk usage:")
    print(f"    .csv files: ~{total_est * 0.08:.0f} MB  (avg ~80KB per array, text format)")
    print(f"    F1 cache:   ~{len(seasons) * 400:.0f} MB  (~400MB per season)")
    print(f"    Total:      ~{total_est * 0.08 / 1024 + len(seasons) * 0.4:.1f} GB")


def main() -> None:
    args = parse_args()

    if args.dry_run:
        dry_run(args.seasons, args.sessions)
        return

    fetch_all(seasons=args.seasons, sessions=args.sessions, target=args.target)


if __name__ == "__main__":
    main()
