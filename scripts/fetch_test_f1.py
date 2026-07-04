#!/usr/bin/env python3
"""
fetch_test_f1.py — Fetch 200K F1 telemetry arrays for test dataset.

Lean version: no derived columns, no transforms, just raw channel arrays.
Each array saved as CSV (one float per line). Resumable. Crash-safe.

Output: data/test_200k/f1/raw/*.csv + index.csv
"""

from __future__ import annotations

import csv
import hashlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fastf1

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent / "data" / "test_200k" / "f1"
RAW_DIR = ROOT / "raw"
INDEX_CSV = ROOT / "index.csv"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "f1_cache"

# ── Config ───────────────────────────────────────────────────────────────
TARGET = 200_000
MIN_ARRAY_LEN = 50
CHANNELS = ["Speed", "Throttle", "RPM", "nGear", "DRS", "X", "Y", "Z", "Distance"]
SEASONS = list(range(2024, 2017, -1))
SESSIONS = ["R", "Q", "S", "SS", "FP3", "FP2", "FP1"]

INDEX_COLUMNS = ["array_id", "file", "year", "event", "round", "session",
                 "driver", "lap", "channel", "n_elements", "dtype", "size_bytes"]

SESSION_NAME_TO_CODE = {
    "Race": "R", "Qualifying": "Q", "Sprint": "S",
    "Sprint Qualifying": "S", "Sprint Shootout": "SS",
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
}
CODE_TO_NAME = {v: k for k, v in SESSION_NAME_TO_CODE.items()}

fastf1.set_log_level("CRITICAL")
logging.disable(logging.CRITICAL)
for name in ("requests_cache", "urllib3"):
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).propagate = False


# ═══════════════════════════════════════════════════════════════════════════
def init() -> tuple[int, set[str]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    if INDEX_CSV.exists() and INDEX_CSV.stat().st_size > 0:
        try:
            df = pd.read_csv(INDEX_CSV, low_memory=False)
            return len(df), set(df["file"].dropna().astype(str))
        except Exception:
            pass

    with INDEX_CSV.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=INDEX_COLUMNS).writeheader()
    return 0, set()


def append_row(row: dict) -> None:
    with INDEX_CSV.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=INDEX_COLUMNS).writerow(row)


def save_array(arr: np.ndarray, channel: str, year: int, event_name: str,
               roundnum: int, sess: str, driver: str, lap_label: str,
               array_id: int, existing: set[str]) -> int:
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_ARRAY_LEN:
        return array_id

    fname = f"f1_{year}_R{roundnum}_{sess}_{driver}_{lap_label}_{channel}.csv"
    if fname in existing:
        return array_id

    np.savetxt(RAW_DIR / fname, arr, fmt="%.10g")
    nbytes = (RAW_DIR / fname).stat().st_size

    append_row({
        "array_id": array_id, "file": fname, "year": year,
        "event": event_name, "round": roundnum, "session": sess,
        "driver": driver, "lap": lap_label, "channel": channel,
        "n_elements": int(arr.size), "dtype": str(arr.dtype),
        "size_bytes": int(nbytes),
    })
    existing.add(fname)
    return array_id + 1


def extract_lap_telemetry(lap) -> tuple:
    car_df, pos_df = None, None
    try:
        car_df = lap.get_car_data()
        if car_df is not None and not car_df.empty:
            car_df = car_df.add_distance()
    except Exception:
        pass
    try:
        pos_df = lap.get_pos_data()
    except Exception:
        pass
    return car_df, pos_df


def bar(val: int, total: int, w: int = 30) -> str:
    r = min(1.0, max(0.0, val / max(total, 1)))
    return "#" * int(round(r * w)) + "-" * (w - int(round(r * w)))


def main() -> None:
    array_id, existing = init()
    print(f"F1: {array_id:,}/{TARGET:,} arrays | {len(existing):,} files tracked")
    if array_id >= TARGET:
        print("Already at target. Done.")
        return

    started = time.time()
    stop = False

    for year in SEASONS:
        if stop or array_id >= TARGET:
            break
        try:
            schedule = fastf1.get_event_schedule(year, backend="ergast")
        except Exception:
            continue

        for _, event in schedule.iterrows():
            if stop or array_id >= TARGET:
                break
            if str(event.get("EventFormat", "")).lower() == "testing":
                continue

            event_name = str(event.get("EventName", ""))
            roundnum = int(event.get("RoundNumber", 0))

            for idx in range(1, 6):
                sname = event.get(f"Session{idx}")
                if pd.isna(sname):
                    continue
                session_name = str(sname).strip()
                code = SESSION_NAME_TO_CODE.get(session_name)
                if not code:
                    continue

                try:
                    event_obj = schedule.get_event_by_round(roundnum)
                    session = event_obj.get_session(session_name)
                    session.load(laps=True, telemetry=True, weather=False,
                                 messages=False)
                except Exception:
                    continue

                for _, driver_info in session.laps.iterrows():
                    if stop or array_id >= TARGET:
                        break
                    driver = str(driver_info.get("Driver", ""))
                    lap_num = str(driver_info.get("LapNumber", ""))

                    try:
                        car_df, pos_df = extract_lap_telemetry(driver_info)
                    except Exception:
                        continue

                    for ch in CHANNELS:
                        if array_id >= TARGET:
                            stop = True
                            break
                        arr = None
                        if ch == "Distance" and car_df is not None:
                            if "Distance" in car_df.columns:
                                arr = car_df["Distance"].values
                        elif car_df is not None and ch in car_df.columns:
                            arr = car_df[ch].values
                        elif pos_df is not None and ch in pos_df.columns:
                            arr = pos_df[ch].values

                        if arr is None:
                            continue
                        array_id = save_array(
                            arr.astype(np.float64), ch, year, event_name,
                            roundnum, code, driver, lap_num, array_id, existing,
                        )

                    if array_id % 200 == 0:
                        elapsed = (time.time() - started) / 60
                        pct = array_id / TARGET * 100
                        print(f"  [{bar(array_id, TARGET)}] {array_id:,}/{TARGET:,} "
                              f"({pct:.1f}%) | {elapsed:.1f} min")

    elapsed = (time.time() - started) / 60
    print(f"\nDone. {array_id:,} arrays in {elapsed:.1f} min")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fetch F1 test arrays")
    p.add_argument("--target", type=int, default=TARGET,
                   help=f"Number of arrays to fetch (default: {TARGET})")
    args = p.parse_args()
    TARGET = args.target
    main()
