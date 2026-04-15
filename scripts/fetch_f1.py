#!/usr/bin/env python3
"""
Download fresh F1 arrays from FastF1 until each target channel reaches 100,000 arrays.

Output files:
- /Volumes/k/thesis_data/real_world_10k/raw/*.csv
- /Volumes/k/thesis_data/real_world_10k/index_f1_only.csv

No CLI arguments. Run directly:
    python3 scripts/fetch_f1.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import fastf1
except Exception as exc:
    raise SystemExit(f"fastf1 import failed: {exc}")

ROOT = Path("/Volumes/k/thesis_data/real_world_10k")
RAW_DIR = ROOT / "raw"
INDEX_CSV = ROOT / "index_f1_only.csv"
CACHE_DIR = Path("/Users/ahmed/Desktop/thesis/My-Master-thesis/data/f1_cache")

TARGET_PER_CHANNEL = 100_000
MIN_ARRAY_LEN = 50

SEASONS = list(range(2018, 2025))
SESSIONS = ["FP1", "FP2", "FP3", "Q", "R", "S", "SS"]

CHANNELS = ["Speed", "Throttle", "RPM", "nGear", "DRS", "X", "Y", "Z", "Distance"]

INDEX_COLUMNS = [
    "array_id",
    "file",
    "year",
    "event",
    "round",
    "session",
    "driver",
    "lap",
    "channel",
    "n_elements",
    "dtype",
    "size_bytes",
]


def _init_outputs() -> tuple[int, set[str], dict[str, int]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not INDEX_CSV.exists() or INDEX_CSV.stat().st_size == 0:
        with INDEX_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
            writer.writeheader()
        return 0, set(), {ch: 0 for ch in CHANNELS}

    df = pd.read_csv(INDEX_CSV, low_memory=False)
    if list(df.columns) != INDEX_COLUMNS:
        # keep header stable
        df = df.reindex(columns=INDEX_COLUMNS)
        df.to_csv(INDEX_CSV, index=False)
    existing_files = set(df["file"].dropna().astype(str).tolist())
    counts = {ch: 0 for ch in CHANNELS}
    if not df.empty:
        vc = df["channel"].value_counts().to_dict()
        for ch in CHANNELS:
            counts[ch] = int(vc.get(ch, 0))
    return len(df), existing_files, counts


def _append_index(row: dict) -> None:
    with INDEX_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writerow(row)


def _done(counts: dict[str, int]) -> bool:
    return all(counts[ch] >= TARGET_PER_CHANNEL for ch in CHANNELS)


def _save_channel_array(
    arr: np.ndarray,
    channel: str,
    year: int,
    event_name: str,
    roundnum: int,
    sess: str,
    driver: str,
    lap_label: str,
    array_id: int,
    existing_files: set[str],
) -> tuple[bool, int]:
    arr = arr[np.isfinite(arr)]
    if arr.size < MIN_ARRAY_LEN:
        return False, array_id

    fname = f"f1_{year}_R{roundnum}_{sess}_{driver}_{lap_label}_{channel}.csv"
    if fname in existing_files:
        return False, array_id

    out_path = RAW_DIR / fname
    np.savetxt(out_path, arr, fmt="%.10g")
    size_bytes = out_path.stat().st_size

    row = {
        "array_id": array_id,
        "file": fname,
        "year": year,
        "event": event_name,
        "round": roundnum,
        "session": sess,
        "driver": driver,
        "lap": lap_label,
        "channel": channel,
        "n_elements": int(arr.size),
        "dtype": str(arr.dtype),
        "size_bytes": int(size_bytes),
    }
    _append_index(row)
    existing_files.add(fname)
    return True, array_id + 1


def _extract_lap_telemetry(lap_row) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    car_df = None
    pos_df = None
    try:
        car_df = lap_row.get_car_data()
        if car_df is not None and not car_df.empty:
            car_df = car_df.add_distance()
    except Exception:
        car_df = None
    try:
        pos_df = lap_row.get_pos_data()
    except Exception:
        pos_df = None
    return car_df, pos_df


def main() -> None:
    print("[1/3] Load existing index")
    array_id, existing_files, counts = _init_outputs()
    print(f"  Source: {INDEX_CSV}")
    for ch in CHANNELS:
        print(f"  - {ch}: {counts[ch]:,}/{TARGET_PER_CHANNEL:,}")

    if _done(counts):
        print("\n[3/3] Already complete. Nothing to fetch.")
        return

    print("\n[2/3] Fetch from FastF1")
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    for year in SEASONS:
        if _done(counts):
            break

        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception:
            continue

        for _, event in schedule.iterrows():
            if _done(counts):
                break

            if str(event.get("EventFormat", "")).lower() == "testing":
                continue

            event_name = str(event.get("EventName", ""))
            roundnum = int(event.get("RoundNumber", 0))

            for sess in SESSIONS:
                if _done(counts):
                    break

                try:
                    session = fastf1.get_session(year, roundnum, sess)
                    session.load(telemetry=True, laps=True, weather=False)
                except Exception:
                    continue

                try:
                    drivers = list(session.drivers)
                except Exception:
                    drivers = []

                for drv in drivers:
                    if _done(counts):
                        break

                    try:
                        laps = session.laps.pick_drivers([drv])
                    except Exception:
                        continue
                    if laps is None or laps.empty:
                        continue

                    for lap_idx, lap_row in laps.iterlaps():
                        if _done(counts):
                            break

                        lap_no = lap_row.get("LapNumber")
                        if pd.notna(lap_no):
                            lap_label = f"L{int(lap_no)}"
                        else:
                            lap_label = f"L{lap_idx}"

                        car_df, pos_df = _extract_lap_telemetry(lap_row)

                        if car_df is not None and not car_df.empty:
                            for ch in ["Speed", "Throttle", "RPM", "nGear", "DRS", "Distance"]:
                                if counts[ch] >= TARGET_PER_CHANNEL:
                                    continue
                                if ch not in car_df.columns:
                                    continue
                                arr = car_df[ch].to_numpy(dtype=np.float64, copy=False)
                                saved, array_id = _save_channel_array(
                                    arr=arr,
                                    channel=ch,
                                    year=year,
                                    event_name=event_name,
                                    roundnum=roundnum,
                                    sess=sess,
                                    driver=str(drv),
                                    lap_label=lap_label,
                                    array_id=array_id,
                                    existing_files=existing_files,
                                )
                                if saved:
                                    counts[ch] += 1

                        if pos_df is not None and not pos_df.empty:
                            for ch in ["X", "Y", "Z"]:
                                if counts[ch] >= TARGET_PER_CHANNEL:
                                    continue
                                if ch not in pos_df.columns:
                                    continue
                                arr = pos_df[ch].to_numpy(dtype=np.float64, copy=False)
                                saved, array_id = _save_channel_array(
                                    arr=arr,
                                    channel=ch,
                                    year=year,
                                    event_name=event_name,
                                    roundnum=roundnum,
                                    sess=sess,
                                    driver=str(drv),
                                    lap_label=lap_label,
                                    array_id=array_id,
                                    existing_files=existing_files,
                                )
                                if saved:
                                    counts[ch] += 1

                    # small throttle so API/cache IO does not spike
                    time.sleep(0.02)

                if any(sum(counts.values()) % 5000 == 0 for _ in [0]):
                    pass

                # concise per-session progress
                status = " | ".join([f"{ch}:{counts[ch]:,}" for ch in CHANNELS])
                print(f"  {year} R{roundnum} {sess} -> {status}")

    print("\n[3/3] Final counts")
    for ch in CHANNELS:
        print(f"  - {ch}: {counts[ch]:,}/{TARGET_PER_CHANNEL:,}")
    print(f"Saved index: {INDEX_CSV}")


if __name__ == "__main__":
    main()
