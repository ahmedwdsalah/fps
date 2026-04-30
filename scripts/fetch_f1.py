#!/usr/bin/env python3
"""
Download fresh F1 arrays from FastF1 until each target channel reaches 100,000 arrays,
then build an F1-only training dataset from those arrays.

Output files:
- /Volumes/k/thesis_data/f1_only/raw/*.csv
- /Volumes/k/thesis_data/f1_only/index.csv
- /Volumes/k/thesis_data/f1_only/training_dataset.csv

Run directly:
    python3 scripts/fetch_f1.py
    python3 scripts/fetch_f1.py --skip-fetch
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import fastf1
    FASTF1_IMPORT_ERROR = None
except Exception as exc:
    fastf1 = None
    FASTF1_IMPORT_ERROR = exc

from build_training_dataset import build_training_dataset

ROOT = Path("/Volumes/k/thesis_data/f1_only")
RAW_DIR = ROOT / "raw"
INDEX_CSV = ROOT / "index.csv"
TRAINING_OUTPUT_CSV = ROOT / "training_dataset.csv"
FAILED_SESSIONS_CSV = ROOT / "failed_sessions.csv"
CACHE_DIR = Path("/Users/ahmed/Desktop/thesis/My-Master-thesis/data/f1_cache")
LEGACY_ROOT = Path("/Volumes/k/thesis_data/real_world_10k")
LEGACY_RAW_DIR = LEGACY_ROOT / "raw"
LEGACY_INDEX_CSV = LEGACY_ROOT / "index_f1_only.csv"

TARGET_PER_CHANNEL = 100_000
MIN_ARRAY_LEN = 50

SEASONS = list(range(2024, 2017, -1))
SESSIONS = ["R", "Q", "S", "SS", "FP3", "FP2", "FP1"]

CHANNELS = ["Speed", "Throttle", "RPM", "nGear", "DRS", "X", "Y", "Z", "Distance"]
FETCH_RENDER_INTERVAL_SEC = 0.25
FETCH_HEARTBEAT_SEC = 5.0
DEFAULT_SESSION_TIMEOUT_SEC = 90
PROGRESS_BAR_WIDTH = 24

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

SESSION_NAME_TO_CODE = {
    "Race": "R",
    "Qualifying": "Q",
    "Sprint": "S",
    "Sprint Qualifying": "S",
    "Sprint Shootout": "SS",
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
}
SESSION_PRIORITY = {code: idx for idx, code in enumerate(SESSIONS)}
FAILED_SESSION_COLUMNS = ["session_key", "year", "round", "session", "reason", "failed_at"]


class SessionLoadTimeout(RuntimeError):
    pass


def _migrate_legacy_f1_layout() -> None:
    """Move previously fetched F1-only files out of the shared real_world_10k layout."""
    if INDEX_CSV.exists() and INDEX_CSV.stat().st_size > 0:
        try:
            current_df = pd.read_csv(INDEX_CSV, low_memory=False)
            if not current_df.empty:
                return
        except Exception:
            pass
    if not LEGACY_INDEX_CSV.exists() or LEGACY_INDEX_CSV.stat().st_size == 0:
        return

    try:
        legacy_df = pd.read_csv(LEGACY_INDEX_CSV, low_memory=False)
    except Exception:
        return

    if legacy_df.empty:
        return

    ROOT.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    migrated_rows = []
    for _, row in legacy_df.iterrows():
        fname = str(row.get("file", ""))
        if not fname:
            continue
        src = LEGACY_RAW_DIR / fname
        dst = RAW_DIR / fname
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
        if dst.exists():
            migrated_rows.append(row)

    migrated_df = pd.DataFrame(migrated_rows)
    if migrated_df.empty:
        return

    migrated_df = migrated_df.reindex(columns=INDEX_COLUMNS)
    migrated_df.to_csv(INDEX_CSV, index=False)


def _init_outputs() -> tuple[int, set[str], dict[str, int]]:
    _migrate_legacy_f1_layout()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not FAILED_SESSIONS_CSV.exists():
        with FAILED_SESSIONS_CSV.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FAILED_SESSION_COLUMNS)
            writer.writeheader()

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


def _load_failed_sessions() -> set[str]:
    if not FAILED_SESSIONS_CSV.exists() or FAILED_SESSIONS_CSV.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(FAILED_SESSIONS_CSV, low_memory=False)
    except Exception:
        return set()
    if "session_key" not in df.columns:
        return set()
    return set(df["session_key"].dropna().astype(str).tolist())


def _append_failed_session(
    session_key: str,
    year: int,
    roundnum: int,
    sess: str,
    reason: str,
) -> None:
    with FAILED_SESSIONS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FAILED_SESSION_COLUMNS)
        writer.writerow(
            {
                "session_key": session_key,
                "year": year,
                "round": roundnum,
                "session": sess,
                "reason": reason,
                "failed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )


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


def _available_sessions(event_row) -> list[tuple[str, str]]:
    sessions: list[tuple[str, str]] = []
    seen_codes: set[str] = set()
    for idx in range(1, 6):
        key = f"Session{idx}"
        name = event_row.get(key)
        if pd.isna(name):
            continue
        name = str(name).strip()
        code = SESSION_NAME_TO_CODE.get(name)
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)
        sessions.append((code, name))
    sessions.sort(key=lambda item: SESSION_PRIORITY.get(item[0], 999))
    return sessions


@contextlib.contextmanager
def _time_limit(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(signum, frame):
        raise SessionLoadTimeout(f"session load exceeded {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _is_rate_limit_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "RateLimitExceededError"


def _progress_bar(count: int, target: int, width: int = PROGRESS_BAR_WIDTH) -> str:
    ratio = 0.0 if target <= 0 else min(1.0, max(0.0, count / target))
    filled = int(round(ratio * width))
    return "#" * filled + "-" * (width - filled)


def _render_fetch_progress(
    counts: dict[str, int],
    context: str,
    started_at: float,
    previous_lines: int = 0,
) -> int:
    total_done = sum(counts.values())
    total_target = len(CHANNELS) * TARGET_PER_CHANNEL
    total_bar = _progress_bar(total_done, total_target, width=40)
    elapsed_min = (time.time() - started_at) / 60.0

    lines = [
        "[2/4] Fetch from FastF1",
        f"  Current: {context}",
        f"  Overall [{total_bar}] {total_done:,}/{total_target:,} arrays | elapsed {elapsed_min:.1f} min",
    ]
    for ch in CHANNELS:
        lines.append(
            f"  {ch:8s} [{_progress_bar(counts[ch], TARGET_PER_CHANNEL, width=32)}] "
            f"{counts[ch]:>7,}/{TARGET_PER_CHANNEL:,}"
        )

    if sys.stdout.isatty():
        if previous_lines:
            sys.stdout.write(f"\x1b[{previous_lines}F")
        for line in lines:
            sys.stdout.write("\x1b[2K")
            sys.stdout.write(line + "\n")
        sys.stdout.flush()
    else:
        if previous_lines == 0:
            for line in lines:
                print(line)

    return len(lines)


def _maybe_render_progress(
    counts: dict[str, int],
    context: str,
    started_at: float,
    previous_lines: int,
    last_render: float,
    force: bool = False,
) -> tuple[int, float]:
    now = time.time()
    if force or (now - last_render >= FETCH_HEARTBEAT_SEC):
        previous_lines = _render_fetch_progress(
            counts,
            context=context,
            started_at=started_at,
            previous_lines=previous_lines,
        )
        return previous_lines, now
    return previous_lines, last_render


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch F1 telemetry arrays and build an F1-only training dataset."
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip FastF1 downloading and build the training CSV from the existing F1-only index.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Fetch raw/index data only and skip the training-dataset build.",
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=TRAINING_OUTPUT_CSV,
        help=f"Output CSV for the F1-only training dataset (default: {TRAINING_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--training-limit",
        type=int,
        default=0,
        help="Process only the first N F1 arrays when building the training dataset (0=all).",
    )
    parser.add_argument(
        "--training-dry-run",
        action="store_true",
        help="Show the training-build plan without extracting features or timing sorts.",
    )
    parser.add_argument(
        "--session-timeout",
        type=int,
        default=DEFAULT_SESSION_TIMEOUT_SEC,
        help=f"Skip a session if load takes longer than this many seconds (default: {DEFAULT_SESSION_TIMEOUT_SEC}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[1/4] Load existing F1-only index")
    array_id, existing_files, counts = _init_outputs()
    failed_sessions = _load_failed_sessions()
    print(f"  Source: {INDEX_CSV}")
    for ch in CHANNELS:
        print(f"  - {ch}: {counts[ch]:,}/{TARGET_PER_CHANNEL:,}")

    if args.skip_fetch:
        print("\n[2/4] Skipping fetch; using existing F1-only index.")
    elif _done(counts):
        print("\n[2/4] Fetch already complete. No new FastF1 download needed.")
    else:
        if fastf1 is None:
            raise SystemExit(f"fastf1 import failed: {FASTF1_IMPORT_ERROR}")
        fastf1.set_log_level("CRITICAL")
        logging.disable(logging.CRITICAL)
        logging.getLogger("requests_cache").setLevel(logging.CRITICAL)
        logging.getLogger("requests_cache").propagate = False
        logging.getLogger("urllib3").setLevel(logging.CRITICAL)
        fastf1.Cache.enable_cache(str(CACHE_DIR))
        fetch_started_at = time.time()
        progress_lines = _render_fetch_progress(
            counts,
            context="starting",
            started_at=fetch_started_at,
        )
        last_render = 0.0
        stop_fetch = False

        for year in SEASONS:
            if _done(counts) or stop_fetch:
                break

            try:
                schedule = fastf1.get_event_schedule(year, backend="ergast")
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    progress_lines, last_render = _maybe_render_progress(
                        counts,
                        context=f"rate limit hit while loading schedule {year}; stop and retry later",
                        started_at=fetch_started_at,
                        previous_lines=progress_lines,
                        last_render=last_render,
                        force=True,
                    )
                    stop_fetch = True
                    break
                continue

            for _, event in schedule.iterrows():
                if _done(counts) or stop_fetch:
                    break

                if str(event.get("EventFormat", "")).lower() == "testing":
                    continue

                event_name = str(event.get("EventName", ""))
                roundnum = int(event.get("RoundNumber", 0))
                event_sessions = _available_sessions(event)
                try:
                    event_obj = schedule.get_event_by_round(roundnum)
                except Exception:
                    continue

                for sess, session_name in event_sessions:
                    if _done(counts) or stop_fetch:
                        break
                    session_key = f"{year}|R{roundnum}|{sess}"
                    if session_key in failed_sessions:
                        progress_lines, last_render = _maybe_render_progress(
                            counts,
                            context=f"{year} R{roundnum} {sess} skipped (previous failure)",
                            started_at=fetch_started_at,
                            previous_lines=progress_lines,
                            last_render=last_render,
                            force=True,
                        )
                        continue

                    progress_lines, last_render = _maybe_render_progress(
                        counts,
                        context=f"{year} R{roundnum} {sess} loading",
                        started_at=fetch_started_at,
                        previous_lines=progress_lines,
                        last_render=last_render,
                        force=True,
                    )

                    try:
                        session = event_obj.get_session(session_name)
                        with _time_limit(args.session_timeout):
                            session.load(telemetry=True, laps=True, weather=False)
                    except SessionLoadTimeout:
                        failed_sessions.add(session_key)
                        _append_failed_session(
                            session_key=session_key,
                            year=year,
                            roundnum=roundnum,
                            sess=sess,
                            reason=f"timeout>{args.session_timeout}s",
                        )
                        progress_lines, last_render = _maybe_render_progress(
                            counts,
                            context=f"{year} R{roundnum} {sess} timed out -> skipped",
                            started_at=fetch_started_at,
                            previous_lines=progress_lines,
                            last_render=last_render,
                            force=True,
                        )
                        continue
                    except Exception as exc:
                        if _is_rate_limit_error(exc):
                            progress_lines, last_render = _maybe_render_progress(
                                counts,
                                context=f"rate limit hit at {year} R{roundnum} {sess}; stop and retry later",
                                started_at=fetch_started_at,
                                previous_lines=progress_lines,
                                last_render=last_render,
                                force=True,
                            )
                            stop_fetch = True
                            break
                        continue

                    try:
                        drivers = list(session.drivers)
                    except Exception:
                        drivers = []

                    dirty = False
                    for drv in drivers:
                        if _done(counts):
                            break

                        progress_lines, last_render = _maybe_render_progress(
                            counts,
                            context=f"{year} R{roundnum} {sess} driver {drv}",
                            started_at=fetch_started_at,
                            previous_lines=progress_lines,
                            last_render=last_render,
                        )

                        try:
                            laps = session.laps.pick_drivers([drv])
                        except Exception:
                            continue
                        if laps is None or laps.empty:
                            continue

                        for lap_idx, lap_row in laps.iterlaps():
                            if _done(counts):
                                break

                            try:
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
                                            dirty = True

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
                                            dirty = True
                            except Exception:
                                continue

                            now = time.time()
                            if dirty and (now - last_render >= FETCH_RENDER_INTERVAL_SEC):
                                progress_lines = _render_fetch_progress(
                                    counts,
                                    context=f"{year} R{roundnum} {sess} {drv} {lap_label}",
                                    started_at=fetch_started_at,
                                    previous_lines=progress_lines,
                                )
                                last_render = now
                                dirty = False
                            elif not dirty:
                                progress_lines, last_render = _maybe_render_progress(
                                    counts,
                                    context=f"{year} R{roundnum} {sess} {drv} {lap_label}",
                                    started_at=fetch_started_at,
                                    previous_lines=progress_lines,
                                    last_render=last_render,
                                )

                        # small throttle so API/cache IO does not spike
                        time.sleep(0.02)
                    if dirty:
                        progress_lines = _render_fetch_progress(
                            counts,
                            context=f"{year} R{roundnum} {sess}",
                            started_at=fetch_started_at,
                            previous_lines=progress_lines,
                        )
                        last_render = time.time()
                    else:
                        progress_lines, last_render = _maybe_render_progress(
                            counts,
                            context=f"{year} R{roundnum} {sess} complete",
                            started_at=fetch_started_at,
                            previous_lines=progress_lines,
                            last_render=last_render,
                            force=True,
                        )

    print("\n[3/4] Final counts")
    for ch in CHANNELS:
        print(f"  - {ch}: {counts[ch]:,}/{TARGET_PER_CHANNEL:,}")
    print(f"Saved index: {INDEX_CSV}")

    if args.skip_training:
        print("\n[4/4] Training build skipped.")
        return

    print("\n[4/4] Build F1-only training dataset")
    build_training_dataset(
        index_csv=INDEX_CSV,
        raw_dir=RAW_DIR,
        output_csv=args.training_output,
        limit=args.training_limit,
        dry_run=args.training_dry_run,
    )


if __name__ == "__main__":
    main()
