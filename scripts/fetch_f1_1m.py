#!/usr/bin/env python3
"""
Run the existing fetch_f1 pipeline with a ~1,000,000 total-array target,
but store raw arrays in a packed HDF5 container (single file).

This reuses scripts/fetch_f1.py logic and only overrides:
  - target size
  - storage backend (CSV files -> packed HDF5)
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import h5py
import numpy as np

import fetch_f1

TARGET_TOTAL = 1_000_000
ROOT_1M = Path("/Volumes/k/thesis_data/f1_only_1m_packed")
H5_PATH = ROOT_1M / "raw_arrays.h5"

_H5_FILE: h5py.File | None = None
_H5_DSETS: dict[str, h5py.Dataset] | None = None


def _open_h5() -> tuple[h5py.File, dict[str, h5py.Dataset]]:
    global _H5_FILE, _H5_DSETS
    if _H5_FILE is not None and _H5_DSETS is not None:
        return _H5_FILE, _H5_DSETS

    ROOT_1M.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(H5_PATH, "a")

    if "values" not in h5:
        h5.create_dataset(
            "values",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(1_000_000,),
            compression="gzip",
            compression_opts=4,
        )
    if "start" not in h5:
        h5.create_dataset(
            "start",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )
    if "length" not in h5:
        h5.create_dataset(
            "length",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )
    if "file" not in h5:
        h5.create_dataset(
            "file",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=True,
            compression="gzip",
            compression_opts=4,
        )

    dsets = {
        "values": h5["values"],
        "start": h5["start"],
        "length": h5["length"],
        "file": h5["file"],
    }
    _H5_FILE = h5
    _H5_DSETS = dsets
    return h5, dsets


def _close_h5() -> None:
    global _H5_FILE, _H5_DSETS
    if _H5_FILE is not None:
        _H5_FILE.flush()
        _H5_FILE.close()
    _H5_FILE = None
    _H5_DSETS = None


def _init_outputs_packed() -> tuple[int, set[str], dict[str, int]]:
    # Keep index/failed sessions format from fetch_f1, store raw arrays in HDF5.
    fetch_f1.ROOT.mkdir(parents=True, exist_ok=True)
    fetch_f1.INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    fetch_f1.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not fetch_f1.FAILED_SESSIONS_CSV.exists():
        with fetch_f1.FAILED_SESSIONS_CSV.open("w", newline="") as f:
            writer = fetch_f1.csv.DictWriter(f, fieldnames=fetch_f1.FAILED_SESSION_COLUMNS)
            writer.writeheader()

    _open_h5()

    if not fetch_f1.INDEX_CSV.exists() or fetch_f1.INDEX_CSV.stat().st_size == 0:
        with fetch_f1.INDEX_CSV.open("w", newline="") as f:
            writer = fetch_f1.csv.DictWriter(f, fieldnames=fetch_f1.INDEX_COLUMNS)
            writer.writeheader()
        return 0, set(), {ch: 0 for ch in fetch_f1.CHANNELS}

    df = fetch_f1.pd.read_csv(fetch_f1.INDEX_CSV, low_memory=False)
    if list(df.columns) != fetch_f1.INDEX_COLUMNS:
        df = df.reindex(columns=fetch_f1.INDEX_COLUMNS)
        df.to_csv(fetch_f1.INDEX_CSV, index=False)

    existing_files = set(df["file"].dropna().astype(str).tolist())
    counts = {ch: 0 for ch in fetch_f1.CHANNELS}
    if not df.empty:
        vc = df["channel"].value_counts().to_dict()
        for ch in fetch_f1.CHANNELS:
            counts[ch] = int(vc.get(ch, 0))
    return len(df), existing_files, counts


def _save_channel_array_packed(
    arr,
    channel: str,
    year: int,
    event_name: str,
    roundnum: int,
    sess: str,
    driver: str,
    lap_label: str,
    array_id: int,
    existing_files: set[str],
):
    arr = arr[fetch_f1.np.isfinite(arr)]
    if arr.size < fetch_f1.MIN_ARRAY_LEN:
        return False, array_id

    fname = f"f1_{year}_R{roundnum}_{sess}_{driver}_{lap_label}_{channel}.h5arr"
    if fname in existing_files:
        return False, array_id

    arr = arr.astype(fetch_f1.np.float64, copy=False)
    size_bytes = int(arr.nbytes)

    h5, dsets = _open_h5()
    values = dsets["values"]
    starts = dsets["start"]
    lengths = dsets["length"]
    files = dsets["file"]

    start = int(values.shape[0])
    n = int(arr.size)
    row = int(starts.shape[0])

    values.resize((start + n,))
    values[start : start + n] = arr

    starts.resize((row + 1,))
    starts[row] = start
    lengths.resize((row + 1,))
    lengths[row] = n
    files.resize((row + 1,))
    files[row] = fname

    h5.flush()

    meta_row = {
        "array_id": array_id,
        "file": fname,
        "year": year,
        "event": event_name,
        "round": roundnum,
        "session": sess,
        "driver": driver,
        "lap": lap_label,
        "channel": channel,
        "n_elements": n,
        "dtype": str(arr.dtype),
        "size_bytes": size_bytes,
    }
    fetch_f1._append_index(meta_row)
    existing_files.add(fname)
    return True, array_id + 1


def main() -> None:
    # Use a dedicated root for the 1M packed run.
    fetch_f1.ROOT = ROOT_1M
    fetch_f1.RAW_DIR = ROOT_1M / "raw"  # unused for packed backend
    fetch_f1.INDEX_CSV = ROOT_1M / "index.csv"
    fetch_f1.TRAINING_OUTPUT_CSV = ROOT_1M / "training_dataset.csv"
    fetch_f1.FAILED_SESSIONS_CSV = ROOT_1M / "failed_sessions.csv"
    fetch_f1.LEGACY_ROOT = ROOT_1M / "_legacy_disabled"
    fetch_f1.LEGACY_RAW_DIR = fetch_f1.LEGACY_ROOT / "raw"
    fetch_f1.LEGACY_INDEX_CSV = fetch_f1.LEGACY_ROOT / "index_f1_only.csv"

    # Swap storage methods only; keep fetch loop logic intact.
    fetch_f1._init_outputs = _init_outputs_packed
    fetch_f1._save_channel_array = _save_channel_array_packed
    # Start from older seasons to avoid repeatedly burning rate budget on 2024 first.
    fetch_f1.SEASONS = list(range(2018, 2025))

    per_channel = int(math.ceil(TARGET_TOTAL / len(fetch_f1.CHANNELS)))
    fetch_f1.TARGET_PER_CHANNEL = per_channel

    print("=" * 70)
    print("  FETCH F1 (~1M TOTAL) USING EXISTING fetch_f1.py LOGIC")
    print("=" * 70)
    print(f"  Channels: {len(fetch_f1.CHANNELS)}")
    print(f"  Target total arrays: {TARGET_TOTAL:,}")
    print(f"  Target per channel:  {per_channel:,}")
    print(f"  Effective total:     {per_channel * len(fetch_f1.CHANNELS):,}")
    print(f"  Dataset root:        {ROOT_1M}")
    print(f"  Packed raw HDF5:     {H5_PATH}")
    print("  Training step:       skipped (fetch-only wrapper)")
    print()

    if "--skip-training" not in sys.argv:
        sys.argv.append("--skip-training")

    try:
        fetch_f1.main()
    finally:
        _close_h5()


if __name__ == "__main__":
    main()
