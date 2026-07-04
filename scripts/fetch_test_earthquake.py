#!/usr/bin/env python3
"""
fetch_test_earthquake.py — Fetch 200K seismic arrays → single .npz file.

Fetches magnitude/depth/lat/lon from USGS API, applies transforms + bootstrap,
saves directly to data/test_200k/earthquake/earthquake_200k.npz. No CSV files. Resumable.
"""

from __future__ import annotations

import argparse, time, warnings
from pathlib import Path
import numpy as np, requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent / "data" / "test_200k" / "earthquake"
NPZ_PATH = ROOT / "earthquake_200k.npz"
ROOT.mkdir(parents=True, exist_ok=True)

TARGET = 200_000
MIN_ARRAY_LEN = 50
API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
ARRAYS = ["magnitude", "depth", "latitude", "longitude"]

QUERIES = [
    {"minmag": 5.0, "maxmag": 10.0},
    {"minmag": 4.0, "maxmag": 5.0},
    {"minmag": 2.5, "maxmag": 4.0},
    {"minmag": 1.0, "maxmag": 2.5},
]

WINDOWS = [
    ("2020-01-01","2020-12-31"),("2021-01-01","2021-12-31"),("2022-01-01","2022-12-31"),
    ("2023-01-01","2023-12-31"),("2024-01-01","2024-12-31"),("2025-01-01","2025-12-31"),
    ("2020-01-01","2020-06-30"),("2020-07-01","2020-12-31"),
    ("2021-01-01","2021-06-30"),("2021-07-01","2021-12-31"),
    ("2022-01-01","2022-06-30"),("2022-07-01","2022-12-31"),
    ("2023-01-01","2023-06-30"),("2023-07-01","2023-12-31"),
    ("2024-01-01","2024-06-30"),("2024-07-01","2024-12-31"),
    ("2015-01-01","2019-12-31"),("2010-01-01","2014-12-31"),
    ("2000-01-01","2009-12-31"),("2005-01-01","2014-12-31"),
    ("2011-03-01","2011-06-30"),("2023-02-01","2023-05-31"),
    ("2010-01-01","2010-03-31"),("2015-04-01","2015-07-31"),
    ("2004-12-01","2005-03-31"),("2016-04-01","2016-07-31"),
    ("2018-09-01","2018-12-31"),("2019-07-01","2019-09-30"),
    ("2017-09-01","2017-12-31"),
]
BOOTSTRAP_PER_ARRAY = 20


def transforms(arr: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = [arr]
    out.append(arr[::-1].copy())
    shuf = arr.copy(); rng.shuffle(shuf); out.append(shuf)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        edges = np.linspace(vmin, vmax, 51)
        idx = np.clip(np.digitize(arr, edges) - 1, 0, 49)
        out.append(((edges[:-1] + edges[1:]) / 2.0)[idx])
    s = np.sort(arr.copy())
    n_sw = max(2, int(len(s) * 0.1))
    si = rng.choice(len(s), size=n_sw, replace=False)
    sv = s[si].copy(); rng.shuffle(sv); s[si] = sv
    out.append(s)
    return out


def bootstrap(arr: np.ndarray, seed: int, n_bootstrap: int = BOOTSTRAP_PER_ARRAY) -> list[np.ndarray]:
    """Generate subsampled arrays from a large array to multiply count."""
    rng = np.random.default_rng(seed)
    out = []
    n = len(arr)
    sizes = [100, 200, 500, 1000, min(2000, n)]
    for sz in sizes:
        if n < sz: continue
        for _ in range(min(n_bootstrap, 5)):
            idx = rng.choice(n, size=sz, replace=False)
            chunk = arr[np.sort(idx)]
            if len(chunk) >= MIN_ARRAY_LEN:
                out.append(chunk)
    return out


def fetch_events(start, end, minmag, maxmag) -> list[dict]:
    params = {"format":"geojson","starttime":start,"endtime":end,
              "minmagnitude":minmag,"maxmagnitude":maxmag,"limit":20000}
    try:
        r = requests.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("features",[])
    except: return []


def clean(arr): arr = arr[np.isfinite(arr)]; return arr if arr.size >= MIN_ARRAY_LEN else None
def bar(v,t,w=30): return "#"*int(round(min(1,v/max(t,1))*w))+"-"*(w-int(round(min(1,v/max(t,1))*w)))


def main():
    arrays, names = [], []
    if NPZ_PATH.exists():
        data = np.load(NPZ_PATH, allow_pickle=True)
        arrays = list(data["arrays"]); names = list(data["file_names"])
        print(f"Resumed: {len(arrays):,} arrays")

    print(f"Earthquake: {len(arrays):,}/{TARGET:,}  (fetching...)")
    if len(arrays) >= TARGET: print("Already at target."); return

    started, outer_seed = time.time(), 0
    last_print = 0
    for q in QUERIES:
        if len(arrays) >= TARGET: break
        qlabel = f"M{q['minmag']:.1f}to{q['maxmag']:.1f}"
        for ws, we in WINDOWS:
            if len(arrays) >= TARGET: break
            events = fetch_events(ws, we, q["minmag"], q["maxmag"])
            if not events: continue

            raw = {
                "magnitude": np.array([e["properties"]["mag"] for e in events
                    if e["properties"].get("mag") is not None], dtype=np.float64),
                "depth": np.array([e["geometry"]["coordinates"][2] for e in events
                    if e["geometry"]["coordinates"][2] is not None], dtype=np.float64),
                "latitude": np.array([e["geometry"]["coordinates"][1] for e in events],
                                     dtype=np.float64),
                "longitude": np.array([e["geometry"]["coordinates"][0] for e in events],
                                      dtype=np.float64),
            }
            for aname in ARRAYS:
                arr = clean(raw.get(aname))
                if arr is None: continue
                outer_seed += 1
                # Transform the raw array
                for t_arr in transforms(arr, outer_seed):
                    arrays.append(t_arr)
                    names.append(f"quake_{qlabel}_{ws}_{we}_{aname}_{outer_seed}")
                    if len(arrays) >= TARGET: break
                # Bootstrap sub-arrays from raw
                for b_arr in bootstrap(arr, outer_seed + 50000):
                    arrays.append(b_arr)
                    names.append(f"quake_{qlabel}_{ws}_{we}_{aname}_bs{len(b_arr)}_{outer_seed}")
                    if len(arrays) >= TARGET: break

            if len(arrays) - last_print >= 100:
                e = (time.time()-started)/60
                print(f"  [{bar(len(arrays),TARGET)}] {len(arrays):,}/{TARGET:,} ({100*len(arrays)/TARGET:.1f}%) | {e:.1f} min")
                last_print = len(arrays)
            time.sleep(0.3)

    arr_obj = np.empty(len(arrays), dtype=object)
    for i,a in enumerate(arrays): arr_obj[i]=a
    print(f"Saving {NPZ_PATH} ...")
    np.savez_compressed(NPZ_PATH, arrays=arr_obj, file_names=np.array(names))
    print(f"Done. {len(arrays):,} arrays → {NPZ_PATH.stat().st_size/(1024**2):.0f} MB in {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--target", type=int, default=TARGET)
    args = p.parse_args(); TARGET = args.target; main()
