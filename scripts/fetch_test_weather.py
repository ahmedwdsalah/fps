#!/usr/bin/env python3
"""
fetch_test_weather.py — Fetch 200K weather arrays → single .npz file.

Fetches hourly weather from Open-Meteo API, applies transforms + windowing,
saves directly to data/test_200k/weather/weather_200k.npz. No CSV files. Resumable.
"""

from __future__ import annotations

import argparse, time, warnings
from pathlib import Path
import numpy as np, requests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent / "data" / "test_200k" / "weather"
NPZ_PATH = ROOT / "weather_200k.npz"
ROOT.mkdir(parents=True, exist_ok=True)

TARGET = 200_000
MIN_ARRAY_LEN = 50
API_URL = "https://archive-api.open-meteo.com/v1/archive"
WINDOW_SIZES = [200, 500, 1000, 2000, 5000]
MAX_WINDOWS = 10

VARIABLES = [
    "temperature_2m","relative_humidity_2m","precipitation",
    "surface_pressure","wind_speed_10m","cloud_cover","shortwave_radiation",
]

CITIES = [
    ("Singapore",1.29,103.85),("Bangkok",13.75,100.52),("Lagos",6.45,3.40),
    ("Manaus",-3.12,-60.02),("Jakarta",-6.21,106.85),("Mumbai",19.08,72.88),
    ("Havana",23.13,-82.38),("Nairobi",-1.29,36.82),
    ("Riyadh",24.69,46.72),("Phoenix",33.45,-112.07),("Cairo",30.04,31.24),
    ("Dubai",25.20,55.27),("Lima",-12.05,-77.04),("Karachi",24.86,67.01),
    ("Las_Vegas",36.17,-115.14),("Alice_Springs",-23.70,133.88),
    ("London",51.51,-0.13),("Paris",48.86,2.35),("Berlin",52.52,13.41),
    ("Rome",41.90,12.50),("Madrid",40.42,-3.70),("Tokyo",35.68,139.69),
    ("Sydney",-33.87,151.21),("San_Francisco",37.77,-122.42),
    ("New_York",40.71,-74.01),("Buenos_Aires",-34.60,-58.38),
    ("Istanbul",41.01,28.98),("Cape_Town",-33.92,18.42),
    ("Moscow",55.76,37.62),("Chicago",41.88,-87.63),("Beijing",39.90,116.41),
    ("Toronto",43.65,-79.38),("Ulaanbaatar",47.92,106.91),("Astana",51.17,71.43),
    ("Minneapolis",44.98,-93.27),("Novosibirsk",55.03,82.92),
    ("Reykjavik",64.15,-21.94),("Anchorage",61.22,-149.90),
    ("Tromsoe",69.65,18.96),("Yakutsk",62.04,129.73),("Murmansk",68.97,33.09),
    ("Dhaka",23.81,90.41),("Yangon",16.84,96.17),("Chennai",13.08,80.27),
    ("La_Paz",-16.50,-68.15),("Lhasa",29.65,91.10),("Quito",-0.18,-78.47),
    ("Seattle",47.61,-122.33),("Denver",39.74,-104.99),("Atlanta",33.75,-84.39),
    ("Miami",25.77,-80.19),("Montreal",45.50,-73.57),("Vancouver",49.25,-123.12),
    ("Osaka",34.69,135.50),("Seoul",37.57,126.98),("Shanghai",31.23,121.47),
    ("Guangzhou",23.13,113.26),("Sao_Paulo",-23.55,-46.63),
    ("Santiago",-33.45,-70.67),("Bogota",4.71,-74.07),("Lisbon",38.72,-9.14),
    ("Athens",37.98,23.73),
]
YEAR_RANGES = [(2020,2024),(2015,2019),(2010,2014)]


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


def windows(arr: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    n = len(arr)
    for ws in WINDOW_SIZES:
        if n < ws: continue
        n_chunks = min(n // ws, MAX_WINDOWS)
        starts = sorted(rng.choice(n - ws + 1, size=n_chunks, replace=False))
        for s in starts:
            chunk = arr[s:s+ws]
            if len(chunk) >= MIN_ARRAY_LEN:
                out.append(chunk)
    return out


def fetch_city(city, lat, lon, y_start, y_end) -> dict[str, np.ndarray] | None:
    params = {"latitude":lat,"longitude":lon,
              "start_date":f"{y_start}-01-01","end_date":f"{y_end}-12-31",
              "hourly":",".join(VARIABLES),"timezone":"UTC"}
    try:
        r = requests.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        hourly = r.json().get("hourly",{})
    except: return None
    result = {}
    for var in VARIABLES:
        vals = hourly.get(var)
        if vals is None: continue
        arr = np.array(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size >= MIN_ARRAY_LEN:
            result[var] = arr
    return result or None


def clean(arr): arr = arr[np.isfinite(arr)]; return arr if arr.size >= MIN_ARRAY_LEN else None
def bar(v,t,w=30): return "#"*int(round(min(1,v/max(t,1))*w))+"-"*(w-int(round(min(1,v/max(t,1))*w)))


def main():
    arrays, names = [], []
    if NPZ_PATH.exists():
        data = np.load(NPZ_PATH, allow_pickle=True)
        arrays = list(data["arrays"]); names = list(data["file_names"])
        print(f"Resumed: {len(arrays):,} arrays")

    print(f"Weather: {len(arrays):,}/{TARGET:,}  (fetching...)")
    if len(arrays) >= TARGET: print("Already at target."); return

    started, outer_seed = time.time(), 0
    last_print = 0
    for city, lat, lon in CITIES:
        if len(arrays) >= TARGET: break
        for y_start, y_end in YEAR_RANGES:
            if len(arrays) >= TARGET: break
            fetched = fetch_city(city, lat, lon, y_start, y_end)
            if fetched is None: continue
            for var, arr in fetched.items():
                outer_seed += 1
                for t_arr in transforms(arr, outer_seed):
                    arrays.append(t_arr)
                    names.append(f"weather_{city}_{y_start}_{y_end}_{var}_{outer_seed}")
                    if len(arrays) >= TARGET: break
                for w_arr in windows(arr, outer_seed + 50000):
                    arrays.append(w_arr)
                    names.append(f"weather_{city}_{y_start}_{y_end}_{var}_w{len(w_arr)}_{outer_seed}")
                    if len(arrays) >= TARGET: break
            if len(arrays) - last_print >= 100:
                e = (time.time()-started)/60
                print(f"  [{bar(len(arrays),TARGET)}] {len(arrays):,}/{TARGET:,} ({100*len(arrays)/TARGET:.1f}%) | {e:.1f} min")
                last_print = len(arrays)
            time.sleep(0.2)

    arr_obj = np.empty(len(arrays), dtype=object)
    for i,a in enumerate(arrays): arr_obj[i]=a
    print(f"Saving {NPZ_PATH} ...")
    np.savez_compressed(NPZ_PATH, arrays=arr_obj, file_names=np.array(names))
    print(f"Done. {len(arrays):,} arrays → {NPZ_PATH.stat().st_size/(1024**2):.0f} MB in {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--target", type=int, default=TARGET)
    args = p.parse_args(); TARGET = args.target; main()
