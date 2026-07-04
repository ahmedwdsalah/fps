#!/usr/bin/env python3
"""
fetch_test_stock.py — Fetch 200K stock market arrays → single .npz file.

Fetches OHLCV data via yfinance, applies structural transforms (REV/SHUF/QBIN50/PSORT10),
windows large arrays into sub-arrays, saves directly to data/test_200k/stock/stock_200k.npz.
No intermediate CSV files. Resumable.
"""

from __future__ import annotations

import argparse, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent / "data" / "test_200k" / "stock"
NPZ_PATH = ROOT / "stock_200k.npz"
ROOT.mkdir(parents=True, exist_ok=True)

TARGET = 200_000
MIN_ARRAY_LEN = 50
COLS = ["Open", "High", "Low", "Close", "Volume"]
PERIODS = [
    ("1y", "1d"), ("2y", "1d"), ("5y", "1d"), ("10y", "1d"),
    ("max", "1d"), ("2y", "1h"), ("5d", "5m"), ("1mo", "5m"),
]
WINDOW_SIZES = [100, 250, 500, 1000, 2000]
MAX_WINDOWS = 8

TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK-B","UNH","XOM","JNJ",
    "JPM","V","PG","MA","HD","CVX","MRK","ABBV","LLY","PEP","KO","COST",
    "AVGO","TMO","MCD","WMT","CSCO","ACN","ABT","DHR","NEE","LIN","TXN",
    "PM","UNP","RTX","AMGN","LOW","HON","IBM","COP","SPGI","CAT","GE",
    "BA","AMAT","BKNG","SBUX","PLD","ADP","MDLZ","GILD","MMC","ISRG",
    "ADI","VRTX","REGN","TJX","SYK","LRCX","CB","ZTS","BDX","CI","MMM",
    "SO","DUK","PGR","MO","CME","CL","ITW","SLB","USB","BSX","TMUS",
    "EQIX","AON","WM","SCHW","FIS","HUM","ICE","GD","EMR","NOC","MCK",
    "PNC","CCI","PSA","APD","ORLY","NSC","AZO","KLAC","SNPS","SHW",
    "ROP","ADSK","AJG","CMG","TDG","MSCI","MCHP","CDNS",
    "TSLA","AMD","INTC","QCOM","MU","NOW","PANW","CRWD","SNOW","DDOG",
    "ZS","NET","FTNT","WDAY","TEAM","OKTA","TTD","BILL","HUBS","TWLO",
    "SQ","SHOP","SPOT","ROKU","SNAP","PINS","UBER","LYFT","DASH","ABNB",
    "COIN","PLTR","RBLX","DKNG","HOOD","SOFI","AFRM",
    "GS","MS","C","BAC","WFC","AXP","BLK","PYPL","DIS","NFLX","PARA",
    "WBD","LYV","XLE","XLF","XLK","XLV","XLI","XLP","XLU","XLB","XLY",
    "BHP","RIO","VALE","FCX","NEM","EWJ","FXI","EWZ","EWG","EWU","EWA",
    "EWC","EWY","EWT","INDA","SPY","QQQ","IWM","DIA","VTI","EFA","VEA",
    "MSTR","MARA","RIOT","GME","AMC",
]


def transforms(arr: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = [arr]
    out.append(arr[::-1].copy())                                    # REV
    shuf = arr.copy(); rng.shuffle(shuf); out.append(shuf)          # SHUF
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        edges = np.linspace(vmin, vmax, 51)
        idx = np.clip(np.digitize(arr, edges) - 1, 0, 49)
        out.append(((edges[:-1] + edges[1:]) / 2.0)[idx])            # QBIN50
    s = np.sort(arr.copy())
    n_sw = max(2, int(len(s) * 0.1))
    si = rng.choice(len(s), size=n_sw, replace=False)
    sv = s[si].copy(); rng.shuffle(sv); s[si] = sv
    out.append(s)                                                    # PSORT10
    return out


def windows(arr: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    n = len(arr)
    for ws in WINDOW_SIZES:
        if n < ws:
            continue
        n_chunks = min(n // ws, MAX_WINDOWS)
        starts = sorted(rng.choice(n - ws + 1, size=n_chunks, replace=False))
        for s in starts:
            chunk = arr[s:s+ws]
            if len(chunk) >= MIN_ARRAY_LEN:
                out.append(chunk)
    return out


def clean(arr: np.ndarray) -> np.ndarray | None:
    arr = arr[np.isfinite(arr)]
    return arr if arr.size >= MIN_ARRAY_LEN else None


def bar(v, t, w=30): return "#" * int(round(min(1,v/max(t,1))*w)) + "-" * (w - int(round(min(1,v/max(t,1))*w)))


def main():
    arrays, names = [], []

    # Resume
    if NPZ_PATH.exists():
        data = np.load(NPZ_PATH, allow_pickle=True)
        arrays = list(data["arrays"])
        names = list(data["file_names"])
        print(f"Resumed: {len(arrays):,} arrays already saved")

    print(f"Stock: {len(arrays):,}/{TARGET:,}  (fetching...)")
    if len(arrays) >= TARGET:
        print("Already at target. Done.")
        return

    started = time.time()
    outer_seed = 0
    last_print = 0

    for ticker in TICKERS:
        if len(arrays) >= TARGET:
            break
        for period, interval in PERIODS:
            if len(arrays) >= TARGET:
                break
            try:
                df = yf.download(ticker, period=period, interval=interval,
                                 progress=False, timeout=15)
            except Exception:
                time.sleep(2)
                continue
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            for col in COLS:
                if col not in df.columns:
                    continue
                arr = clean(df[col].dropna().values.astype(np.float64))
                if arr is None:
                    continue
                outer_seed += 1
                for t_arr in transforms(arr, outer_seed):
                    name = f"stock_{ticker}_{period}_{interval}_{col}_{outer_seed}"
                    arrays.append(t_arr)
                    names.append(name)
                    if len(arrays) >= TARGET:
                        break
                # window the original
                for w_arr in windows(arr, outer_seed + 10000):
                    name = f"stock_{ticker}_{period}_{interval}_{col}_w{len(w_arr)}_{outer_seed}"
                    arrays.append(w_arr)
                    names.append(name)
                    if len(arrays) >= TARGET:
                        break

            if len(arrays) - last_print >= 100:
                elapsed = (time.time() - started) / 60
                print(f"  [{bar(len(arrays), TARGET)}] {len(arrays):,}/{TARGET:,} "
                      f"({100*len(arrays)/TARGET:.1f}%) | {elapsed:.1f} min")
                last_print = len(arrays)
            time.sleep(0.15)

    # Save
    arr_obj = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        arr_obj[i] = a
    print(f"Saving {NPZ_PATH} ...")
    np.savez_compressed(NPZ_PATH, arrays=arr_obj, file_names=np.array(names))
    size_mb = NPZ_PATH.stat().st_size / (1024**2)
    elapsed = (time.time() - started) / 60
    print(f"Done. {len(arrays):,} arrays → {size_mb:.0f} MB in {elapsed:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=TARGET)
    args = p.parse_args()
    TARGET = args.target
    main()
