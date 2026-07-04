#!/usr/bin/env python3
"""
fetch_test_crypto.py — Fetch 200K crypto arrays → single .npz file.

Fetches OHLCV data via yfinance, applies transforms + windowing,
saves directly to data/test_200k/crypto/crypto_200k.npz. No CSV files. Resumable.
"""

from __future__ import annotations

import argparse, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent / "data" / "test_200k" / "crypto"
NPZ_PATH = ROOT / "crypto_200k.npz"
ROOT.mkdir(parents=True, exist_ok=True)

TARGET = 200_000
MIN_ARRAY_LEN = 50
COLS = ["Open", "High", "Low", "Close", "Volume"]
PERIODS = [
    ("1y","1d"),("2y","1d"),("5y","1d"),("max","1d"),("3mo","1d"),
    ("6mo","1d"),("10y","1d"),("2y","1h"),("1y","1h"),("3mo","1h"),
    ("5d","5m"),("1mo","5m"),("1mo","1h"),
]
WINDOW_SIZES = [100, 250, 500, 1000, 2000]
MAX_WINDOWS = 8

TICKERS = [
    "BTC-USD","ETH-USD","LTC-USD","XRP-USD","BCH-USD","ADA-USD","DOT-USD",
    "LINK-USD","BNB-USD","SOL-USD","DOGE-USD","AVAX-USD","MATIC-USD",
    "UNI-USD","ATOM-USD","XLM-USD","ALGO-USD","FIL-USD","VET-USD","NEAR-USD",
    "AAVE-USD","MKR-USD","COMP-USD","SNX-USD","CRV-USD","SUSHI-USD","YFI-USD",
    "UMA-USD","BAL-USD","LDO-USD","FXS-USD","LQTY-USD","RPL-USD","PENDLE-USD",
    "USDT-USD","USDC-USD","DAI-USD","BUSD-USD","TUSD-USD","USDP-USD",
    "FRAX-USD","GUSD-USD",
    "SHIB-USD","FLOKI-USD","BONK-USD","WIF-USD","TURBO-USD","BABYDOGE-USD",
    "ARB11841-USD","OP-USD","IMX-USD","STRK-USD","MANTA-USD","BLAST-USD",
    "ZK-USD",
    "EOS-USD","TRX-USD","XTZ-USD","NEO-USD","DASH-USD","ZEC-USD","XMR-USD",
    "ETC-USD","IOTA-USD","THETA-USD","FTM-USD","HBAR-USD","EGLD-USD",
    "SAND-USD","MANA-USD","AXS-USD","ENJ-USD","CHZ-USD","GALA-USD","ICP-USD",
    "CRO-USD","OKB-USD","LEO-USD","KCS-USD","GT-USD","HT-USD",
    "FET-USD","RNDR-USD","WLD-USD","ARKM-USD","AR-USD","GRT-USD","OCEAN-USD",
    "AKT-USD","TAO22974-USD",
    "RONIN-USD","PYR-USD","MAGIC-USD","ALICE-USD","SUPER8290-USD","GODS-USD",
    "PIXEL-USD","PORTAL-USD",
    "SUI20947-USD","APT21794-USD","SEI-USD","INJ-USD","TIA22861-USD",
    "PYTH-USD","JUP29210-USD","ONDO-USD","ENA-USD","W-USD",
    "STORJ-USD","SC-USD","HNT-USD","IOTX-USD","ANKR-USD","CELO-USD",
    "RLC-USD","BAND-USD","CTSI-USD","NKN-USD",
    "LRC-USD","ZIL-USD","ONE-USD","QTUM-USD","ICX-USD","ONT-USD","WAVES-USD",
    "KSM-USD","KAVA-USD","CKB-USD","HOT-USD","SKL-USD","CELR-USD","MTL-USD",
    "DENT-USD","STMX-USD","SLP-USD","C98-USD","SPELL-USD","JOE-USD",
]


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
        if n < ws:
            continue
        n_chunks = min(n // ws, MAX_WINDOWS)
        starts = sorted(rng.choice(n - ws + 1, size=n_chunks, replace=False))
        for s in starts:
            chunk = arr[s:s+ws]
            if len(chunk) >= MIN_ARRAY_LEN:
                out.append(chunk)
    return out


def clean(arr): arr = arr[np.isfinite(arr)]; return arr if arr.size >= MIN_ARRAY_LEN else None
def bar(v,t,w=30): return "#"*int(round(min(1,v/max(t,1))*w))+"-"*(w-int(round(min(1,v/max(t,1))*w)))


def main():
    arrays, names = [], []
    if NPZ_PATH.exists():
        data = np.load(NPZ_PATH, allow_pickle=True)
        arrays = list(data["arrays"]); names = list(data["file_names"])
        print(f"Resumed: {len(arrays):,} arrays")

    print(f"Crypto: {len(arrays):,}/{TARGET:,}  (fetching...)")
    if len(arrays) >= TARGET: print("Already at target."); return

    started, outer_seed = time.time(), 0
    last_print = 0
    for ticker in TICKERS:
        if len(arrays) >= TARGET: break
        for period, interval in PERIODS:
            if len(arrays) >= TARGET: break
            try:
                df = yf.download(ticker, period=period, interval=interval, progress=False, timeout=15)
            except Exception: time.sleep(2); continue
            if df is None or df.empty: continue
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            for col in COLS:
                if col not in df.columns: continue
                arr = clean(df[col].dropna().values.astype(np.float64))
                if arr is None: continue
                outer_seed += 1
                for t_arr in transforms(arr, outer_seed):
                    arrays.append(t_arr); names.append(f"crypto_{ticker}_{period}_{interval}_{col}_{outer_seed}")
                    if len(arrays) >= TARGET: break
                for w_arr in windows(arr, outer_seed+10000):
                    arrays.append(w_arr); names.append(f"crypto_{ticker}_{period}_{interval}_{col}_w{len(w_arr)}_{outer_seed}")
                    if len(arrays) >= TARGET: break
            if len(arrays) - last_print >= 100:
                e = (time.time()-started)/60
                print(f"  [{bar(len(arrays),TARGET)}] {len(arrays):,}/{TARGET:,} ({100*len(arrays)/TARGET:.1f}%) | {e:.1f} min")
                last_print = len(arrays)
            time.sleep(0.15)

    arr_obj = np.empty(len(arrays), dtype=object)
    for i,a in enumerate(arrays): arr_obj[i]=a
    print(f"Saving {NPZ_PATH} ...")
    np.savez_compressed(NPZ_PATH, arrays=arr_obj, file_names=np.array(names))
    print(f"Done. {len(arrays):,} arrays → {NPZ_PATH.stat().st_size/(1024**2):.0f} MB in {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--target", type=int, default=TARGET)
    args = p.parse_args(); TARGET = args.target; main()
