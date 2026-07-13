#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/Volumes/k/thesis_data/f1_only_1m_packed")
INPUT_CSV = ROOT / "training_dataset.csv"
RAW_H5 = ROOT / "raw_arrays.h5"
OUTPUT_CSV = ROOT / "extra6_all9_retimed_labels.csv"
REPORT_MD = Path("reports/extra6_all9_retimed_labels_report.md")

V5 = ["timsort", "introsort", "heapsort"]
EXTRA6 = ["quick_sort", "merge_sort", "shell_sort", "counting_sort", "insertion_sort", "comb_sort"]
ALL9 = V5 + EXTRA6
TIME_COLS = {a: f"time_{a}" for a in ALL9}

C = {
    "r": "\033[31m",
    "g": "\033[32m",
    "y": "\033[33m",
    "b": "\033[34m",
    "m": "\033[35m",
    "c": "\033[36m",
    "x": "\033[0m",
    "bold": "\033[1m",
}

_H5 = None
_IX = None
_REPEATS = 1


def worker_init(raw_h5, repeats):
    global _H5, _IX, _REPEATS
    _H5 = h5py.File(raw_h5, "r")
    _IX = h5_index(_H5)
    _REPEATS = repeats


def quick_sort(data):
    arr = data[:]
    if len(arr) < 2:
        return arr
    stack = [(0, len(arr) - 1)]
    while stack:
        low, high = stack.pop()
        if low >= high:
            continue
        mid = (low + high) // 2
        trio = [(arr[low], low), (arr[mid], mid), (arr[high], high)]
        trio.sort(key=lambda x: x[0])
        pivot_i = trio[1][1]
        arr[pivot_i], arr[high] = arr[high], arr[pivot_i]
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        i += 1
        arr[i], arr[high] = arr[high], arr[i]
        if i - low > high - i:
            stack.extend([(low, i - 1), (i + 1, high)])
        else:
            stack.extend([(i + 1, high), (low, i - 1)])
    return arr


def merge_sort(data):
    n = len(data)
    if n < 2:
        return data[:]
    arr = data[:]
    buf = [0.0] * n
    width = 1
    while width < n:
        for left in range(0, n, 2 * width):
            mid = min(left + width, n)
            right = min(left + 2 * width, n)
            i, j, k = left, mid, left
            while i < mid and j < right:
                if arr[i] <= arr[j]:
                    buf[k] = arr[i]
                    i += 1
                else:
                    buf[k] = arr[j]
                    j += 1
                k += 1
            while i < mid:
                buf[k] = arr[i]
                i += 1
                k += 1
            while j < right:
                buf[k] = arr[j]
                j += 1
                k += 1
            arr[left:right] = buf[left:right]
        width *= 2
    return arr


def shell_sort(data):
    arr = data[:]
    gap = len(arr) // 2
    while gap > 0:
        for i in range(gap, len(arr)):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr


def counting_sort(data):
    if len(data) < 2:
        return data[:]
    out = []
    counts = Counter(data)
    for value in sorted(counts):
        out.extend([value] * counts[value])
    return out


def insertion_sort(data):
    arr = data[:]
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def comb_sort(data):
    arr = data[:]
    gap = len(arr)
    shrink = 1.3
    ok = False
    while not ok:
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            ok = True
        i = 0
        while i + gap < len(arr):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                ok = False
            i += 1
    return arr


FUNCS = {
    "quick_sort": quick_sort,
    "merge_sort": merge_sort,
    "shell_sort": shell_sort,
    "counting_sort": counting_sort,
    "insertion_sort": insertion_sort,
    "comb_sort": comb_sort,
}


def best_name(times, names):
    good = {n: times[n] for n in names if np.isfinite(times[n])}
    return min(good, key=good.get) if good else ""


def time_numpy(arr, kind, repeats):
    values = []
    for _ in range(repeats):
        data = arr.copy()
        gc.disable()
        t0 = time.perf_counter()
        np.sort(data, kind=kind)
        t1 = time.perf_counter()
        gc.enable()
        values.append(t1 - t0)
    return float(np.median(np.array(values, dtype=np.float64)))


def time_algo(arr, name, repeats):
    values = []
    data0 = arr.tolist()
    for _ in range(repeats):
        data = data0[:]
        gc.disable()
        t0 = time.perf_counter()
        FUNCS[name](data)
        t1 = time.perf_counter()
        gc.enable()
        values.append(t1 - t0)
    return float(np.median(np.array(values, dtype=np.float64)))


def h5_index(h5):
    files = h5["file"][:]
    return {(x.decode("utf-8") if isinstance(x, bytes) else str(x)): i for i, x in enumerate(files)}


def read_array(h5, ix, name):
    r = ix[name]
    start = int(h5["start"][r])
    length = int(h5["length"][r])
    return h5["values"][start:start + length]


def process_row(row):
    name = row["file"]
    arr = read_array(_H5, _IX, name)
    times = {
        "timsort": time_numpy(arr, "stable", _REPEATS),
        "introsort": time_numpy(arr, "quicksort", _REPEATS),
        "heapsort": time_numpy(arr, "heapsort", _REPEATS),
    }
    for algo in EXTRA6:
        times[algo] = time_algo(arr, algo, _REPEATS)
    best_v5 = best_name(times, V5)
    best_extra6 = best_name(times, EXTRA6)
    best_all9 = best_name(times, ALL9)
    winner_group = "final3" if best_all9 in V5 else "extra6"
    out = {"file": name, "domain": row.get("domain", ""), "n_elements": row.get("n_elements", "")}
    out.update({TIME_COLS[a]: times[a] for a in ALL9})
    out.update({
        "best_v5": best_v5,
        "best_extra6": best_extra6,
        "best_all9": best_all9,
        "winner_group": winner_group,
        "v5_best_time": times[best_v5],
        "extra6_best_time": times[best_extra6],
        "all9_best_time": times[best_all9],
    })
    return out


def existing_done(path):
    if not path.exists():
        return set()
    with path.open("r", newline="") as f:
        return {row["file"] for row in csv.DictReader(f)}


def write_header(path):
    cols = ["file", "domain", "n_elements"] + [TIME_COLS[a] for a in ALL9] + [
        "best_v5", "best_extra6", "best_all9", "winner_group",
        "v5_best_time", "extra6_best_time", "all9_best_time"
    ]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()
    return cols


def bar(pct, width=30):
    n = int(round(width * pct / 100))
    return "█" * n + "░" * (width - n)


def print_live(done, total, counters, last):
    pct = 100 * done / total if total else 0
    print(f"\n{C['bold']}{C['c']}processed {done:,}/{total:,}  {pct:.2f}%{C['x']}")
    print(f"last file: {last.get('file','')}")
    print(f"n={last.get('n_elements','')}  group={last.get('winner_group','')}  v5={last.get('best_v5','')}  extra6={last.get('best_extra6','')}  all9={last.get('best_all9','')}")
    for title, counter in counters.items():
        print(f"{C['y']}{title}{C['x']}")
        s = sum(counter.values())
        for a, n in counter.most_common():
            p = 100 * n / s if s else 0
            print(f"  {a:16s} {n:8,} {p:6.2f}% {bar(p)}")


def build_report(output_csv, report_md):
    df = pd.read_csv(output_csv)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Extra 6 vs Final 3 Sorting Benchmark", ""]
    lines += [f"- source file list: `{INPUT_CSV}`", f"- source raw arrays: `{RAW_H5}`", f"- result CSV: `{output_csv}`", f"- rows: `{len(df):,}`", ""]
    lines += ["## Main Question: Final 3 vs Extra 6", ""]
    vc = df["winner_group"].value_counts(dropna=False)
    for k, v in vc.items():
        lines.append(f"- `{k}`: `{v:,}` ({100*v/len(df):.2f}%)")
    lines.append("")
    for col, title in [("best_v5", "Final v5 trio winners"), ("best_extra6", "Extra 6 winners"), ("best_all9", "All 9 winners")]:
        lines += [f"## {title}", ""]
        vc = df[col].value_counts(dropna=False)
        for k, v in vc.items():
            lines.append(f"- `{k}`: `{v:,}` ({100*v/len(df):.2f}%)")
        lines.append("")
    lines += ["## Timing Summary", "", "| algorithm | mean us | median us | p95 us |", "|---|---:|---:|---:|"]
    for a in ALL9:
        col = TIME_COLS[a]
        s = df[col].dropna() * 1e6
        lines.append(f"| `{a}` | {s.mean():.3f} | {s.median():.3f} | {s.quantile(0.95):.3f} |")
    extra_beats = (df["winner_group"] == "extra6").sum()
    final3_beats = (df["winner_group"] == "final3").sum()
    lines += ["", "## Thesis Use", "", f"- final 3 win: `{final3_beats:,}` / `{len(df):,}` ({100*final3_beats/len(df):.2f}%)", f"- extra 6 win: `{extra_beats:,}` / `{len(df):,}` ({100*extra_beats/len(df):.2f}%)", "- This is direct answer: are timsort/introsort/heapsort enough, or did excluded algorithms win meaningful share?", ""]
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=INPUT_CSV)
    ap.add_argument("--raw-h5", type=Path, default=RAW_H5)
    ap.add_argument("--output", type=Path, default=OUTPUT_CSV)
    ap.add_argument("--report", type=Path, default=REPORT_MD)
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()
    print(f"{C['bold']}extra6 all9 full retiming benchmark{C['x']}")
    print(f"input:  {args.input}")
    print(f"raw:    {args.raw_h5}")
    print(f"output: {args.output}")
    print(f"report: {args.report}")
    print(f"workers: {args.workers}")
    cols = write_header(args.output)
    done_files = existing_done(args.output)
    total = sum(1 for _ in open(args.input, "rb")) - 1
    if args.limit:
        total = min(total, args.limit)
    counters = {"winner_group": Counter(), "best_v5": Counter(), "best_extra6": Counter(), "best_all9": Counter()}
    processed = len(done_files)
    last = {}
    def save_done(done):
        nonlocal processed, last
        for fut in done:
            out = fut.result()
            writer.writerow(out)
            f.flush()
            done_files.add(out["file"])
            processed += 1
            last = out
            counters["winner_group"][out["winner_group"]] += 1
            counters["best_v5"][out["best_v5"]] += 1
            counters["best_extra6"][out["best_extra6"]] += 1
            counters["best_all9"][out["best_all9"]] += 1
            if processed % args.progress_every == 0:
                print_live(processed, total, counters, last)

    with args.output.open("a", newline="") as f, ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(str(args.raw_h5), args.repeats),
    ) as pool:
        writer = csv.DictWriter(f, fieldnames=cols)
        seen = 0
        in_flight = set()
        for chunk in pd.read_csv(args.input, chunksize=args.chunk_size):
            for row in chunk[["file", "domain", "n_elements"]].to_dict("records"):
                seen += 1
                if args.limit and seen > args.limit:
                    while in_flight:
                        done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                        save_done(done)
                    build_report(args.output, args.report)
                    print(f"\nreport saved: {args.report}")
                    return
                if row["file"] in done_files:
                    continue
                in_flight.add(pool.submit(process_row, row))
                if len(in_flight) >= args.batch_size:
                    done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                    save_done(done)
        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            save_done(done)
    build_report(args.output, args.report)
    print_live(processed, total, counters, last)
    print(f"\n{C['g']}done{C['x']} report saved: {args.report}")


if __name__ == "__main__":
    main()
