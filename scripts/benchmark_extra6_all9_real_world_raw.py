#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from benchmark_extra6_all9_labels import (
    ALL9,
    EXTRA6,
    TIME_COLS,
    V5,
    C,
    bar,
    best_name,
    time_numpy,
)

ROOT = Path("/Volumes/k/thesis_data/real_world_10k")
INDEX_CSV = ROOT / "index.csv"
RAW_DIR = ROOT / "raw"
OUTPUT_CSV = ROOT / "extra6_all9_retimed_labels.csv"
REPORT_MD = Path("reports/extra6_all9_real_world_raw_report.md")

_RAW_DIR = None
_REPEATS = 1


@njit(cache=True)
def quick_sort_jit(a):
    arr = a.copy()
    stack_l = np.empty(arr.size, dtype=np.int64)
    stack_h = np.empty(arr.size, dtype=np.int64)
    top = 0
    stack_l[top] = 0
    stack_h[top] = arr.size - 1
    while top >= 0:
        low = stack_l[top]
        high = stack_h[top]
        top -= 1
        if low >= high:
            continue
        mid = (low + high) // 2
        if arr[low] > arr[mid]:
            arr[low], arr[mid] = arr[mid], arr[low]
        if arr[mid] > arr[high]:
            arr[mid], arr[high] = arr[high], arr[mid]
        if arr[low] > arr[mid]:
            arr[low], arr[mid] = arr[mid], arr[low]
        arr[mid], arr[high] = arr[high], arr[mid]
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        i += 1
        arr[i], arr[high] = arr[high], arr[i]
        if low < i - 1:
            top += 1
            stack_l[top] = low
            stack_h[top] = i - 1
        if i + 1 < high:
            top += 1
            stack_l[top] = i + 1
            stack_h[top] = high
    return arr


@njit(cache=True)
def merge_sort_jit(a):
    n = a.size
    arr = a.copy()
    buf = np.empty(n, dtype=np.float64)
    width = 1
    while width < n:
        left = 0
        while left < n:
            mid = min(left + width, n)
            right = min(left + 2 * width, n)
            i = left
            j = mid
            k = left
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
            for k2 in range(left, right):
                arr[k2] = buf[k2]
            left += 2 * width
        width *= 2
    return arr


@njit(cache=True)
def shell_sort_jit(a):
    arr = a.copy()
    gap = arr.size // 2
    while gap > 0:
        for i in range(gap, arr.size):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2
    return arr


@njit(cache=True)
def insertion_sort_jit(a):
    arr = a.copy()
    for i in range(1, arr.size):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


@njit(cache=True)
def comb_sort_jit(a):
    arr = a.copy()
    gap = arr.size
    ok = False
    while not ok:
        gap = int(gap / 1.3)
        if gap <= 1:
            gap = 1
            ok = True
        i = 0
        while i + gap < arr.size:
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                ok = False
            i += 1
    return arr


def counting_sort_fast(a):
    values, counts = np.unique(a, return_counts=True)
    return np.repeat(values, counts)


JIT_FUNCS = {
    "quick_sort": quick_sort_jit,
    "merge_sort": merge_sort_jit,
    "shell_sort": shell_sort_jit,
    "counting_sort": counting_sort_fast,
    "insertion_sort": insertion_sort_jit,
    "comb_sort": comb_sort_jit,
}


def time_extra_fast(arr, name, repeats):
    values = []
    fn = JIT_FUNCS[name]
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(arr)
        t1 = time.perf_counter()
        values.append(t1 - t0)
    return float(np.median(np.array(values, dtype=np.float64)))


def warmup_jit():
    a = np.array([3.0, 1.0, 2.0, 1.0, -1.0], dtype=np.float64)
    for name in EXTRA6:
        JIT_FUNCS[name](a)


def worker_init(raw_dir, repeats):
    global _RAW_DIR, _REPEATS
    _RAW_DIR = Path(raw_dir)
    _REPEATS = repeats
    warmup_jit()


def read_array(name):
    return np.loadtxt(_RAW_DIR / name, dtype=np.float64, delimiter=",")


def process_row(row):
    name = row["file"]
    arr = np.atleast_1d(read_array(name))
    times = {
        "timsort": time_numpy(arr, "stable", _REPEATS),
        "introsort": time_numpy(arr, "quicksort", _REPEATS),
        "heapsort": time_numpy(arr, "heapsort", _REPEATS),
    }
    for algo in EXTRA6:
        times[algo] = time_extra_fast(arr, algo, _REPEATS)
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
    lines = ["# Extra 6 vs Final 3 Sorting Benchmark - Real-World Raw", ""]
    lines += [f"- source index: `{INDEX_CSV}`", f"- source raw dir: `{RAW_DIR}`", f"- result CSV: `{output_csv}`", f"- rows: `{len(df):,}`", ""]
    lines += ["## Main Question: Final 3 vs Extra 6", ""]
    for k, v in df["winner_group"].value_counts(dropna=False).items():
        lines.append(f"- `{k}`: `{v:,}` ({100*v/len(df):.2f}%)")
    lines.append("")
    for col, title in [("best_v5", "Final v5 trio winners"), ("best_extra6", "Extra 6 winners"), ("best_all9", "All 9 winners")]:
        lines += [f"## {title}", ""]
        for k, v in df[col].value_counts(dropna=False).items():
            lines.append(f"- `{k}`: `{v:,}` ({100*v/len(df):.2f}%)")
        lines.append("")
    lines += ["## Timing Summary", "", "| algorithm | mean us | median us | p95 us |", "|---|---:|---:|---:|"]
    for a in ALL9:
        s = df[TIME_COLS[a]].dropna() * 1e6
        lines.append(f"| `{a}` | {s.mean():.3f} | {s.median():.3f} | {s.quantile(0.95):.3f} |")
    final3 = (df["winner_group"] == "final3").sum()
    extra6 = (df["winner_group"] == "extra6").sum()
    lines += ["", "## Thesis Use", "", f"- final 3 win: `{final3:,}` / `{len(df):,}` ({100*final3/len(df):.2f}%)", f"- extra 6 win: `{extra6:,}` / `{len(df):,}` ({100*extra6/len(df):.2f}%)", ""]
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=INDEX_CSV)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--output", type=Path, default=OUTPUT_CSV)
    ap.add_argument("--report", type=Path, default=REPORT_MD)
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()
    print(f"{C['bold']}real-world raw all9 retiming benchmark{C['x']}")
    print(f"index:   {args.index}")
    print(f"raw:     {args.raw_dir}")
    print(f"output:  {args.output}")
    print(f"report:  {args.report}")
    print(f"workers: {args.workers}")
    cols = write_header(args.output)
    done_files = existing_done(args.output)
    total = sum(1 for _ in open(args.index, "rb")) - 1
    if args.limit:
        total = min(total, args.limit)
    counters = {"winner_group": Counter(), "best_v5": Counter(), "best_extra6": Counter(), "best_all9": Counter()}
    processed = len(done_files)
    last = {}
    with args.output.open("a", newline="") as f, ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(str(args.raw_dir), args.repeats),
    ) as pool:
        writer = csv.DictWriter(f, fieldnames=cols)
        in_flight = set()

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

        seen = 0
        for chunk in pd.read_csv(args.index, chunksize=args.chunk_size):
            for row in chunk[["file", "channel", "n_elements"]].to_dict("records"):
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
                row["domain"] = row.pop("channel")
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
