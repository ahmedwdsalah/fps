#!/usr/bin/env python3
"""
Build per-channel balanced dataset for v2 algorithm labels.

Input must contain:
- file
- best_algorithm_v2
- features/timing columns (kept as-is)

If channel column missing, joins from index.csv via file.
Balance rule:
- per channel, keep classes with count >= min_class_count
- downsample each kept class to channel-local minimum
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("/Volumes/k/thesis_data/f1_only/training_dataset_algos_v2.csv")
DEFAULT_INDEX = Path("/Volumes/k/thesis_data/f1_only_1m_packed/index.csv")
DEFAULT_OUTPUT = Path("/Volumes/k/thesis_data/f1_only_1m_packed/training_dataset_algos_v2_channel_balanced.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-channel balanced v2 dataset.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--min-class-count", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="Pilot mode: use first N rows from input.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing input: {args.input}")
    if not args.index.exists():
        raise SystemExit(f"Missing index: {args.index}")

    print("=" * 80)
    print("BUILD F1 V2 CHANNEL-BALANCED DATASET")
    print("=" * 80)
    print(f"Input:  {args.input}")
    print(f"Index:  {args.index}")
    print(f"Output: {args.output}")
    print(f"Min class count: {args.min_class_count}")
    if args.limit > 0:
        print(f"Limit: {args.limit}")

    df = pd.read_csv(args.input)
    if args.limit > 0:
        df = df.head(args.limit).copy()
    if "best_algorithm_v2" not in df.columns:
        raise SystemExit("Missing best_algorithm_v2 in input.")

    if "channel" not in df.columns:
        idx = pd.read_csv(args.index, usecols=["file", "channel"])
        df = df.merge(idx, on="file", how="left", validate="many_to_one")
        miss = int(df["channel"].isna().sum())
        if miss > 0:
            # Auto-fallback: try sibling index next to input file root.
            sibling_index = args.input.parent / "index.csv"
            if sibling_index != args.index and sibling_index.exists():
                print(
                    f"\nPrimary index mismatch ({miss:,} missing). "
                    f"Retry with sibling index: {sibling_index}"
                )
                base = pd.read_csv(args.input)
                if args.limit > 0:
                    base = base.head(args.limit).copy()
                idx2 = pd.read_csv(sibling_index, usecols=["file", "channel"])
                df = base.merge(idx2, on="file", how="left", validate="many_to_one")
                miss = int(df["channel"].isna().sum())
                if miss == 0:
                    print("Sibling index join succeeded.")
                else:
                    raise SystemExit(f"Missing channel after fallback join for {miss:,} rows.")
            else:
                raise SystemExit(f"Missing channel after join for {miss:,} rows.")

    print(f"\nRows in working input: {len(df):,}")
    print("Global class counts (before):")
    print(df["best_algorithm_v2"].value_counts().to_string())

    parts = []
    channels = sorted(df["channel"].unique().tolist())
    print("\nPer-channel balancing:")
    for ch in channels:
        sub = df[df["channel"] == ch].copy()
        counts = sub["best_algorithm_v2"].value_counts()
        keep = counts[counts >= args.min_class_count].index.tolist()
        sub = sub[sub["best_algorithm_v2"].isin(keep)].copy()
        if sub.empty or sub["best_algorithm_v2"].nunique() < 2:
            print(f"  [{ch}] skipped (supported classes < 2)")
            continue
        min_n = int(sub["best_algorithm_v2"].value_counts().min())
        bal = (
            sub.groupby("best_algorithm_v2", group_keys=False)
            .apply(lambda g: g.sample(n=min_n, random_state=args.seed))
            .reset_index(drop=True)
        )
        parts.append(bal)
        kept = sorted(bal["best_algorithm_v2"].unique().tolist())
        print(f"  [{ch}] kept_classes={kept} min_n={min_n} rows={len(bal):,}")

    if not parts:
        raise SystemExit("No channel produced a usable balanced subset.")

    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output rows: {len(out):,}")
    print(f"Saved: {args.output}")
    print("\nGlobal class counts (after):")
    print(out["best_algorithm_v2"].value_counts().to_string())
    print("\nChannel counts (after):")
    print(out["channel"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
