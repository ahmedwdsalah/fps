#!/usr/bin/env python3
"""
Separate F1 rows from a combined dataset into two independent files.

Primary path (fast): DuckDB multi-threaded split (CSV/Parquet).
Fallback path: chunked pandas CSV split if DuckDB is unavailable.

Default behavior:
  - Input: /Users/ahmed/Desktop/thesis/My-Master-thesis/data/training_dataset.csv
  - Output files: created in the SAME directory as the input file.
  - Input file is replaced in-place with non-F1 rows.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_INPUT = Path("/Users/ahmed/Desktop/thesis/My-Master-thesis/data/training_dataset.csv")
DEFAULT_THREADS = 12
DEFAULT_MEMORY_LIMIT = "16GB"
DEFAULT_IN_PLACE_NON_F1 = True
CHUNK_SIZE = 250_000


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _detect_columns_csv(path: Path) -> list[str]:
    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if not header:
        raise RuntimeError(f"No header found in CSV: {path}")
    return header


def _pick_column(columns: Iterable[str], target: str) -> str | None:
    mapping = {c.strip().lower(): c for c in columns}
    return mapping.get(target.lower())


def _build_is_f1_sql(columns: list[str]) -> str:
    domain_col = _pick_column(columns, "domain")
    file_col = _pick_column(columns, "file")
    if domain_col:
        dom = _qident(domain_col)
        return f"COALESCE(lower(trim(CAST({dom} AS VARCHAR))) = 'f1', FALSE)"
    if file_col:
        fil = _qident(file_col)
        return f"COALESCE(starts_with(lower(CAST({fil} AS VARCHAR)), 'f1_'), FALSE)"
    raise RuntimeError(
        "Cannot detect F1 rows: missing both 'domain' and 'file' columns."
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _out_paths(input_path: Path, output_dir: Path | None) -> tuple[Path, Path]:
    parent = output_dir if output_dir else input_path.parent
    stem = input_path.stem
    suffix = input_path.suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise RuntimeError(f"Unsupported file extension: {suffix}")
    return (
        parent / f"{stem}_f1_only{suffix}",
        parent / f"{stem}_non_f1{suffix}",
    )


def split_with_duckdb(
    input_path: Path, f1_out: Path, non_f1_out: Path, threads: int, memory_limit: str
) -> dict:
    import duckdb

    suffix = input_path.suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise RuntimeError("DuckDB mode supports only CSV or Parquet input.")

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={max(1, threads)};")
    if memory_limit:
        con.execute(f"PRAGMA memory_limit='{memory_limit}';")

    if suffix == ".csv":
        columns = _detect_columns_csv(input_path)
        source = f"read_csv_auto('{input_path.as_posix()}', HEADER=TRUE, SAMPLE_SIZE=-1)"
    else:
        # For Parquet, ask DuckDB for schema.
        source = f"read_parquet('{input_path.as_posix()}')"
        desc = con.execute(f"DESCRIBE SELECT * FROM {source} LIMIT 0").fetchall()
        columns = [r[0] for r in desc]

    is_f1_sql = _build_is_f1_sql(columns)

    stats = con.execute(
        f"""
        SELECT
            COUNT(*)::BIGINT AS total_rows,
            SUM(CASE WHEN {is_f1_sql} THEN 1 ELSE 0 END)::BIGINT AS f1_rows
        FROM {source}
        """
    ).fetchone()
    total_rows = int(stats[0] or 0)
    f1_rows = int(stats[1] or 0)
    non_f1_rows = total_rows - f1_rows

    _ensure_parent(f1_out)
    _ensure_parent(non_f1_out)
    f1_tmp = f1_out.with_suffix(f1_out.suffix + ".tmp")
    non_tmp = non_f1_out.with_suffix(non_f1_out.suffix + ".tmp")

    if suffix == ".csv":
        con.execute(
            f"""
            COPY (
                SELECT * FROM {source}
                WHERE {is_f1_sql}
            ) TO '{f1_tmp.as_posix()}'
            (FORMAT CSV, HEADER TRUE, DELIMITER ',')
            """
        )
        con.execute(
            f"""
            COPY (
                SELECT * FROM {source}
                WHERE NOT {is_f1_sql}
            ) TO '{non_tmp.as_posix()}'
            (FORMAT CSV, HEADER TRUE, DELIMITER ',')
            """
        )
    else:
        con.execute(
            f"""
            COPY (
                SELECT * FROM {source}
                WHERE {is_f1_sql}
            ) TO '{f1_tmp.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        con.execute(
            f"""
            COPY (
                SELECT * FROM {source}
                WHERE NOT {is_f1_sql}
            ) TO '{non_tmp.as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

    # Validate output row counts before final rename.
    if suffix == ".csv":
        out_f1 = con.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{f1_tmp.as_posix()}', HEADER=TRUE, SAMPLE_SIZE=-1)"
        ).fetchone()[0]
        out_non = con.execute(
            f"SELECT COUNT(*) FROM read_csv_auto('{non_tmp.as_posix()}', HEADER=TRUE, SAMPLE_SIZE=-1)"
        ).fetchone()[0]
    else:
        out_f1 = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{f1_tmp.as_posix()}')"
        ).fetchone()[0]
        out_non = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{non_tmp.as_posix()}')"
        ).fetchone()[0]

    out_f1 = int(out_f1 or 0)
    out_non = int(out_non or 0)
    if out_f1 != f1_rows or out_non != non_f1_rows or (out_f1 + out_non) != total_rows:
        raise RuntimeError(
            "Validation failed: output row counts do not match input."
        )

    os.replace(f1_tmp, f1_out)
    os.replace(non_tmp, non_f1_out)

    return {
        "engine": "duckdb",
        "total_rows": total_rows,
        "f1_rows": f1_rows,
        "non_f1_rows": non_f1_rows,
    }


def split_csv_with_pandas(input_path: Path, f1_out: Path, non_f1_out: Path) -> dict:
    columns = _detect_columns_csv(input_path)
    domain_col = _pick_column(columns, "domain")
    file_col = _pick_column(columns, "file")
    if not domain_col and not file_col:
        raise RuntimeError("CSV fallback needs 'domain' or 'file' column.")

    _ensure_parent(f1_out)
    _ensure_parent(non_f1_out)
    f1_tmp = f1_out.with_suffix(f1_out.suffix + ".tmp")
    non_tmp = non_f1_out.with_suffix(non_f1_out.suffix + ".tmp")
    for p in (f1_tmp, non_tmp):
        if p.exists():
            p.unlink()

    total_rows = 0
    f1_rows = 0
    non_rows = 0
    first_f1 = True
    first_non = True

    for chunk in pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False):
        if domain_col and domain_col in chunk.columns:
            is_f1 = (
                chunk[domain_col]
                .astype("string")
                .str.strip()
                .str.lower()
                .eq("f1")
                .fillna(False)
            )
        else:
            is_f1 = (
                chunk[file_col]
                .astype("string")
                .str.lower()
                .str.startswith("f1_")
                .fillna(False)
            )

        f1_chunk = chunk[is_f1]
        non_chunk = chunk[~is_f1]

        f1_chunk.to_csv(f1_tmp, mode="a", header=first_f1, index=False)
        non_chunk.to_csv(non_tmp, mode="a", header=first_non, index=False)

        first_f1 = False
        first_non = False

        n_chunk = len(chunk)
        n_f1 = len(f1_chunk)
        total_rows += n_chunk
        f1_rows += n_f1
        non_rows += n_chunk - n_f1

    # Validate by counting lines (minus header).
    def _csv_rows(path: Path) -> int:
        with path.open("r", newline="") as f:
            return max(sum(1 for _ in f) - 1, 0)

    out_f1 = _csv_rows(f1_tmp)
    out_non = _csv_rows(non_tmp)
    if out_f1 != f1_rows or out_non != non_rows or (out_f1 + out_non) != total_rows:
        raise RuntimeError("Validation failed in pandas fallback.")

    os.replace(f1_tmp, f1_out)
    os.replace(non_tmp, non_f1_out)
    return {
        "engine": "pandas",
        "total_rows": total_rows,
        "f1_rows": f1_rows,
        "non_f1_rows": non_rows,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split combined training data into F1-only and non-F1 files."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV/Parquet path (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same directory as input)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Threads for DuckDB (default: {DEFAULT_THREADS})",
    )
    p.add_argument(
        "--memory-limit",
        type=str,
        default=DEFAULT_MEMORY_LIMIT,
        help=f"DuckDB memory limit, e.g. 8GB, 16GB (default: {DEFAULT_MEMORY_LIMIT})",
    )
    p.add_argument(
        "--force-pandas",
        action="store_true",
        help="Force chunked pandas CSV mode (debug/fallback).",
    )
    p.add_argument(
        "--in-place-non-f1",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_IN_PLACE_NON_F1,
        help="Write F1 to *_f1_only.* and replace the input file with non-F1 rows.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}", file=sys.stderr)
        return 2

    f1_out, non_f1_out = _out_paths(input_path, args.output_dir)
    replace_input = bool(args.in_place_non_f1)
    if replace_input:
        # Write non-F1 to a temporary sibling path first, then atomically replace input.
        non_f1_out = input_path.with_name(f".{input_path.stem}.non_f1_tmp{input_path.suffix}")
    print("=" * 72)
    print("F1 SPLITTER")
    print("=" * 72)
    print(f"Input:       {input_path}")
    print(f"F1 output:   {f1_out}")
    if replace_input:
        print(f"Non-F1 out:  {input_path} (in-place replacement)")
    else:
        print(f"Non-F1 out:  {non_f1_out}")
    print(f"Threads:     {args.threads}")
    print(f"Memory:      {args.memory_limit}")
    print("=" * 72)

    t0 = time.perf_counter()
    try:
        if args.force_pandas:
            if input_path.suffix.lower() != ".csv":
                raise RuntimeError("--force-pandas supports CSV input only.")
            stats = split_csv_with_pandas(input_path, f1_out, non_f1_out)
        else:
            try:
                stats = split_with_duckdb(
                    input_path=input_path,
                    f1_out=f1_out,
                    non_f1_out=non_f1_out,
                    threads=args.threads,
                    memory_limit=args.memory_limit,
                )
            except Exception as duck_err:
                if input_path.suffix.lower() != ".csv":
                    raise
                print(f"WARN: DuckDB path failed ({duck_err}). Falling back to pandas CSV mode.")
                stats = split_csv_with_pandas(input_path, f1_out, non_f1_out)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    dt = time.perf_counter() - t0
    rows = stats["total_rows"]
    rate = rows / dt if dt > 0 else 0.0

    if replace_input:
        os.replace(non_f1_out, input_path)

    print("\nSplit complete.")
    print(f"Engine:      {stats['engine']}")
    print(f"Total rows:  {rows:,}")
    print(f"F1 rows:     {stats['f1_rows']:,}")
    print(f"Non-F1 rows: {stats['non_f1_rows']:,}")
    if replace_input:
        print(f"Input file replaced with non-F1 rows: {input_path}")
    print(f"Time:        {dt:.2f}s")
    print(f"Throughput:  {rate:,.0f} rows/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
