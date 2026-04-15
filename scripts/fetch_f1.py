#!/usr/bin/env python3
"""
F1-only dataset balancer.

Behavior:
1) Load /Users/ahmed/Desktop/thesis/My-Master-thesis/data/training_dataset_f1_only.csv
2) Show F1 domain distribution (line by line with count and %)
3) Keep only Distance rows, then class-balance by best_algorithm, write back in-place
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

F1_ONLY_CSV = Path("/Volumes/k/thesis_data/real_world_10k/index_f1_only.csv")
BALANCE_SEED = 42
BALANCE_MEMORY_LIMIT = "16GB"
TARGET_ROWS_PER_CHANNEL = 100_000
CHANNEL_REGEX = (
    r"(?:^|[_\s(])"
    r"(distance|speed|throttle|rpm|ngear|gear|drs|status|time|sessiontime|relativedistance|distancetodriverahead|driverahead)"
    r"(?:[_\s)]|$)"
)


def _print_domain_stats(counts: pd.Series) -> None:
    total = int(counts.sum())
    print("[2/3] Domain distribution")
    for domain, count in counts.items():
        pct = (count * 100.0 / total) if total else 0.0
        print(f"  - {domain}: {count:,} ({pct:.2f}%)")


def _build_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "domain" in df.columns:
        f1_mask = (
            df["domain"].astype("string").str.strip().str.lower()
            .isin(["f1", "f1_telemetry", "formula1", "formula_1"])
            .fillna(False)
        )
    else:
        f1_mask = pd.Series(True, index=df.index)

    if "channel" in df.columns or "file" in df.columns or "array" in df.columns:
        channel_keep_mask = pd.Series(True, index=df.index)
    else:
        raise RuntimeError("Missing one of columns: channel/file/array.")

    return f1_mask, channel_keep_mask


def _channel_series(df: pd.DataFrame) -> pd.Series:
    if "channel" in df.columns:
        return (
            df["channel"].astype("string").str.strip().str.lower().fillna("")
        )
    if "file" in df.columns:
        return (
            df["file"].astype("string").str.strip().str.lower()
            .str.extract(CHANNEL_REGEX, expand=False)
            .fillna("")
        )
    if "array" in df.columns:
        return (
            df["array"].astype("string").str.strip().str.lower()
            .str.extract(CHANNEL_REGEX, expand=False)
            .fillna("")
        )
    return pd.Series([""] * len(df), index=df.index, dtype="string")


def _prepare_with_duckdb(path: Path) -> None:
    import duckdb

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={max(1, os.cpu_count() or 1)};")
    con.execute(f"PRAGMA memory_limit='{BALANCE_MEMORY_LIMIT}';")

    source = f"read_csv_auto('{path.as_posix()}', HEADER=TRUE, SAMPLE_SIZE=-1)"
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source} LIMIT 0").fetchall()]
    lower = {c.strip().lower(): c for c in cols}

    if "best_algorithm" not in lower:
        raise RuntimeError("Missing required 'best_algorithm' column in training dataset.")

    domain_col = lower.get("domain")
    channel_col = lower.get("channel")
    file_col = lower.get("file")
    array_col = lower.get("array")
    best_col = lower["best_algorithm"]

    q_domain = f'"{domain_col}"' if domain_col else None
    q_channel = f'"{channel_col}"' if channel_col else None
    q_file = f'"{file_col}"' if file_col else None
    q_array = f'"{array_col}"' if array_col else None
    q_best = f'"{best_col}"'

    f1_filter = (
        f"lower(trim(CAST({q_domain} AS VARCHAR))) IN ('f1', 'f1_telemetry', 'formula1', 'formula_1')"
        if q_domain
        else "TRUE"
    )

    if q_channel:
        channel_key_expr = f"lower(trim(CAST({q_channel} AS VARCHAR)))"
    elif q_file:
        channel_key_expr = f"regexp_extract(lower(CAST({q_file} AS VARCHAR)), '{CHANNEL_REGEX}', 1)"
    else:
        channel_key_expr = f"regexp_extract(lower(CAST({q_array} AS VARCHAR)), '{CHANNEL_REGEX}', 1)"

    where_sql = f"({f1_filter}) AND (coalesce({channel_key_expr}, '') <> '')"

    rows = int(con.execute(f"SELECT COUNT(*) FROM {source} WHERE {where_sql}").fetchone()[0] or 0)
    if rows == 0:
        raise RuntimeError("No F1 rows found.")

    if q_domain:
        stats_df = con.execute(
            f"""
            SELECT CAST({q_domain} AS VARCHAR) AS domain, COUNT(*)::BIGINT AS n
            FROM {source}
            WHERE {where_sql}
            GROUP BY 1
            ORDER BY n DESC, domain ASC
            """
        ).df()
    else:
        stats_df = pd.DataFrame({"domain": ["F1"], "n": [rows]})
    _print_domain_stats(pd.Series(stats_df["n"].values, index=stats_df["domain"].values))

    ch_df = con.execute(
        f"""
        SELECT CAST({channel_key_expr} AS VARCHAR) AS channel_key, COUNT(*)::BIGINT AS n
        FROM {source}
        WHERE {where_sql}
        GROUP BY 1
        ORDER BY n DESC, channel_key ASC
        """
    ).df()
    min_count = int(ch_df["n"].min())

    hash_expr = "''"
    if q_file:
        hash_expr = f"COALESCE(CAST({q_file} AS VARCHAR), '')"
    elif q_array:
        hash_expr = f"COALESCE(CAST({q_array} AS VARCHAR), '')"
    elif q_channel:
        hash_expr = f"COALESCE(CAST({q_channel} AS VARCHAR), '')"

    out_tmp = path.with_suffix(".tmp")
    con.execute(
        f"""
        COPY (
            WITH filtered AS (
                SELECT *, CAST({channel_key_expr} AS VARCHAR) AS __channel_key
                FROM {source}
                WHERE {where_sql}
            ),
            ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY __channel_key
                           ORDER BY hash({hash_expr}, {BALANCE_SEED})
                       ) AS rn
                FROM filtered
            )
            SELECT * EXCLUDE (rn, __channel_key)
            FROM ranked
            WHERE rn <= {min_count}
        ) TO '{out_tmp.as_posix()}' (FORMAT CSV, HEADER TRUE, DELIMITER ',')
        """
    )
    os.replace(out_tmp, path)
    print(f"[3/3] Balanced channels to {min_count:,} rows each")


def _prepare_with_pandas(path: Path) -> None:
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise RuntimeError("Training dataset is empty.")

    f1_mask, channel_keep_mask = _build_masks(df)
    filtered = df[f1_mask & channel_keep_mask].copy()
    if filtered.empty:
        raise RuntimeError("No F1 rows found.")

    if "domain" in filtered.columns:
        _print_domain_stats(filtered["domain"].value_counts(dropna=False))
    else:
        _print_domain_stats(pd.Series([len(filtered)], index=["F1"]))

    channel_key = _channel_series(filtered)
    filtered = filtered[channel_key.ne("")].copy()
    filtered["__channel_key"] = channel_key[channel_key.ne("")].values
    if filtered.empty:
        raise RuntimeError("No valid channel rows found for balancing.")

    balanced_parts = []
    for _, grp in filtered.groupby("__channel_key", dropna=False, sort=False):
        n = len(grp)
        if n >= TARGET_ROWS_PER_CHANNEL:
            part = grp.sample(n=TARGET_ROWS_PER_CHANNEL, replace=False, random_state=BALANCE_SEED)
        else:
            part = grp.sample(n=TARGET_ROWS_PER_CHANNEL, replace=True, random_state=BALANCE_SEED)
        balanced_parts.append(part)

    balanced = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1.0, random_state=BALANCE_SEED)
        .reset_index(drop=True)
    )
    balanced = balanced.drop(columns=["__channel_key"])

    out_tmp = path.with_suffix(".tmp")
    balanced.to_csv(out_tmp, index=False)
    chk_df = pd.read_csv(out_tmp, low_memory=False)
    check = _channel_series(chk_df)
    check = check[check.ne("")].value_counts(dropna=False)
    if check.nunique() != 1 or int(check.iloc[0]) != TARGET_ROWS_PER_CHANNEL:
        raise RuntimeError("Balanced output validation failed: channel counts differ.")
    os.replace(out_tmp, path)
    print(f"[3/3] Balanced channels to {TARGET_ROWS_PER_CHANNEL:,} rows each")


def main() -> None:
    print("[1/3] Loading data")
    print(f"  Source: {F1_ONLY_CSV}")

    if not F1_ONLY_CSV.exists():
        print(f"ERROR: Training dataset not found: {F1_ONLY_CSV}")
        return

    _prepare_with_pandas(F1_ONLY_CSV)

    print(f"Saved: {F1_ONLY_CSV}")


if __name__ == "__main__":
    main()
