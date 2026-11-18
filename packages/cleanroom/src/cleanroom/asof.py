"""As-known joins: use a value only as it was published, not as it was later revised.

Many series are revised after first release: the figure for March may be
published in April and corrected in May and June. A record of every release
("vintage") has one row per (key, release time), where the key names what
was measured (for example the period) and `realtime_start` is when that
release became available. A forecast made at time t may only use, for each
key, the latest release that existed before t. Using today's revised figure
in a backtest is a leak: nobody had it at the time.

Choice: by default a release counts as known only when its `realtime_start`
is strictly before the query time, matching the past-only rule used
everywhere else in this package (rows with time less than t). A release
stamped with a date is often available only later that day, so a query at
that day's midnight should not see it. Pass `inclusive=True` when the data
guarantee that a release is available at its stamped time.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

import polars as pl


def _as_datetime_expr(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Datetime("us"))


def join_as_known(
    values: pl.DataFrame,
    key_col: str,
    value_col: str,
    realtime_start_col: str,
    at: datetime | date | Sequence[datetime | date] | pl.Series | pl.DataFrame,
    *,
    at_col: str = "at",
    inclusive: bool = False,
) -> pl.DataFrame:
    """For each query time (and key), the latest value released before that time.

    `values` holds one row per release: `key_col`, `value_col`, and
    `realtime_start_col` (date or datetime). Two releases of one key with the
    same `realtime_start` are ambiguous and raise ValueError.

    `at` is either:

    - a DataFrame of queries with columns `key_col` and `at_col`: every query
      row is returned with `value_col` and `realtime_start_col` added (null
      when nothing about that key was released yet), in the original order;
    - one time or a sequence of times: a snapshot of everything known at each
      time, one row per (time, key) with columns `at_col`, `key_col`,
      `value_col`, `realtime_start_col`. Keys with no release yet are left out.

    Times are compared as microsecond datetimes; a date means its midnight.
    """
    for c in (key_col, value_col, realtime_start_col):
        if c not in values.columns:
            raise ValueError(f"values has no column {c!r}")
    if values[realtime_start_col].null_count():
        raise ValueError(f"{realtime_start_col!r} has missing release times")
    dupes = values.select(key_col, realtime_start_col).is_duplicated()
    if dupes.any():
        bad = values.filter(dupes)[key_col].head(3).to_list()
        raise ValueError(f"several releases share a key and release time, for keys {bad}")

    releases = (
        values.select(key_col, value_col, realtime_start_col)
        .with_columns(_asof_t=_as_datetime_expr(realtime_start_col))
        .sort("_asof_t")
    )

    if isinstance(at, pl.DataFrame):
        missing = {key_col, at_col} - set(at.columns)
        if missing:
            raise ValueError(f"query frame is missing columns {sorted(missing)}")
        clash = {value_col, realtime_start_col} & set(at.columns)
        if clash:
            raise ValueError(f"query frame already has columns {sorted(clash)}")
        if at[at_col].null_count():
            raise ValueError(f"query column {at_col!r} has missing times")
        queries = at.with_row_index("_asof_row").with_columns(_asof_t=_as_datetime_expr(at_col))
        snapshot = False
    else:
        times = _query_times(at)
        keys = values.select(key_col).unique().sort(key_col)
        queries = (
            pl.DataFrame({at_col: times})
            .with_columns(pl.col(at_col).cast(pl.Datetime("us")))
            .join(keys, how="cross")
            .with_row_index("_asof_row")
            .with_columns(_asof_t=pl.col(at_col))
        )
        snapshot = True

    joined = (
        queries.sort("_asof_t")
        .join_asof(
            releases,
            on="_asof_t",
            by=key_col,
            strategy="backward",
            allow_exact_matches=inclusive,
            check_sortedness=False,  # both sides are sorted on the time just above
        )
        .sort("_asof_row")
        .drop("_asof_row", "_asof_t")
    )
    if snapshot:
        joined = joined.filter(pl.col(realtime_start_col).is_not_null()).select(
            at_col, key_col, value_col, realtime_start_col
        )
    return joined


def _query_times(at: Any) -> list[datetime]:
    items = at.to_list() if isinstance(at, pl.Series) else at
    if isinstance(items, datetime | date):
        items = [items]
    out = []
    for t in items:
        if isinstance(t, datetime):
            out.append(t)
        elif isinstance(t, date):
            out.append(datetime(t.year, t.month, t.day))
        else:
            raise ValueError(f"cannot read {t!r} as a query time")
    if not out:
        raise ValueError("no query times given")
    return out
