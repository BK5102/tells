"""Turn transactions into account-day cases, the unit an analyst reviews.

One case is one account on one day it was active. Its label is 1 if any of
that day's transactions is labeled; its score (when scores exist) is the
highest transaction score that day. See docs/decisions.md D3.
"""

from __future__ import annotations

import polars as pl


def build_cases(
    tx: pl.DataFrame,
    score_col: str | None = None,
    known_positive_accounts: set[str] | None = None,
) -> pl.DataFrame:
    """Group standard-schema transactions into account-day cases.

    Columns: account_id, day, n_tx, label, first_ts, last_ts, is_first_positive,
    and score if `score_col` is given.

    `is_first_positive` marks an account's first-ever positive day: the first
    positive day in `tx`, for accounts NOT in `known_positive_accounts` (accounts
    already labeled positive before the evaluation window). Catching an account
    the first day it goes bad is what stops further losses; catching an account
    that was already known bad is not a new catch. See methodology M12.
    """
    aggs = [
        pl.len().cast(pl.Int64).alias("n_tx"),
        pl.col("label").max().alias("label"),
        pl.col("ts").min().alias("first_ts"),
        pl.col("ts").max().alias("last_ts"),
    ]
    if score_col is not None:
        aggs.append(pl.col(score_col).max().alias("score"))
    if "n_prior" in tx.columns:
        # Events the account had before its first transaction that day.
        aggs.append(pl.col("n_prior").min().alias("history_len"))

    cases = tx.group_by("account_id", pl.col("ts").dt.date().alias("day")).agg(aggs)

    known = list(known_positive_accounts or ())
    first_positive_day = (
        cases.filter((pl.col("label") == 1) & ~pl.col("account_id").is_in(known))
        .group_by("account_id")
        .agg(pl.col("day").min().alias("_first_pos_day"))
    )
    return (
        cases.join(first_positive_day, on="account_id", how="left")
        .with_columns(
            is_first_positive=(pl.col("day") == pl.col("_first_pos_day")).fill_null(False)
        )
        .drop("_first_pos_day")
        .sort("day", "account_id")
    )
