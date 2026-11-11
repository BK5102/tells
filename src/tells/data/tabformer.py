"""Loader for the IBM TabFormer synthetic credit card dataset.

Source: https://github.com/IBM/TabFormer (Apache-2.0 repository). One raw row
is one card transaction. An account is one card, identified as "<User>_<Card>".
Measured facts about the file are in docs/data-fit.md.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from tells.data.schema import SORT_KEYS, TRANSACTION_SCHEMA

RAW_FILE = Path("data/raw/tabformer/card_transaction.v1.csv")

# Read every raw column as text, then convert explicitly, so a surprise in the
# file fails loudly instead of being guessed at.
_RAW_COLUMNS = [
    "User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip", "Merchant Name",
    "Merchant City", "Merchant State", "Zip", "MCC", "Errors?", "Is Fraud?",
]  # fmt: skip


def scan_tabformer(path: str | Path = RAW_FILE) -> pl.LazyFrame:
    """Lazily read the raw CSV into the standard schema (unsorted)."""
    raw = pl.scan_csv(path, schema={c: pl.Utf8 for c in _RAW_COLUMNS}).with_row_index("row")
    hour_minute = pl.col("Time").str.split_exact(":", 1)
    standard = [
        pl.col("row").cast(pl.Int64).alias("tx_id"),
        pl.format("{}_{}", pl.col("User"), pl.col("Card")).alias("account_id"),
        pl.datetime(
            pl.col("Year").cast(pl.Int32),
            pl.col("Month").cast(pl.Int32),
            pl.col("Day").cast(pl.Int32),
            hour_minute.struct.field("field_0").cast(pl.Int32),
            hour_minute.struct.field("field_1").cast(pl.Int32),
            time_unit="us",
        ).alias("ts"),
        pl.col("Amount").str.strip_prefix("$").cast(pl.Float64).alias("amount"),
        pl.col("Merchant Name").alias("counterparty"),
        pl.col("MCC").alias("category"),
        pl.col("Use Chip").alias("channel"),
        pl.col("Is Fraud?")
        .replace_strict({"Yes": 1, "No": 0}, return_dtype=pl.Int8)
        .alias("label"),
    ]
    extras = [
        pl.col("Errors?").alias("errors"),
        pl.col("Merchant City").alias("merchant_city"),
        pl.col("Merchant State").alias("merchant_state"),
    ]
    return raw.select(standard + extras).cast(TRANSACTION_SCHEMA)


def load_tabformer(
    path: str | Path = RAW_FILE, accounts: Iterable[str] | None = None
) -> pl.DataFrame:
    """Load TabFormer in the standard schema, sorted, optionally for some accounts only."""
    lf = scan_tabformer(path)
    if accounts is not None:
        lf = lf.filter(pl.col("account_id").is_in(list(accounts)))
    return lf.sort(SORT_KEYS).collect()
