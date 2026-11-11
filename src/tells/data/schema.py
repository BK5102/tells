"""The standard transaction schema every dataset loader must produce.

One row is one transaction. Rows are sorted so that each account's history
runs oldest to newest, with `tx_id` breaking ties between transactions that
share a timestamp. Loaders may add extra, dataset-specific columns after these.
"""

from __future__ import annotations

import polars as pl

TRANSACTION_SCHEMA: dict[str, pl.DataType] = {
    "tx_id": pl.Int64(),  # unique, stable id (for TabFormer: the raw file row number)
    "account_id": pl.Utf8(),  # whose history this transaction belongs to
    "ts": pl.Datetime("us"),  # when it happened
    "amount": pl.Float64(),  # negative values are refunds or reversals
    "counterparty": pl.Utf8(),  # merchant or receiving account
    "category": pl.Utf8(),  # merchant category code or payment format
    "channel": pl.Utf8(),  # how it was made (swipe, chip, online, ...)
    "label": pl.Int8(),  # 1 = fraud or laundering, 0 = normal
}

REQUIRED_NON_NULL = ("tx_id", "account_id", "ts", "amount", "label")
SORT_KEYS = ["account_id", "ts", "tx_id"]


class SchemaError(ValueError):
    """A loader produced data that does not match the standard schema."""


def validate_transactions(df: pl.DataFrame) -> None:
    """Raise SchemaError unless `df` follows the standard transaction schema."""
    missing = [c for c in TRANSACTION_SCHEMA if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    for col, dtype in TRANSACTION_SCHEMA.items():
        if df.schema[col] != dtype:
            raise SchemaError(f"column {col!r} has dtype {df.schema[col]}, expected {dtype}")

    for col in REQUIRED_NON_NULL:
        n_null = df[col].null_count()
        if n_null:
            raise SchemaError(f"column {col!r} has {n_null} null values")

    bad_labels = df.filter(~pl.col("label").is_in([0, 1])).height
    if bad_labels:
        raise SchemaError(f"label must be 0 or 1; {bad_labels} rows are not")

    if df["tx_id"].n_unique() != df.height:
        raise SchemaError("tx_id values are not unique")

    if not df.select(SORT_KEYS).equals(df.select(SORT_KEYS).sort(SORT_KEYS)):
        raise SchemaError(f"rows must be sorted by {SORT_KEYS}")
