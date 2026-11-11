from datetime import datetime

import polars as pl
import pytest

from tells.data.schema import TRANSACTION_SCHEMA, SchemaError, validate_transactions


def _valid() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tx_id": [0, 1, 2],
            "account_id": ["a", "a", "b"],
            "ts": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 1)],
            "amount": [1.0, -2.0, 3.0],
            "counterparty": ["m1", "m2", "m1"],
            "category": ["5411", "5411", "5300"],
            "channel": ["swipe", "chip", "online"],
            "label": [0, 1, 0],
        },
        schema=TRANSACTION_SCHEMA,
    )


def test_valid_frame_passes() -> None:
    validate_transactions(_valid())


def test_extra_columns_are_allowed() -> None:
    validate_transactions(_valid().with_columns(errors=pl.lit(None, pl.Utf8)))


def test_missing_column_fails() -> None:
    with pytest.raises(SchemaError, match="missing"):
        validate_transactions(_valid().drop("label"))


def test_wrong_dtype_fails() -> None:
    with pytest.raises(SchemaError, match="dtype"):
        validate_transactions(_valid().with_columns(pl.col("amount").cast(pl.Utf8)))


def test_null_in_required_column_fails() -> None:
    df = _valid().with_columns(ts=pl.when(pl.col("tx_id") == 1).then(None).otherwise(pl.col("ts")))
    with pytest.raises(SchemaError, match="null"):
        validate_transactions(df)


def test_label_outside_zero_one_fails() -> None:
    df = _valid().with_columns(label=pl.Series([0, 2, 0], dtype=pl.Int8))
    with pytest.raises(SchemaError, match="label"):
        validate_transactions(df)


def test_duplicate_tx_id_fails() -> None:
    df = _valid().with_columns(tx_id=pl.Series([0, 0, 2], dtype=pl.Int64))
    with pytest.raises(SchemaError, match="tx_id"):
        validate_transactions(df)


def test_unsorted_history_fails() -> None:
    # Each account's history must run oldest to newest.
    df = _valid().reverse()
    with pytest.raises(SchemaError, match="sorted"):
        validate_transactions(df)
