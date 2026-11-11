from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from tells.data.schema import validate_transactions
from tells.data.tabformer import load_tabformer

FIXTURE = Path(__file__).parent / "fixtures" / "tabformer_tiny.csv"
FULL_PARQUET = Path(__file__).parents[1] / "data" / "interim" / "tabformer.parquet"


@pytest.fixture(scope="module")
def tx() -> pl.DataFrame:
    return load_tabformer(FIXTURE)


def test_matches_standard_schema(tx: pl.DataFrame) -> None:
    validate_transactions(tx)
    assert tx.height == 7


def test_account_id_is_user_and_card(tx: pl.DataFrame) -> None:
    assert tx["account_id"].unique().sort().to_list() == ["0_0", "0_1", "1_0"]


def test_history_sorted_oldest_first_with_raw_row_as_id(tx: pl.DataFrame) -> None:
    # Raw rows 0 and 1 are out of time order in the file; the loader fixes that
    # and keeps the raw row number as tx_id.
    first_two = tx.filter(pl.col("account_id") == "0_0").head(2)
    assert first_two["ts"].to_list() == [datetime(2002, 9, 1, 6, 21), datetime(2002, 9, 1, 6, 42)]
    assert first_two["tx_id"].to_list() == [1, 0]


def test_amount_parses_dollars_and_refunds(tx: pl.DataFrame) -> None:
    amounts = dict(zip(tx["tx_id"], tx["amount"], strict=True))
    assert amounts[1] == pytest.approx(134.09)
    assert amounts[2] == pytest.approx(-77.0)


def test_labels_and_text_fields(tx: pl.DataFrame) -> None:
    assert tx["label"].sum() == 3
    row = tx.filter(pl.col("tx_id") == 3).row(0, named=True)
    assert row["channel"] == "Online Transaction"
    assert row["category"] == "5311"
    assert row["counterparty"] == "999"
    assert row["merchant_state"] is None
    errors = dict(zip(tx["tx_id"], tx["errors"], strict=True))
    assert errors[2] == "Bad PIN,"


def test_account_filter(tx: pl.DataFrame) -> None:
    only = load_tabformer(FIXTURE, accounts=["0_1"])
    assert only["account_id"].unique().to_list() == ["0_1"]
    assert only.height == 3


@pytest.mark.data
@pytest.mark.skipif(not FULL_PARQUET.exists(), reason="run `uv run tells prepare tabformer` first")
def test_full_file_matches_measured_counts() -> None:
    # Numbers measured independently in docs/data-fit.md.
    lf = pl.scan_parquet(FULL_PARQUET)
    stats = lf.select(
        rows=pl.len(), accounts=pl.col("account_id").n_unique(), positives=pl.col("label").sum()
    ).collect()
    assert stats.row(0) == (24_386_900, 6_139, 29_757)
