from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from tells.cases import build_cases
from tells.data.amlworld import TRUNCATE_AT, load_amlworld, usd_rates
from tells.data.schema import validate_transactions

FIXTURE = Path(__file__).parent / "fixtures" / "amlworld_tiny.csv"
FULL = Path(__file__).parents[1] / "data" / "interim" / "amlworld.parquet"


@pytest.fixture(scope="module")
def ev() -> pl.DataFrame:
    return load_amlworld(FIXTURE)


def test_matches_standard_schema(ev: pl.DataFrame) -> None:
    validate_transactions(ev)


def test_end_of_data_tail_is_truncated(ev: pl.DataFrame) -> None:
    # The 2022-09-11 row is after the cut and must be gone.
    assert datetime(2022, 9, 11) == TRUNCATE_AT
    assert ev["ts"].max() < TRUNCATE_AT
    assert ev["source_tx"].max() == 5


def test_each_transaction_appears_in_both_histories(ev: pl.DataFrame) -> None:
    # Raw row 1 (A1 pays B1) is an outgoing event for A1 and incoming for B1.
    row1 = ev.filter(pl.col("source_tx") == 1).sort("direction")
    assert row1["account_id"].to_list() == ["020_B1", "010_A1"]
    assert row1["direction"].to_list() == [-1, 1]
    assert row1["channel"].to_list() == ["received", "sent"]
    assert row1["counterparty"].to_list() == ["010_A1", "020_B1"]


def test_self_transfer_is_counted_once(ev: pl.DataFrame) -> None:
    row0 = ev.filter(pl.col("source_tx") == 0)
    assert row0.height == 1
    assert row0["channel"].item() == "self"
    assert row0["direction"].item() == 0


def test_amounts_converted_to_usd_with_day_one_rates(ev: pl.DataFrame) -> None:
    # 100 EUR paid; the fixture's only EUR-to-USD trade gives 1.1718 USD per EUR.
    amt = ev.filter(pl.col("source_tx") == 1)["amount"].unique().item()
    assert amt == pytest.approx(117.18)


def test_labels_follow_the_transaction(ev: pl.DataFrame) -> None:
    lab = ev.filter(pl.col("source_tx") == 3)
    assert lab["label"].to_list() == [1, 1]
    assert ev["label"].sum() == 2


def test_payment_format_is_the_category(ev: pl.DataFrame) -> None:
    assert set(ev["category"]) == {"Reinvestment", "ACH", "Cheque", "Credit Card", "Wire"}


def test_cases_flag_both_accounts_of_a_laundering_transaction(ev: pl.DataFrame) -> None:
    cases = build_cases(ev)
    positive = cases.filter(pl.col("label") == 1)["account_id"].sort().to_list()
    assert positive == ["010_A1", "030_C1"]


def test_rates_use_first_day_only() -> None:
    raw = pl.DataFrame(
        {
            "ts": [datetime(2022, 9, 1), datetime(2022, 9, 5)],
            "paid": [100.0, 100.0],
            "pay_cur": ["Euro", "Euro"],
            "received": [120.0, 999.0],
            "rec_cur": ["US Dollar", "US Dollar"],
        }
    )
    assert usd_rates(raw)["Euro"] == pytest.approx(1.2)


@pytest.mark.data
@pytest.mark.skipif(not FULL.exists(), reason="run `uv run tells prepare amlworld` first")
def test_full_file_counts() -> None:
    # Measured in docs/data-fit.md: 4,522 laundering transactions before 2022-09-11.
    lf = pl.scan_parquet(FULL)
    stats = lf.select(
        tx=pl.col("source_tx").n_unique(),
        laundering=pl.col("source_tx").filter(pl.col("label") == 1).n_unique(),
    ).collect()
    assert stats.row(0) == (5_077_237, 4_522)
