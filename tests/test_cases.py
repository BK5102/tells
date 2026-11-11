from datetime import date
from pathlib import Path

import polars as pl
import pytest

from tells.cases import build_cases
from tells.data.tabformer import load_tabformer

FIXTURE = Path(__file__).parent / "fixtures" / "tabformer_tiny.csv"


@pytest.fixture(scope="module")
def tx() -> pl.DataFrame:
    return load_tabformer(FIXTURE)


def _case(cases: pl.DataFrame, account: str, day: date) -> dict:
    return cases.filter((pl.col("account_id") == account) & (pl.col("day") == day)).row(
        0, named=True
    )


def test_one_case_per_account_per_active_day(tx: pl.DataFrame) -> None:
    cases = build_cases(tx)
    assert cases.height == 5
    assert cases.select("account_id", "day").is_unique().all()
    assert cases["n_tx"].sum() == tx.height


def test_case_label_is_any_positive(tx: pl.DataFrame) -> None:
    cases = build_cases(tx)
    assert _case(cases, "0_0", date(2002, 9, 1))["label"] == 0
    assert _case(cases, "0_1", date(2002, 9, 2))["label"] == 1
    assert _case(cases, "0_1", date(2002, 9, 2))["n_tx"] == 2


def test_first_positive_day_flag(tx: pl.DataFrame) -> None:
    cases = build_cases(tx)
    assert _case(cases, "0_1", date(2002, 9, 2))["is_first_positive"] is True
    assert _case(cases, "0_1", date(2002, 9, 3))["is_first_positive"] is False
    assert cases["is_first_positive"].sum() == 1


def test_case_score_is_max_transaction_score(tx: pl.DataFrame) -> None:
    scored = tx.with_columns(score=pl.col("amount").abs())
    cases = build_cases(scored, score_col="score")
    assert _case(cases, "0_0", date(2002, 9, 1))["score"] == pytest.approx(134.09)


def test_cases_sorted_by_day_then_account(tx: pl.DataFrame) -> None:
    cases = build_cases(tx)
    assert cases.equals(cases.sort("day", "account_id"))
