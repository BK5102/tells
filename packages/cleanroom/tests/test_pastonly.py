from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from cleanroom.pastonly import PastOnlyError, assert_ignores, assert_past_only


def _frame(n: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    start = datetime(2020, 1, 1)
    # Hour-rounded times create many ties within each group.
    t = [start + timedelta(hours=int(h)) for h in np.sort(rng.integers(0, 48, n))]
    return pl.DataFrame(
        {
            "id": range(n),
            "g": rng.integers(0, 3, n),
            "t": t,
            "x": rng.random(n),
            "y": rng.integers(0, 2, n),
        }
    ).sort("g", "t", "id")


def honest(df: pl.DataFrame) -> pl.DataFrame:
    # Sum of x over strictly earlier times in the same group.
    per_time = df.group_by("g", "t").agg(sx=pl.col("x").sum()).sort("g", "t")
    per_time = per_time.with_columns(past_sum=(pl.col("sx").cum_sum() - pl.col("sx")).over("g"))
    return df.join(per_time, on=["g", "t"]).select("id", "past_sum")


def leaky_future(df: pl.DataFrame) -> pl.DataFrame:
    # Total of x over the whole group, future included.
    return df.select("id", total=pl.col("x").sum().over("g"))


def leaky_ties(df: pl.DataFrame) -> pl.DataFrame:
    # Counts rows at the same timestamp, which the probe could not know yet.
    return df.select("id", same_time=pl.len().over("g", "t"))


def reads_outcome(df: pl.DataFrame) -> pl.DataFrame:
    return df.select("id", y_copy=pl.col("y") * 2)


def test_honest_builder_passes() -> None:
    assert_past_only(honest, _frame(), "t", "id", n_probes=30)


def test_future_leak_is_caught() -> None:
    with pytest.raises(PastOnlyError):
        assert_past_only(leaky_future, _frame(), "t", "id", n_probes=30)


def test_same_timestamp_leak_is_caught() -> None:
    with pytest.raises(PastOnlyError):
        assert_past_only(leaky_ties, _frame(), "t", "id", n_probes=30)


def test_outcome_reader_is_caught_and_honest_passes() -> None:
    with pytest.raises(PastOnlyError):
        assert_ignores(reads_outcome, _frame(), ["y"], "id")
    assert_ignores(honest, _frame(), ["y"], "id")
