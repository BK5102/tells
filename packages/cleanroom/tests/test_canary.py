from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from cleanroom.canary import CanaryNotCaughtError, run_canary


def _frame() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    return pl.DataFrame(
        {
            "id": range(n),
            "g": rng.integers(0, 4, n),
            "t": [datetime(2020, 1, 1) + timedelta(hours=int(h)) for h in rng.integers(0, 200, n)],
            "x": rng.random(n),
            "y": rng.integers(0, 2, n),
        }
    ).sort("g", "t", "id")


def honest(df: pl.DataFrame) -> pl.DataFrame:
    per_time = df.group_by("g", "t").agg(sx=pl.col("x").sum()).sort("g", "t")
    per_time = per_time.with_columns(past=(pl.col("sx").cum_sum() - pl.col("sx")).over("g"))
    return df.join(per_time, on=["g", "t"]).select("id", "past")


def test_both_canaries_caught_and_honest_builder_clean() -> None:
    report = run_canary(honest, _frame(), "t", "id", "g", "x", "y")
    assert report.ok
    assert report.to_dict()["ok"] is True


def test_a_leaky_builder_is_reported_unclean() -> None:
    def leaky(df: pl.DataFrame) -> pl.DataFrame:
        return df.select("id", total=pl.col("x").sum().over("g"))

    report = run_canary(leaky, _frame(), "t", "id", "g", "x", "y")
    assert report.honest_builder_clean is False
    assert report.ok is False


def test_broken_checks_raise(monkeypatch) -> None:
    # If the past-only check stopped working, the canary must say so loudly.
    import cleanroom.canary as canary

    monkeypatch.setattr(canary, "assert_past_only", lambda *a, **k: None)
    monkeypatch.setattr(canary, "assert_ignores", lambda *a, **k: None)
    with pytest.raises(CanaryNotCaughtError):
        run_canary(honest, _frame(), "t", "id", "g", "x", "y")


def test_sample_without_both_outcomes_is_refused() -> None:
    with pytest.raises(ValueError, match="both outcomes"):
        run_canary(honest, _frame().with_columns(y=pl.lit(0)), "t", "id", "g", "x", "y")
