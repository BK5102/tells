from datetime import date

import polars as pl
import pytest

from tells.evaluate import case_metrics


def test_case_metrics_use_daily_budget_and_first_day_flag() -> None:
    # Two days, 4 cases each. Positives: one card on both days (first day = day 1).
    cases = pl.DataFrame(
        {
            "day": [date(2020, 1, 1)] * 4 + [date(2020, 1, 2)] * 4,
            "score": [4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0],
            "label": [1, 0, 0, 0, 0, 0, 0, 1],
            "is_first_positive": [True] + [False] * 7,
        }
    )
    m = case_metrics(cases, (0.25,))
    b = m["by_budget"]["0.25"]
    # One slot per day: day 1 catches its positive, day 2 misses its positive.
    assert b["recall"] == pytest.approx(0.5)
    assert b["precision"] == pytest.approx(0.5)
    assert b["first_day_recall"] == pytest.approx(1.0)
    assert m["n_positive_cases"] == 2
