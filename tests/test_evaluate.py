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


def test_calibration_only_for_probability_scores() -> None:
    from tells.evaluate import transaction_metrics

    scored = pl.DataFrame({"score": [0.9, 0.1, 0.2, 0.8], "label": [1, 0, 0, 1]})
    assert "calibration" not in transaction_metrics(scored, outputs_probability=False)
    cal = transaction_metrics(scored, outputs_probability=True)["calibration"]
    assert cal["brier"] == pytest.approx((0.01 + 0.01 + 0.04 + 0.04) / 4)
    assert cal["brier_of_base_rate"] == pytest.approx(0.25)


def test_period_without_positives_reports_undefined_not_zero() -> None:
    cases = pl.DataFrame(
        {
            "day": [date(2020, 1, 1)] * 3,
            "score": [1.0, 2.0, 3.0],
            "label": [0, 0, 0],
            "is_first_positive": [False] * 3,
        }
    )
    m = case_metrics(cases, (0.5,))
    assert m["case_pr_auc"] is None
    assert m["by_budget"]["0.5"]["recall"] is None
    assert m["by_budget"]["0.5"]["first_day_recall"] is None


def test_bootstrap_is_centered_on_the_point_estimate() -> None:
    # Regression: resampling accounts with replacement and re-ranking made
    # duplicate accounts compete for the same review slots, which biased every
    # resample downward (point outside its own interval). The queue is now
    # formed once on the real data and only the averaging is resampled.
    import numpy as np

    from tells.evaluate import compare_to_reference

    rng = np.random.default_rng(0)
    n_acct, n_days = 300, 20
    rows = []
    for a in range(n_acct):
        bad_day = rng.integers(0, n_days) if a < 60 else None
        for d in range(n_days):
            y = int(bad_day == d)
            rows.append((f"a{a}", date(2020, 1, 1 + d % 28), y, y and True, rng.random() + 2 * y))
    df = pl.DataFrame(
        rows, schema=["account_id", "day", "label", "is_first_positive", "s"], orient="row"
    )
    df = df.sort("day", "account_id")
    ref = df.with_columns(score=pl.col("s"))
    weak = df.with_columns(score=pl.col("s") * 0 + pl.Series(rng.random(df.height)))
    out = compare_to_reference(
        {"ref": ref, "weak": weak}, "ref", "first_day_recall", 0.05, n_boot=300, seed=0
    )
    r = out["results"]["weak"]
    assert r["ci_low"] <= r["diff"] <= r["ci_high"]
    assert abs(r["boot_mean_diff"] - r["diff"]) < 0.03


def test_history_slices_keep_the_real_queue() -> None:
    # One day, 4 cases, 1 slot (25%). The top case is a new bad account with
    # short history; a second new bad account with long history is ranked last.
    cases = pl.DataFrame(
        {
            "day": [date(2020, 1, 1)] * 4,
            "score": [4.0, 3.0, 2.0, 1.0],
            "label": [1, 0, 0, 1],
            "is_first_positive": [True, False, False, True],
            "history_len": [3, 50, 50, 40],
        }
    )
    m = case_metrics(cases, (0.25,))
    h = m["by_history"]
    assert h["short_history"]["by_budget"]["0.25"]["first_day_recall"] == pytest.approx(1.0)
    assert h["long_history"]["by_budget"]["0.25"]["first_day_recall"] == pytest.approx(0.0)
    assert h["short_history"]["n_first_positive_cases"] == 1
