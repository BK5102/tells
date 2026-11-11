from datetime import date, datetime, timedelta

import polars as pl
import pytest
from cleanroom.split import LeakError, Split, SplitError, Window

D = datetime


def _frame() -> pl.DataFrame:
    # One row per day for all of 2020.
    days = pl.datetime_range(D(2020, 1, 1), D(2020, 12, 31), "1d", eager=True)
    return pl.DataFrame({"t": days, "i": range(len(days))})


def _split(**kw) -> Split:
    params = dict(
        train=Window(None, D(2020, 6, 1)),
        valid=Window(D(2020, 7, 1), D(2020, 9, 1)),
        test=Window(D(2020, 10, 1), D(2021, 1, 1)),
        min_gap=timedelta(days=30),
    )
    params.update(kw)
    return Split(**params)


# --- config validation -------------------------------------------------------


def test_window_end_must_follow_start() -> None:
    with pytest.raises(SplitError):
        Window(D(2020, 2, 1), D(2020, 1, 1))


def test_overlapping_windows_rejected() -> None:
    with pytest.raises(SplitError, match="before"):
        _split(test=Window(D(2020, 5, 1), D(2021, 1, 1)), valid=None)


def test_gap_smaller_than_minimum_rejected() -> None:
    with pytest.raises(SplitError, match="gap"):
        _split(test=Window(D(2020, 9, 15), D(2021, 1, 1)))


def test_only_the_first_window_may_be_open_ended() -> None:
    with pytest.raises(SplitError):
        _split(test=Window(None, D(2021, 1, 1)))


# --- applying the split: the leak tests ---------------------------------------


def test_latest_train_time_is_before_earliest_test_time() -> None:
    parts = _split().apply(_frame(), "t")
    assert parts["train"]["t"].max() < parts["valid"]["t"].min()
    assert parts["valid"]["t"].max() < parts["test"]["t"].min()


def test_observed_gap_is_at_least_the_minimum() -> None:
    parts = _split().apply(_frame(), "t")
    assert parts["test"]["t"].min() - parts["valid"]["t"].max() >= timedelta(days=30)
    assert parts["valid"]["t"].min() - parts["train"]["t"].max() >= timedelta(days=30)


def test_no_row_in_two_parts_and_gap_rows_dropped() -> None:
    df = _frame()
    parts = _split().apply(df, "t")
    ids = pl.concat([p["i"] for p in parts.values()])
    assert ids.is_unique().all()
    gap_rows = df.filter((pl.col("t") >= D(2020, 6, 1)) & (pl.col("t") < D(2020, 7, 1)))
    assert not set(gap_rows["i"]) & set(ids)


def test_windows_are_half_open() -> None:
    # A row exactly at a window's end belongs to the next window (or the gap),
    # never to both.
    parts = _split().apply(_frame(), "t")
    assert parts["train"]["t"].max() == D(2020, 5, 31)
    assert parts["test"]["t"].min() == D(2020, 10, 1)


def test_works_on_lazy_frames() -> None:
    parts = _split().apply(_frame().lazy(), "t")
    assert isinstance(parts["train"], pl.LazyFrame)


def test_check_catches_a_leaky_part() -> None:
    split = _split()
    parts = split.apply(_frame(), "t")
    # Smuggle a test-period row into train, as a buggy pipeline might.
    parts["train"] = pl.concat([parts["train"], parts["test"].head(1)])
    with pytest.raises(LeakError):
        split.check(parts, "t")


def test_check_passes_on_clean_parts() -> None:
    split = _split()
    split.check(split.apply(_frame(), "t"), "t")


def test_split_without_valid() -> None:
    parts = _split(valid=None).apply(_frame(), "t")
    assert list(parts) == ["train", "test"]


# --- config round trip (needed by the lock on Day 9) ---------------------------


def test_from_dict_accepts_yaml_style_dates() -> None:
    cfg = {
        "min_gap_days": 30,
        "train": {"start": None, "end": date(2020, 6, 1)},
        "valid": {"start": "2020-07-01", "end": "2020-09-01"},
        "test": {"start": date(2020, 10, 1), "end": date(2021, 1, 1)},
    }
    assert Split.from_dict(cfg) == _split()


def test_to_dict_round_trips() -> None:
    split = _split()
    assert Split.from_dict(split.to_dict()) == split
