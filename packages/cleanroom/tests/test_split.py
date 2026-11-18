from datetime import date, datetime, timedelta

import polars as pl
import pytest
from cleanroom.split import LeakError, Split, SplitError, Window, rolling_origins

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


# --- rolling origins -------------------------------------------------------------


def test_rolling_origins_hand_computed() -> None:
    # Origins Jan 10, 17, 24; each tests the 7 days after a 1-day gap. The third
    # test window ends exactly at `end` and is kept; a fourth would end Feb 8.
    splits = rolling_origins(
        D(2020, 1, 10),
        D(2020, 2, 1),
        step=timedelta(days=7),
        horizon=timedelta(days=7),
        gap=timedelta(days=1),
    )
    assert [(s.train.end, s.test.start, s.test.end) for s in splits] == [
        (D(2020, 1, 10), D(2020, 1, 11), D(2020, 1, 18)),
        (D(2020, 1, 17), D(2020, 1, 18), D(2020, 1, 25)),
        (D(2020, 1, 24), D(2020, 1, 25), D(2020, 2, 1)),
    ]
    assert all(s.train.start is None and s.min_gap == timedelta(days=1) for s in splits)


def test_rolling_origins_accepts_dates_and_fixed_train_start() -> None:
    splits = rolling_origins(
        "2020-03-01",
        date(2020, 4, 1),
        timedelta(days=10),
        timedelta(days=10),
        train_start=date(2020, 1, 1),
    )
    assert len(splits) == 3
    assert all(s.train.start == D(2020, 1, 1) for s in splits)
    assert splits[0].test.start == splits[0].train.end  # zero gap by default


def test_rolling_origins_leak_check_on_rows() -> None:
    df = _frame()
    for split in rolling_origins(
        D(2020, 3, 1),
        D(2021, 1, 1),
        timedelta(days=30),
        timedelta(days=30),
        gap=timedelta(days=14),
    ):
        parts = split.apply(df, "t")
        split.check(parts, "t")
        assert parts["train"]["t"].max() < split.train.end
        assert parts["test"]["t"].min() - parts["train"]["t"].max() >= timedelta(days=14)


def test_rolling_origins_round_trip_through_dicts() -> None:
    splits = rolling_origins(D(2020, 2, 1), D(2020, 6, 1), timedelta(days=30), timedelta(days=30))
    assert [Split.from_dict(s.to_dict()) for s in splits] == splits


def test_rolling_origins_input_checks() -> None:
    with pytest.raises(SplitError, match="no origin fits"):
        rolling_origins(D(2020, 1, 1), D(2020, 1, 5), timedelta(days=1), timedelta(days=7))
    with pytest.raises(SplitError, match="step"):
        rolling_origins(D(2020, 1, 1), D(2020, 3, 1), timedelta(0), timedelta(days=7))
    with pytest.raises(SplitError, match="gap"):
        rolling_origins(
            D(2020, 1, 1),
            D(2020, 3, 1),
            timedelta(days=7),
            timedelta(days=7),
            gap=timedelta(days=-1),
        )
    with pytest.raises(SplitError):
        # A training start after the first origin is not a window.
        rolling_origins(
            D(2020, 1, 1),
            D(2020, 3, 1),
            timedelta(days=7),
            timedelta(days=7),
            train_start=D(2020, 2, 1),
        )
