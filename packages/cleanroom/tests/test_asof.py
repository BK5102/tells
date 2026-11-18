from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from cleanroom.asof import join_as_known
from cleanroom.pastonly import PastOnlyError, assert_past_only

D = datetime

# Releases of two measured periods. Period "2020-01" is first published on
# Feb 1, then revised on Mar 1 and Apr 1; "2020-02" is published on Mar 1.
RELEASES = pl.DataFrame(
    {
        "period": ["2020-01", "2020-01", "2020-01", "2020-02"],
        "value": [1.0, 1.1, 1.2, 2.0],
        "released": [date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1), date(2020, 3, 1)],
    }
)


def _snap(at, **kw) -> list[tuple]:
    out = join_as_known(RELEASES, "period", "value", "released", at, **kw)
    return list(zip(out["at"], out["period"], out["value"], strict=True))


def test_snapshot_strictly_before_by_default() -> None:
    # At Mar 1 midnight the Mar 1 releases are not known yet.
    assert _snap(date(2020, 3, 1)) == [(D(2020, 3, 1), "2020-01", 1.0)]
    # Later that day they are.
    assert _snap(D(2020, 3, 1, 12)) == [
        (D(2020, 3, 1, 12), "2020-01", 1.1),
        (D(2020, 3, 1, 12), "2020-02", 2.0),
    ]


def test_snapshot_inclusive_sees_releases_at_the_query_time() -> None:
    assert _snap(date(2020, 3, 1), inclusive=True) == [
        (D(2020, 3, 1), "2020-01", 1.1),
        (D(2020, 3, 1), "2020-02", 2.0),
    ]


def test_snapshot_at_several_times_and_before_anything() -> None:
    got = _snap([date(2020, 1, 15), date(2020, 5, 1)])
    # Nothing was known on Jan 15, so that time has no rows; by May the latest revision.
    assert got == [(D(2020, 5, 1), "2020-01", 1.2), (D(2020, 5, 1), "2020-02", 2.0)]
    series = pl.Series([D(2020, 5, 1)])
    assert _snap(series) == got


def test_query_frame_keeps_order_and_marks_unknown() -> None:
    queries = pl.DataFrame(
        {
            "period": ["2020-02", "2020-01", "2020-03", "2020-01"],
            "at": [D(2020, 3, 2), D(2020, 2, 15), D(2020, 6, 1), D(2020, 1, 1)],
            "tag": ["a", "b", "c", "d"],
        }
    )
    out = join_as_known(RELEASES, "period", "value", "released", queries)
    assert out["tag"].to_list() == ["a", "b", "c", "d"]
    assert out["value"].to_list() == [2.0, 1.0, None, None]
    assert out["released"].to_list() == [date(2020, 3, 1), date(2020, 2, 1), None, None]


def test_input_checks() -> None:
    dup = pl.concat([RELEASES, RELEASES.head(1)])
    with pytest.raises(ValueError, match="share a key and release time"):
        join_as_known(dup, "period", "value", "released", date(2020, 5, 1))
    with pytest.raises(ValueError, match="no column"):
        join_as_known(RELEASES, "period", "nope", "released", date(2020, 5, 1))
    with pytest.raises(ValueError, match="missing columns"):
        join_as_known(
            RELEASES, "period", "value", "released", pl.DataFrame({"at": [D(2020, 1, 1)]})
        )
    with pytest.raises(ValueError, match="query time"):
        join_as_known(RELEASES, "period", "value", "released", ["2020-01-01"])


# --- past-only proof with the pastonly helpers ---------------------------------------


def _events(n: int = 80) -> pl.DataFrame:
    # Each row is a new release of a key's value at time t, with unique (key, t).
    rng = np.random.default_rng(0)
    start = D(2021, 1, 1)
    hours = rng.choice(500, size=n, replace=False)
    return pl.DataFrame(
        {
            "id": range(n),
            "key": rng.integers(0, 4, n),
            "t": [start + timedelta(hours=int(h)) for h in hours],
            "x": rng.random(n),
        }
    ).sort("t")


def _last_known(df: pl.DataFrame, shift: timedelta = timedelta(0)) -> pl.DataFrame:
    queries = df.select("id", "key", at=pl.col("t") + shift)
    releases = df.select("key", "x", "t")
    out = join_as_known(releases, "key", "x", "t", queries)
    return out.select("id", last_x=pl.col("x"))


def _latest_revision(df: pl.DataFrame) -> pl.DataFrame:
    # The classic backtest leak: use whatever value is known at the end of the data.
    snap = join_as_known(df.select("key", "x", "t"), "key", "x", "t", df["t"].max(), inclusive=True)
    return df.join(snap.select("key", last_x="x"), on="key", how="left").select("id", "last_x")


def test_as_known_feature_is_past_only() -> None:
    assert_past_only(_last_known, _events(), "t", "id", n_probes=40)


def test_shifted_query_time_is_caught() -> None:
    with pytest.raises(PastOnlyError):
        assert_past_only(
            lambda df: _last_known(df, timedelta(hours=30)), _events(), "t", "id", n_probes=40
        )


def test_latest_revision_leak_is_caught() -> None:
    with pytest.raises(PastOnlyError):
        assert_past_only(_latest_revision, _events(), "t", "id", n_probes=40)
