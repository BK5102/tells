"""Prove that a feature builder only uses the past.

`assert_past_only` picks probe rows, rebuilds the features from a copy of the
data that contains only rows strictly earlier than the probe (plus the probe
itself), and requires the probe's features to be identical to the ones built
from the full data. Any use of later rows, or of other rows sharing the
probe's timestamp, makes the two disagree.

`assert_ignores` shuffles columns the builder must never read (for example
the outcome) and requires the features to stay identical.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import polars as pl

Builder = Callable[[pl.DataFrame], pl.DataFrame]


class PastOnlyError(AssertionError):
    """A feature builder used information it should not have."""


def _compare(expected: pl.DataFrame, got: pl.DataFrame, id_col: str, what: str) -> None:
    cols = [c for c in expected.columns if c != id_col]
    joined = expected.join(got, on=id_col, how="left", suffix="__got")
    if joined.height != expected.height:
        raise PastOnlyError(f"{what}: row count changed")
    for c in cols:
        a = joined[c]
        b = joined[f"{c}__got"]
        if a.dtype.is_float():
            a_np, b_np = a.to_numpy(), b.to_numpy()
            same = np.isclose(a_np, b_np, rtol=1e-9, atol=1e-12, equal_nan=True)
        else:
            same = (a.eq_missing(b)).to_numpy()
        if not same.all():
            bad = joined.filter(~pl.Series(same))[id_col].head(5).to_list()
            raise PastOnlyError(f"{what}: feature {c!r} differs for {id_col} {bad}")


def assert_past_only(
    build: Builder,
    df: pl.DataFrame,
    time_col: str,
    id_col: str,
    n_probes: int = 25,
    seed: int = 0,
    probe_ids: Sequence | None = None,
) -> None:
    """Raise PastOnlyError if `build` uses rows at or after a row's own time."""
    full = build(df)
    if id_col not in full.columns:
        raise PastOnlyError(f"builder output must keep the id column {id_col!r}")
    if probe_ids is None:
        rng = np.random.default_rng(seed)
        ids = df[id_col].to_numpy()
        probe_ids = rng.choice(ids, size=min(n_probes, len(ids)), replace=False).tolist()

    times = dict(zip(df[id_col].to_list(), df[time_col].to_list(), strict=True))
    for pid in probe_ids:
        t = times[pid]
        past = df.filter((pl.col(time_col) < t) | (pl.col(id_col) == pid))
        got = build(past).filter(pl.col(id_col) == pid)
        expected = full.filter(pl.col(id_col) == pid)
        _compare(expected, got, id_col, f"probe {pid} at {t}")


def assert_ignores(
    build: Builder, df: pl.DataFrame, columns: Sequence[str], id_col: str, seed: int = 0
) -> None:
    """Raise PastOnlyError if shuffling `columns` changes any built feature."""
    rng = np.random.default_rng(seed)
    shuffled = df.with_columns(
        pl.Series(c, df[c].to_numpy()[rng.permutation(df.height)]) for c in columns
    )
    _compare(build(df), build(shuffled), id_col, f"shuffling {list(columns)}")
