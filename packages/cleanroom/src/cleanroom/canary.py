"""Canaries: plant a deliberate cheat and prove the leak checks catch it.

A coal-mine canary is there to fail first. Here, two leaky features are added
on top of an honest feature builder:

- `future_value`: the next row's value of a column in the same group (reads
  the future).
- `future_label`: the next row's outcome in the same group (reads the future
  and the outcome).

`run_canary` requires that (1) the honest builder passes the past-only and
ignore-outcome checks, and (2) both canaries are caught. If a canary slips
through, the checks themselves are broken, and nothing they approved can be
trusted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from cleanroom.pastonly import PastOnlyError, assert_ignores, assert_past_only

Builder = Callable[[pl.DataFrame], pl.DataFrame]


class CanaryNotCaughtError(AssertionError):
    """A planted leak was not detected, so the leak checks cannot be trusted."""


@dataclass(frozen=True)
class CanaryReport:
    honest_builder_clean: bool
    future_value_caught: bool
    future_label_caught: bool

    @property
    def ok(self) -> bool:
        return self.honest_builder_clean and self.future_value_caught and self.future_label_caught

    def to_dict(self) -> dict[str, bool]:
        return {**self.__dict__, "ok": self.ok}


def _with_canary(build: Builder, expr: pl.Expr, id_col: str) -> Builder:
    def leaky(df: pl.DataFrame) -> pl.DataFrame:
        return build(df).join(df.select(id_col, canary=expr), on=id_col, how="left")

    return leaky


def run_canary(
    build: Builder,
    df: pl.DataFrame,
    time_col: str,
    id_col: str,
    group_col: str,
    value_col: str,
    label_col: str,
    n_probes: int = 15,
    seed: int = 0,
) -> CanaryReport:
    """Check the honest builder, then plant both canaries. Raises if a canary is missed."""
    df = df.sort(group_col, time_col, id_col)
    if df[label_col].n_unique() < 2:
        # Shuffling a constant outcome changes nothing, so the outcome canary
        # could never be caught. Refuse instead of reporting a false failure.
        raise ValueError("the canary sample needs both outcomes; include positive groups")
    honest_clean = True
    try:
        assert_past_only(build, df, time_col, id_col, n_probes=n_probes, seed=seed)
        assert_ignores(build, df, [label_col], id_col, seed=seed)
    except PastOnlyError:
        honest_clean = False

    # Probe rows that have a later row in their group, so the canary has something to leak.
    has_next = df.filter(pl.col(id_col).shift(-1).over(group_col).is_not_null())
    probes = has_next[id_col].sample(min(n_probes, has_next.height), seed=seed).to_list()

    future_value = pl.col(value_col).shift(-1).over(group_col)
    try:
        assert_past_only(_with_canary(build, future_value, id_col), df, time_col, id_col,
                         probe_ids=probes)  # fmt: skip
        value_caught = False
    except PastOnlyError:
        value_caught = True

    future_label = pl.col(label_col).shift(-1).over(group_col)
    try:
        assert_ignores(
            _with_canary(build, future_label, id_col), df, [label_col], id_col, seed=seed
        )
        label_caught = False
    except PastOnlyError:
        label_caught = True

    report = CanaryReport(honest_clean, value_caught, label_caught)
    if not (value_caught and label_caught):
        raise CanaryNotCaughtError(f"planted leak not detected: {report.to_dict()}")
    return report
