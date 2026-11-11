"""The contestant interface and contestant A, the lazy baseline.

Every contestant learns from training rows and then gives each row it is
asked about a score where higher means more suspicious. Contestants never
see labels of the rows they score.
"""

from __future__ import annotations

from typing import Any, Protocol

import polars as pl


class Contestant(Protocol):
    name: str
    # True when scores are probabilities, so calibration metrics apply.
    outputs_probability: bool

    def fit(self, train: pl.DataFrame) -> None: ...

    def score(self, rows: pl.DataFrame) -> pl.Series: ...


class LazyAmount:
    """Contestant A: rank by transaction amount, biggest first. No learning at all.

    The walker in the race: any model worth its cost must beat this.
    """

    name = "A_lazy_amount"
    outputs_probability = False

    def fit(self, train: pl.DataFrame) -> None:
        return None

    def score(self, rows: pl.DataFrame) -> pl.Series:
        return rows["amount"].alias("score")


def make_contestant(key: str, model_config: dict[str, Any] | None = None) -> Contestant:
    """Build a contestant from its config key."""
    if key == "lazy_amount":
        return LazyAmount()
    raise ValueError(f"unknown contestant {key!r}")
