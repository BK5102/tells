"""Calibration: when a model says 10%, does it happen about 10% of the time?

Only meaningful for scores that are probabilities. Ranking-only scores (for
example a raw amount) have no calibration to measure.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl


def _check(probs: npt.ArrayLike, labels: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("probs and labels must be one-dimensional and the same length")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("probs must be probabilities in [0, 1]")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    return p, y


def brier_score(probs: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Mean squared gap between predicted probability and outcome (lower is better)."""
    p, y = _check(probs, labels)
    return float(np.mean((p - y) ** 2))


def reliability_table(
    probs: npt.ArrayLike,
    labels: npt.ArrayLike,
    n_bins: int = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
) -> pl.DataFrame:
    """Bin predictions and compare the average prediction with the observed rate per bin.

    "quantile" bins hold about the same number of items, which suits rare
    positives where most predictions sit near zero. Columns: bin, lower,
    upper, count, mean_predicted, observed_rate.
    """
    p, y = _check(probs, labels)
    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    df = pl.DataFrame({"bin": bins, "p": p, "y": y})
    table = (
        df.group_by("bin")
        .agg(
            count=pl.len(),
            mean_predicted=pl.col("p").mean(),
            observed_rate=pl.col("y").mean(),
        )
        .sort("bin")
    )
    return table.with_columns(
        lower=pl.col("bin").map_elements(lambda b: float(edges[b]), return_dtype=pl.Float64),
        upper=pl.col("bin").map_elements(lambda b: float(edges[b + 1]), return_dtype=pl.Float64),
    ).select("bin", "lower", "upper", "count", "mean_predicted", "observed_rate")


def expected_calibration_error(
    probs: npt.ArrayLike, labels: npt.ArrayLike, n_bins: int = 10
) -> float:
    """Count-weighted average gap between predicted and observed rate over quantile bins."""
    table = reliability_table(probs, labels, n_bins=n_bins, strategy="quantile")
    gaps = (table["mean_predicted"] - table["observed_rate"]).abs()
    return float((gaps * table["count"]).sum() / table["count"].sum())
