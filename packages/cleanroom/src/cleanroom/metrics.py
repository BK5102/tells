"""Ranking metrics for rare positives under a fixed review capacity.

`budget_selection` answers: if reviewers can open `budget` share of the items
in each group (for example each day), how much of each item gets reviewed?
Items are taken from the highest score down. When several items tie at the
cutoff, the remaining capacity is shared evenly among them, which equals the
expected result of breaking ties at random. Capacity may be fractional
(1% of 250 items is 2.5 slots) and is counted proportionally.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl

ArrayLike = npt.ArrayLike


def _as_scores(scores: ArrayLike) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    return s


def _as_labels(labels: ArrayLike, n: int) -> np.ndarray:
    y = np.asarray(labels, dtype=float)
    if y.shape != (n,):
        raise ValueError(f"labels must have shape ({n},), got {y.shape}")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    return y


def budget_selection(
    scores: ArrayLike, budget: float, groups: ArrayLike | None = None
) -> np.ndarray:
    """Share of each item reviewed (0 to 1) when each group gets `budget` x its size in slots."""
    s = _as_scores(scores)
    if not 0 < budget <= 1:
        raise ValueError(f"budget must be in (0, 1], got {budget}")
    g = np.zeros(len(s), dtype=np.int64) if groups is None else np.asarray(groups)
    if g.shape != s.shape:
        raise ValueError("groups must match scores")

    df = pl.DataFrame({"g": g, "s": s, "i": np.arange(len(s))})
    blocks = (
        df.group_by("g", "s")
        .agg(size=pl.len())
        .sort(["g", "s"], descending=[False, True])
        .with_columns(
            before=pl.col("size").cum_sum().over("g") - pl.col("size"),
            capacity=pl.col("size").sum().over("g") * budget,
        )
        .with_columns(
            share=(pl.col("capacity") - pl.col("before")).clip(0, pl.col("size")) / pl.col("size")
        )
    )
    out = df.join(blocks.select("g", "s", "share"), on=["g", "s"], how="left").sort("i")
    return out["share"].to_numpy()


def recall_at_budget(
    scores: ArrayLike, labels: ArrayLike, budget: float, groups: ArrayLike | None = None
) -> float:
    """Share of all positives that fall inside the review budget (expected under ties)."""
    share = budget_selection(scores, budget, groups)
    y = _as_labels(labels, len(share))
    if y.sum() == 0:
        raise ValueError("recall is undefined with no positives")
    return float((share * y).sum() / y.sum())


def precision_at_budget(
    scores: ArrayLike, labels: ArrayLike, budget: float, groups: ArrayLike | None = None
) -> float:
    """Share of reviewed items that are positive."""
    share = budget_selection(scores, budget, groups)
    y = _as_labels(labels, len(share))
    return float((share * y).sum() / share.sum())


def average_precision(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the precision-recall curve (step-wise, ties treated as one threshold).

    This is what "PR-AUC" means in this project. A model that ranks at random
    scores about the positive rate.
    """
    s = _as_scores(scores)
    y = _as_labels(labels, len(s))
    if y.sum() == 0:
        raise ValueError("average precision is undefined with no positives")
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    # Last index of each run of equal scores = one threshold.
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
