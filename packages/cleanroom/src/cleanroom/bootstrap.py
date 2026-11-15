"""Paired cluster bootstrap: is one contestant's win bigger than luck?

Rows that belong to the same cluster (for example the same entity across
many days) are not independent, so the bootstrap resamples whole clusters
with replacement and keeps all of a cluster's rows together. "Paired" means
both contestants are scored on the very same resample each time, so the
interval is for their difference, which is what the question asks about.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

Stat = Callable[[np.ndarray], tuple[float, float]]


@dataclass(frozen=True)
class PairedResult:
    a: float  # statistic for the first contestant on the full data
    b: float  # statistic for the second contestant on the full data
    diff: float  # b - a
    ci_low: float
    ci_high: float
    p_value: float  # two-sided bootstrap p-value for "no difference"
    n_boot: int
    n_undefined: int  # resamples where the statistic could not be computed
    n_clusters: int
    confidence: float
    boot_mean_diff: float  # far from `diff` means the statistic is biased under resampling

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def cluster_bootstrap_indices(
    clusters: np.ndarray, n_boot: int, seed: int = 0
) -> Iterator[np.ndarray]:
    """Yield `n_boot` row-index arrays, each a resample of whole clusters."""
    clusters = np.asarray(clusters)
    order = np.argsort(clusters, kind="stable")
    _, starts, sizes = np.unique(clusters[order], return_index=True, return_counts=True)
    n_clusters = len(starts)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, n_clusters, n_clusters)
        lengths = sizes[pick]
        offsets = np.repeat(np.cumsum(lengths) - lengths, lengths)
        within = np.arange(lengths.sum()) - offsets
        yield order[np.repeat(starts[pick], lengths) + within]


def paired_bootstrap(
    stat: Stat,
    clusters: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> PairedResult:
    """Percentile interval and two-sided p-value for stat_b - stat_a.

    `stat(idx)` returns (value_a, value_b) computed on rows `idx`. A resample
    where either value is NaN (for example, no positives drawn) is skipped and
    counted in `n_undefined`.
    """
    clusters = np.asarray(clusters)
    a, b = stat(np.arange(len(clusters)))
    diffs = []
    undefined = 0
    for idx in cluster_bootstrap_indices(clusters, n_boot, seed):
        va, vb = stat(idx)
        if np.isnan(va) or np.isnan(vb):
            undefined += 1
            continue
        diffs.append(vb - va)
    d = np.asarray(diffs)
    if len(d) == 0:
        raise ValueError("every bootstrap resample was undefined")
    tail = (1 - confidence) / 2
    # (count + 1) / (n + 1): a finite number of resamples can never show p = 0.
    n = len(d)
    p = min(1.0, 2 * min((np.sum(d <= 0) + 1) / (n + 1), (np.sum(d >= 0) + 1) / (n + 1)))
    return PairedResult(
        a=float(a),
        b=float(b),
        diff=float(b - a),
        ci_low=float(np.quantile(d, tail)),
        ci_high=float(np.quantile(d, 1 - tail)),
        p_value=float(p),
        n_boot=n_boot,
        n_undefined=undefined,
        n_clusters=int(len(np.unique(clusters))),
        confidence=confidence,
        boot_mean_diff=float(d.mean()),
    )
