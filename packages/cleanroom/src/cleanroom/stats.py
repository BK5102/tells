"""Multiple-comparison correction: test many ideas and some will look good by chance.

Holm controls the chance of even one false "win" across the whole family of
comparisons (strict). Benjamini-Hochberg controls the expected share of false
wins among the declared wins (more lenient). Both return adjusted p-values in
the original order, capped at 1.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _check(p_values: npt.ArrayLike) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or ((p < 0) | (p > 1)).any() or np.isnan(p).any():
        raise ValueError("p-values must be a one-dimensional array in [0, 1]")
    return p


def holm(p_values: npt.ArrayLike) -> np.ndarray:
    p = _check(p_values)
    m = len(p)
    order = np.argsort(p)
    stepped = (m - np.arange(m)) * p[order]
    adjusted = np.minimum(np.maximum.accumulate(stepped), 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out


def benjamini_hochberg(p_values: npt.ArrayLike) -> np.ndarray:
    p = _check(p_values)
    m = len(p)
    order = np.argsort(p)
    stepped = p[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum(np.minimum.accumulate(stepped[::-1])[::-1], 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out
