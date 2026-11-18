"""Statistical tests and corrections for comparing models honestly.

Multiple-comparison correction: test many ideas and some will look good by
chance. Holm controls the chance of even one false "win" across the whole
family of comparisons (strict). Benjamini-Hochberg controls the expected share
of false wins among the declared wins (more lenient). Both return adjusted
p-values in the original order, capped at 1.

`diebold_mariano` asks whether two forecasters' losses differ by more than
luck, when the losses form a time series and may be correlated over time.
`wilson_interval` is a confidence interval for a share (for example the share
of outcomes that fell inside a forecast interval) that stays sensible near 0
and 1 and for small counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats as _sps


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


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial share `successes / n`.

    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. Journal of the American Statistical Association
    22(158), 209-212. Unlike the plain "share plus or minus 1.96 standard
    errors", it never leaves [0, 1] and does not collapse to zero width when
    every outcome is a success or every one a failure.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be between 0 and n={n}, got {successes}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    z = float(_sps.norm.ppf(1 - (1 - confidence) / 2))
    p = successes / n
    z2n = z * z / n
    center = (p + z2n / 2) / (1 + z2n)
    half = z / (1 + z2n) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == n else min(1.0, center + half)
    return float(low), float(high)


@dataclass(frozen=True)
class DMResult:
    statistic: float  # positive: forecaster a has the larger loss, so b is better
    p_value: float  # two-sided, from a t distribution with n - 1 degrees of freedom
    mean_diff: float  # mean of loss_a - loss_b
    n: int
    horizon: int
    kernel: str

    def to_dict(self) -> dict[str, float | int | str]:
        return self.__dict__.copy()


def diebold_mariano(
    loss_a: npt.ArrayLike,
    loss_b: npt.ArrayLike,
    horizon: int = 1,
    kernel: Literal["uniform", "bartlett"] = "uniform",
) -> DMResult:
    """Diebold-Mariano test of equal expected loss, with the small-sample correction.

    Inputs are the per-period losses of two forecasts of the same target, in
    time order, made `horizon` steps ahead (for example squared or pinball
    errors). The loss difference d_t = loss_a_t - loss_b_t of h-step forecasts
    is usually correlated up to lag h - 1, so the long-run (HAC) variance of
    its mean uses autocovariances up to lag h - 1:

        V = gamma_0 + 2 * sum_{k=1}^{h-1} w_k * gamma_k,   DM = mean(d) / sqrt(V / n)

    with gamma_k = (1/n) * sum_t (d_t - mean d)(d_{t-k} - mean d).

    `kernel="uniform"` (default) uses w_k = 1, the choice in Diebold and
    Mariano (1995) and the one the Harvey, Leybourne and Newbold correction
    below was derived for. It can give a negative V on short series; that case
    raises ValueError. `kernel="bartlett"` uses Newey-West weights
    w_k = 1 - k / h (Newey, W. K. and West, K. D. (1987). Econometrica 55(3),
    703-708), which keep V positive but shrink the lag terms; with only h - 1
    lags that understates V when the loss differences are positively
    correlated, so the test rejects too often (in this package's tests, about
    9% of the time at a nominal 5% for h = 2 and 3, against about 5% for
    uniform weights). For horizon 1 the two are identical.

    The statistic is multiplied by the correction
    sqrt((n + 1 - 2h + h(h - 1) / n) / n) and compared with a t distribution
    with n - 1 degrees of freedom (Harvey, D., Leybourne, S. and Newbold, P.
    (1997). Testing the equality of prediction mean squared errors.
    International Journal of Forecasting 13(2), 281-291). The original test is
    Diebold, F. X. and Mariano, R. S. (1995). Comparing predictive accuracy.
    Journal of Business and Economic Statistics 13(3), 253-263.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("loss_a and loss_b must be one-dimensional and the same length")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("losses contain NaN")
    if not isinstance(horizon, int | np.integer) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    if kernel not in ("bartlett", "uniform"):
        raise ValueError(f"kernel must be 'bartlett' or 'uniform', got {kernel!r}")
    n = len(a)
    if n < 3 or horizon >= n:
        raise ValueError(f"need at least 3 periods and more periods than the horizon, got n={n}")

    d = a - b
    mean_diff = float(d.mean())
    dev = d - mean_diff
    v = float(dev @ dev) / n
    for k in range(1, horizon):
        weight = 1 - k / horizon if kernel == "bartlett" else 1.0
        v += 2 * weight * float(dev[k:] @ dev[:-k]) / n
    if v <= 0 or not np.isfinite(v):
        raise ValueError(
            f"estimated variance of the loss difference is {v:.3g}, not positive; the "
            "losses may be identical, or the series too short for this horizon "
            '(kernel="bartlett" always gives a positive variance)'
        )
    dm = mean_diff / np.sqrt(v / n)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = float(dm * correction)
    p = float(2 * _sps.t.sf(abs(stat), df=n - 1))
    return DMResult(
        statistic=stat,
        p_value=min(1.0, p),
        mean_diff=mean_diff,
        n=n,
        horizon=int(horizon),
        kernel=kernel,
    )
