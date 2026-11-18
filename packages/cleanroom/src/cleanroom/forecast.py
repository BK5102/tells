"""Scores for probabilistic forecasts: quantiles and prediction intervals.

A quantile forecast says "there is a 90% chance the value is below q". The
pinball loss scores one such statement, the weighted quantile loss averages
it over several quantiles and scales it so series of different sizes can be
compared, and the interval score rewards intervals that are narrow but still
contain the outcome. All losses here are lower-is-better.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cleanroom.stats import wilson_interval


def _float_array(x: npt.ArrayLike, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if np.isnan(a).any():
        raise ValueError(f"{name} contains NaN")
    return a


def _check_quantile(q: float) -> None:
    if not 0 < q < 1:
        raise ValueError(f"quantile must be in (0, 1), got {q}")


def _pinball(y: np.ndarray, q_pred: np.ndarray, quantile: float) -> np.ndarray:
    u = y - q_pred
    return np.maximum(quantile * u, (quantile - 1) * u)


def pinball_loss(y: npt.ArrayLike, q_pred: npt.ArrayLike, quantile: float) -> float:
    """Mean pinball (quantile) loss of predicting `q_pred` for the `quantile` of `y`.

    For one item, with u = y - q_pred: quantile * u when the outcome is above
    the prediction, (1 - quantile) * (-u) when it is below. At quantile 0.5 it
    is half the absolute error.
    """
    _check_quantile(quantile)
    ya = _float_array(y, "y")
    qa = _float_array(q_pred, "q_pred")
    if ya.shape != qa.shape or ya.size == 0:
        raise ValueError("y and q_pred must have the same, non-empty shape")
    return float(_pinball(ya, qa, quantile).mean())


def weighted_quantile_loss(
    y: npt.ArrayLike, q_preds: npt.ArrayLike, quantiles: Sequence[float]
) -> float:
    """Weighted quantile loss (WQL): pinball loss over several quantiles, scaled by |y|.

        WQL = (1/K) * sum_k [ 2 * sum_i pinball_{tau_k}(y_i, q_{i,k}) / sum_i |y_i| ]

    `q_preds` has the quantiles on its last axis, in the order of `quantiles`;
    `y` has the shape of `q_preds` without that axis (items may be any mix of
    series and time steps). This is the definition used in the Chronos paper
    (Ansari, A. F. et al. (2024). Chronos: Learning the language of time
    series. Transactions on Machine Learning Research; its evaluation metrics
    section, at the nine quantiles 0.1 to 0.9) and in GluonTS, whose `quantile_loss`
    already includes the factor 2 and whose `mean_wQuantileLoss` averages the
    scaled losses over quantiles (`gluonts.evaluation`). The factor 2 makes the
    0.5 term equal to the absolute error, so WQL with only the median is the
    weighted absolute percentage error.
    """
    taus = [float(t) for t in quantiles]
    if not taus:
        raise ValueError("quantiles must not be empty")
    for t in taus:
        _check_quantile(t)
    ya = _float_array(y, "y")
    qa = _float_array(q_preds, "q_preds")
    if qa.shape != (*ya.shape, len(taus)):
        raise ValueError(
            f"q_preds must have shape {(*ya.shape, len(taus))} (y's shape plus one axis "
            f"for the quantiles), got {qa.shape}"
        )
    scale = float(np.abs(ya).sum())
    if scale == 0:
        raise ValueError("WQL is undefined when every y is zero")
    per_quantile = [2 * _pinball(ya, qa[..., k], t).sum() / scale for k, t in enumerate(taus)]
    return float(np.mean(per_quantile))


@dataclass(frozen=True)
class Coverage:
    coverage: float  # share of outcomes inside [lower, upper], ends included
    n_inside: int
    n: int
    ci_low: float  # Wilson interval for the coverage
    ci_high: float
    confidence: float

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def _interval_inputs(
    y: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ya = _float_array(y, "y")
    lo = np.broadcast_to(_float_array(lower, "lower"), ya.shape)
    hi = np.broadcast_to(_float_array(upper, "upper"), ya.shape)
    if ya.size == 0:
        raise ValueError("y must not be empty")
    if (lo > hi).any():
        raise ValueError("every lower bound must be at or below its upper bound")
    return ya, lo, hi


def interval_coverage(
    y: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike, confidence: float = 0.95
) -> Coverage:
    """Share of outcomes inside their interval, with a Wilson interval for that share.

    A 90% interval should contain about 90% of outcomes. The Wilson interval
    says how far from the nominal level the observed share may be by chance,
    treating items as independent; outcomes from the same series are often
    correlated, so the true uncertainty can be larger.
    """
    ya, lo, hi = _interval_inputs(y, lower, upper)
    inside = int(((ya >= lo) & (ya <= hi)).sum())
    n = int(ya.size)
    ci_low, ci_high = wilson_interval(inside, n, confidence)
    return Coverage(inside / n, inside, n, ci_low, ci_high, confidence)


def interval_score(
    y: npt.ArrayLike, lower: npt.ArrayLike, upper: npt.ArrayLike, alpha: float
) -> float:
    """Mean interval score of central (1 - alpha) prediction intervals (lower is better).

        S = (upper - lower) + (2 / alpha) * (lower - y) * [y < lower]
                            + (2 / alpha) * (y - upper) * [y > upper]

    Width is penalized, and so is every unit by which an outcome falls
    outside, at a rate that grows as the promised coverage grows. Gneiting, T.
    and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
    estimation. Journal of the American Statistical Association 102(477),
    359-378, section 6.2 (they state it with the opposite sign, higher is
    better).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    ya, lo, hi = _interval_inputs(y, lower, upper)
    below = np.clip(lo - ya, 0, None)
    above = np.clip(ya - hi, 0, None)
    return float(((hi - lo) + 2 / alpha * (below + above)).mean())


def relative_error(model_loss: npt.ArrayLike, reference_loss: npt.ArrayLike) -> float:
    """Model loss relative to a reference forecaster's loss (below 1 means the model is better).

    With one pair this is the plain ratio. With several (for example one loss
    per dataset or per series), it is the geometric mean of the ratios, the
    aggregation used for relative scores in the Chronos paper (Ansari et al.
    2024) following Fleming, P. J. and Wallace, J. J. (1986). How not to lie
    with statistics: the correct way to summarize benchmark results.
    Communications of the ACM 29(3), 218-221. A geometric mean treats "twice
    as good" and "twice as bad" symmetrically, which an arithmetic mean of
    ratios does not.
    """
    m = np.atleast_1d(_float_array(model_loss, "model_loss"))
    r = np.atleast_1d(_float_array(reference_loss, "reference_loss"))
    if m.shape != r.shape or m.ndim != 1 or m.size == 0:
        raise ValueError("model_loss and reference_loss must be scalars or equal-length 1-D")
    if (r <= 0).any():
        raise ValueError("reference losses must be positive")
    if (m < 0).any():
        raise ValueError("model losses must not be negative")
    ratios = m / r
    if (ratios == 0).any():
        return 0.0
    return float(np.exp(np.log(ratios).mean()))
