import numpy as np
import pytest
from cleanroom.forecast import (
    interval_coverage,
    interval_score,
    pinball_loss,
    relative_error,
    weighted_quantile_loss,
)
from cleanroom.stats import wilson_interval

# --- pinball loss ----------------------------------------------------------------


def test_pinball_hand_computed() -> None:
    # u = y - q = [-1, 0, 1]; at 0.9: max(0.9u, -0.1u) = [0.1, 0, 0.9]; mean 1/3.
    assert pinball_loss([1, 2, 3], [2, 2, 2], 0.9) == pytest.approx(1 / 3)
    # At 0.1 the same errors cost [0.9, 0, 0.1]; mean 1/3 again by symmetry.
    assert pinball_loss([1, 2, 3], [2, 2, 2], 0.1) == pytest.approx(1 / 3)


def test_pinball_at_median_is_half_absolute_error() -> None:
    rng = np.random.default_rng(0)
    y, q = rng.normal(size=100), rng.normal(size=100)
    assert pinball_loss(y, q, 0.5) == pytest.approx(0.5 * np.abs(y - q).mean())


def test_pinball_is_minimized_by_the_true_quantile() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(size=20_000)
    q90 = np.quantile(y, 0.9)
    best = pinball_loss(y, np.full_like(y, q90), 0.9)
    for shift in (-0.3, 0.3):
        assert pinball_loss(y, np.full_like(y, q90 + shift), 0.9) > best


def test_pinball_input_checks() -> None:
    with pytest.raises(ValueError):
        pinball_loss([1, 2], [1, 2], 1.0)
    with pytest.raises(ValueError):
        pinball_loss([1, 2], [1], 0.5)
    with pytest.raises(ValueError):
        pinball_loss([1, np.nan], [1, 2], 0.5)


# --- weighted quantile loss ----------------------------------------------------------


def test_wql_hand_computed() -> None:
    # y = [1, 2, 3], sum |y| = 6.
    # 0.1: q = [0, 1, 2], u = [1, 1, 1], losses 0.1 each, sum 0.3.
    # 0.5: q = [1, 2, 4], u = [0, 0, -1], losses [0, 0, 0.5], sum 0.5.
    # 0.9: q = [2, 3, 4], u = [-1, -1, -1], losses 0.1 each, sum 0.3.
    # WQL = mean(2 * [0.3, 0.5, 0.3] / 6) = 1.1 / 9.
    y = [1, 2, 3]
    q = np.array([[0, 1, 2], [1, 2, 4], [2, 3, 4]], dtype=float).T
    assert weighted_quantile_loss(y, q, [0.1, 0.5, 0.9]) == pytest.approx(1.1 / 9)


def test_wql_with_only_the_median_is_wape() -> None:
    rng = np.random.default_rng(2)
    y = rng.uniform(1, 10, size=(4, 30))  # four series, thirty steps
    m = y + rng.normal(size=y.shape)
    wape = np.abs(y - m).sum() / np.abs(y).sum()
    assert weighted_quantile_loss(y, m[..., None], [0.5]) == pytest.approx(wape)


def test_wql_is_scale_free() -> None:
    rng = np.random.default_rng(3)
    y = rng.uniform(1, 10, 50)
    q = np.stack([y - 1, y, y + 1], axis=-1) + rng.normal(size=(50, 3))
    taus = [0.1, 0.5, 0.9]
    assert weighted_quantile_loss(1000 * y, 1000 * q, taus) == pytest.approx(
        weighted_quantile_loss(y, q, taus)
    )


def test_wql_input_checks() -> None:
    with pytest.raises(ValueError, match="shape"):
        weighted_quantile_loss([1, 2], [[1, 2]], [0.5])
    with pytest.raises(ValueError, match="zero"):
        weighted_quantile_loss([0, 0], [[0], [0]], [0.5])
    with pytest.raises(ValueError):
        weighted_quantile_loss([1, 2], np.ones((2, 0)), [])


# --- interval coverage and score -------------------------------------------------------


def test_coverage_hand_computed_ends_included() -> None:
    c = interval_coverage([0, 1, 2, 3, 4], 1, 3)
    assert (c.n_inside, c.n, c.coverage) == (3, 5, 0.6)
    assert (c.ci_low, c.ci_high) == pytest.approx(wilson_interval(3, 5))
    assert c.ci_low < 0.6 < c.ci_high


def test_coverage_of_a_true_90_percent_interval() -> None:
    rng = np.random.default_rng(4)
    y = rng.normal(size=5000)
    c = interval_coverage(y, -1.6448536, 1.6448536)
    assert c.ci_low < 0.9 < c.ci_high


def test_interval_score_hand_computed() -> None:
    # alpha 0.2, interval [1, 3]: width 2; y = 2 inside -> 2; y = 0 is 1 below
    # -> 2 + 10 * 1 = 12; y = 5 is 2 above -> 2 + 10 * 2 = 22. Mean 12.
    assert interval_score([2, 0, 5], [1, 1, 1], [3, 3, 3], alpha=0.2) == pytest.approx(12.0)


def test_interval_score_rewards_the_right_width() -> None:
    rng = np.random.default_rng(5)
    y = rng.normal(size=20_000)
    z = 1.6448536  # true 90% central interval of a standard normal
    right = interval_score(y, -z, z, alpha=0.1)
    assert interval_score(y, -0.5 * z, 0.5 * z, alpha=0.1) > right
    assert interval_score(y, -2 * z, 2 * z, alpha=0.1) > right


def test_interval_input_checks() -> None:
    with pytest.raises(ValueError, match="lower bound"):
        interval_coverage([1, 2], [3, 0], [2, 4])
    with pytest.raises(ValueError, match="alpha"):
        interval_score([1], [0], [2], alpha=0)


# --- relative error ---------------------------------------------------------------


def test_relative_error_hand_computed() -> None:
    assert relative_error(2.0, 4.0) == pytest.approx(0.5)
    # Ratios 0.5 and 4: geometric mean sqrt(2).
    assert relative_error([1, 4], [2, 1]) == pytest.approx(np.sqrt(2))
    # Twice as good and twice as bad cancel.
    assert relative_error([1, 4], [2, 2]) == pytest.approx(1.0)


def test_relative_error_input_checks() -> None:
    assert relative_error([0.0, 1.0], [1.0, 1.0]) == 0.0
    with pytest.raises(ValueError, match="positive"):
        relative_error(1.0, 0.0)
    with pytest.raises(ValueError):
        relative_error([1, 2], [1])
