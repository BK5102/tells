import numpy as np
import pytest
from cleanroom.stats import diebold_mariano, wilson_interval
from scipy import stats

# Loss differences d = [1, -1, 2, 0, 3]: mean 1, deviations [0, -2, 1, -1, 2],
# gamma_0 = (0 + 4 + 1 + 1 + 4) / 5 = 2, gamma_1 = (0 - 2 - 1 - 2) / 5 = -1.
D = [1.0, -1.0, 2.0, 0.0, 3.0]
ZEROS = [0.0] * 5


def _t4_cdf(t: float) -> float:
    # Closed form of the t distribution CDF with 4 degrees of freedom.
    x = t / np.sqrt(1 + t * t / 4)
    return 0.5 + 3 / 8 * x * (1 - t * t / (12 * (1 + t * t / 4)))


def test_dm_horizon_one_hand_computed() -> None:
    # DM = 1 / sqrt(2 / 5); HLN factor sqrt((5 + 1 - 2) / 5) = sqrt(4 / 5); product sqrt(2).
    r = diebold_mariano(D, ZEROS, horizon=1)
    assert r.statistic == pytest.approx(np.sqrt(2))
    # p = 2 * (1 - F_4(sqrt 2)) = 1 - 4 / (3 sqrt 3).
    assert r.p_value == pytest.approx(1 - 4 / (3 * np.sqrt(3)))
    assert r.p_value == pytest.approx(2 * (1 - _t4_cdf(np.sqrt(2))))
    assert r.mean_diff == 1.0 and r.n == 5 and r.horizon == 1


def test_dm_horizon_two_bartlett_hand_computed() -> None:
    # Bartlett weight 1 - 1/2: V = 2 + 2 * 0.5 * (-1) = 1, DM = sqrt(5).
    # HLN factor sqrt((5 + 1 - 4 + 2/5) / 5) = sqrt(0.48); product sqrt(2.4).
    r = diebold_mariano(D, ZEROS, horizon=2, kernel="bartlett")
    assert r.statistic == pytest.approx(np.sqrt(2.4))
    assert r.p_value == pytest.approx(2 * (1 - _t4_cdf(np.sqrt(2.4))))


def test_dm_uniform_kernel_refuses_a_negative_variance() -> None:
    # Uniform weights: V = 2 + 2 * (-1) = 0, which is not a usable variance.
    with pytest.raises(ValueError, match="not positive"):
        diebold_mariano(D, ZEROS, horizon=2)


def test_dm_horizon_two_uniform_hand_computed() -> None:
    # d = [1, 2, 2, 3, 1, 0]: mean 1.5, deviations [-.5, .5, .5, 1.5, -.5, -1.5].
    # gamma_0 = 5.5 / 6, gamma_1 = (-.25 + .25 + .75 - .75 + .75) / 6 = .75 / 6,
    # V = 7 / 6, DM = 1.5 / sqrt(7 / 36) = 9 / sqrt(7).
    # HLN factor sqrt((6 + 1 - 4 + 2/6) / 6) = sqrt(5) / 3; product 3 * sqrt(5 / 7).
    r = diebold_mariano([1.0, 2.0, 2.0, 3.0, 1.0, 0.0], np.zeros(6), horizon=2)
    assert r.statistic == pytest.approx(3 * np.sqrt(5 / 7))
    assert r.p_value == pytest.approx(2 * stats.t.sf(3 * np.sqrt(5 / 7), df=5))


def test_dm_sign_and_symmetry() -> None:
    ab = diebold_mariano(D, ZEROS)
    ba = diebold_mariano(ZEROS, D)
    assert ab.statistic == pytest.approx(-ba.statistic)
    assert ab.p_value == pytest.approx(ba.p_value)
    assert ab.statistic > 0  # a has the larger loss


def test_dm_input_checks() -> None:
    with pytest.raises(ValueError, match="same length"):
        diebold_mariano([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="horizon"):
        diebold_mariano(D, ZEROS, horizon=0)
    with pytest.raises(ValueError, match="more periods"):
        diebold_mariano(D, ZEROS, horizon=5)
    with pytest.raises(ValueError, match="NaN"):
        diebold_mariano([1, np.nan, 3], [0, 0, 0])
    with pytest.raises(ValueError, match="not positive"):
        diebold_mariano([1, 1, 1, 1], [0, 0, 0, 0])  # constant difference
    with pytest.raises(ValueError, match="kernel"):
        diebold_mariano(D, ZEROS, kernel="parzen")  # type: ignore[arg-type]


def _null_rejection_rate(horizon: int, ma: list[float], n: int, reps: int, kernel: str) -> float:
    # Loss differences with mean zero (the null) and MA(len(ma)) correlation,
    # as h-step forecast errors have when h = len(ma) + 1.
    rng = np.random.default_rng(12345)
    q = len(ma)
    rejected = 0
    for _ in range(reps):
        e = rng.normal(size=n + q)
        d = e[q:].copy()
        for j, c in enumerate(ma, start=1):
            d += c * e[q - j : len(e) - j]
        rejected += diebold_mariano(d, np.zeros(n), horizon, kernel=kernel).p_value < 0.05
    return rejected / reps


def test_dm_size_close_to_nominal_one_step() -> None:
    rate = _null_rejection_rate(horizon=1, ma=[], n=50, reps=2000, kernel="uniform")
    # Binomial standard error at 5% over 2000 draws is about 0.5 points.
    assert 0.035 <= rate <= 0.065


def test_dm_size_close_to_nominal_three_steps() -> None:
    rate = _null_rejection_rate(horizon=3, ma=[0.5, 0.3], n=200, reps=2000, kernel="uniform")
    assert 0.035 <= rate <= 0.065


def test_dm_bartlett_with_h_minus_one_lags_over_rejects() -> None:
    # Documents why uniform weights are the default: Bartlett weights shrink
    # the lag terms, so with positively correlated differences the test
    # rejects a true null too often (about 9% at a nominal 5% here).
    rate = _null_rejection_rate(horizon=3, ma=[0.5, 0.3], n=200, reps=1000, kernel="bartlett")
    assert rate > 0.07


def test_dm_detects_a_real_difference() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 1.0, 200) ** 2
    b = 0.5 * a
    assert diebold_mariano(a, b).p_value < 0.001


def test_wilson_hand_computed() -> None:
    # 8 of 10 at 95%: z = 1.959964, z^2/n = 0.3841459,
    # center = (0.8 + 0.192073) / 1.384146 = 0.716744,
    # half = 1.959964 / 1.384146 * sqrt(0.016 + 0.009604) = 0.226582.
    low, high = wilson_interval(8, 10)
    assert low == pytest.approx(0.490162, abs=1e-6)
    assert high == pytest.approx(0.943318, abs=1e-6)


def test_wilson_edges_stay_in_unit_interval() -> None:
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and 0 < high < 0.3
    low, high = wilson_interval(10, 10)
    assert high == 1.0 and 0.7 < low < 1
    narrow = wilson_interval(500, 1000)
    wide = wilson_interval(5, 10)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]


def test_wilson_input_checks() -> None:
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(1, 10, confidence=1.0)
