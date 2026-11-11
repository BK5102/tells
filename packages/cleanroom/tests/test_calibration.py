import numpy as np
import pytest
from cleanroom.calibration import brier_score, expected_calibration_error, reliability_table


def test_brier_hand_computed() -> None:
    # ((0.9-1)^2 + (0.2-0)^2 + (0.5-1)^2 + (0-0)^2) / 4
    assert brier_score([0.9, 0.2, 0.5, 0.0], [1, 0, 1, 0]) == pytest.approx(
        (0.01 + 0.04 + 0.25 + 0.0) / 4
    )


def test_brier_rejects_non_probabilities() -> None:
    with pytest.raises(ValueError):
        brier_score([1.5, 0.2], [1, 0])


def test_reliability_table_uniform_bins() -> None:
    probs = np.array([0.05, 0.15, 0.15, 0.95])
    labels = np.array([0, 0, 1, 1])
    table = reliability_table(probs, labels, n_bins=10, strategy="uniform")
    rows = {r["bin"]: r for r in table.iter_rows(named=True)}
    assert rows[1]["count"] == 2
    assert rows[1]["mean_predicted"] == pytest.approx(0.15)
    assert rows[1]["observed_rate"] == pytest.approx(0.5)
    assert table["count"].sum() == 4
    assert rows[9]["lower"] < 0.95  # top bin includes values up to 1.0


def test_reliability_table_quantile_bins_have_equal_counts() -> None:
    rng = np.random.default_rng(0)
    probs = rng.random(1000)
    table = reliability_table(probs, (rng.random(1000) < probs).astype(int), n_bins=5)
    assert table["count"].to_list() == [200] * 5


def test_ece_is_zero_for_perfect_calibration_and_large_when_overconfident() -> None:
    probs = np.array([0.5] * 1000)
    labels = np.array([1, 0] * 500)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)
    # Says 90% but it happens 50% of the time.
    assert expected_calibration_error(np.full(1000, 0.9), labels) == pytest.approx(0.4)
