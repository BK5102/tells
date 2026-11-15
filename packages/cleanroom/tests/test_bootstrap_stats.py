import numpy as np
import pytest
from cleanroom.bootstrap import cluster_bootstrap_indices, paired_bootstrap
from cleanroom.stats import benjamini_hochberg, holm


def test_resampling_keeps_clusters_whole() -> None:
    clusters = np.array(["a", "a", "b", "c", "c", "c"])
    for idx in cluster_bootstrap_indices(clusters, n_boot=50, seed=0):
        picked = clusters[idx]
        # Every cluster appears a whole number of times, all its rows together.
        for c, size in (("a", 2), ("b", 1), ("c", 3)):
            assert (picked == c).sum() % size == 0
        assert len({*picked}) >= 1


def test_same_seed_same_resamples() -> None:
    clusters = np.repeat(np.arange(20), 3)
    a = [i.tolist() for i in cluster_bootstrap_indices(clusters, 5, seed=1)]
    b = [i.tolist() for i in cluster_bootstrap_indices(clusters, 5, seed=1)]
    assert a == b


def test_clear_difference_is_detected() -> None:
    rng = np.random.default_rng(0)
    n = 400
    clusters = np.arange(n)
    x = rng.normal(1.0, 1.0, n)  # contestant A per-row "score"
    y = x + 0.5  # B is always 0.5 better

    def stat(idx: np.ndarray) -> tuple[float, float]:
        return float(x[idx].mean()), float(y[idx].mean())

    r = paired_bootstrap(stat, clusters, n_boot=300, seed=0)
    assert r.diff == pytest.approx(0.5)
    assert r.ci_low == pytest.approx(0.5) and r.ci_high == pytest.approx(0.5)
    assert r.p_value < 0.01


def test_no_difference_gives_interval_around_zero() -> None:
    rng = np.random.default_rng(2)
    n = 500
    clusters = np.arange(n)
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)

    def stat(idx: np.ndarray) -> tuple[float, float]:
        return float(x[idx].mean()), float(y[idx].mean())

    r = paired_bootstrap(stat, clusters, n_boot=400, seed=0)
    assert r.ci_low < 0 < r.ci_high
    assert r.p_value > 0.05


def test_undefined_resamples_are_skipped_and_counted() -> None:
    clusters = np.arange(10)

    def stat(idx: np.ndarray) -> tuple[float, float]:
        if 0 not in idx:
            return float("nan"), float("nan")
        return 1.0, 2.0

    r = paired_bootstrap(stat, clusters, n_boot=100, seed=0)
    assert 0 < r.n_undefined < 100
    assert r.diff == 1.0


def test_holm_hand_computed() -> None:
    # Sorted 0.01, 0.03, 0.04 -> 0.03, 0.06, 0.04 -> running max 0.03, 0.06, 0.06.
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_bh_hand_computed() -> None:
    # Sorted 0.01, 0.03, 0.04 -> 0.03, 0.045, 0.04 -> running min from the top 0.03, 0.04, 0.04.
    np.testing.assert_allclose(benjamini_hochberg([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])


def test_corrections_cap_at_one_and_keep_single_test() -> None:
    assert holm([0.8, 0.9]).tolist() == [1.0, 1.0]
    assert benjamini_hochberg([0.2]).tolist() == [0.2]
    assert holm([0.2]).tolist() == [0.2]


def test_p_value_is_never_exactly_zero() -> None:
    clusters = np.arange(50)

    def stat(idx: np.ndarray) -> tuple[float, float]:
        return 0.0, 1.0

    r = paired_bootstrap(stat, clusters, n_boot=99, seed=0)
    assert r.p_value == pytest.approx(2 / 100)
