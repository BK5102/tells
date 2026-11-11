"""Metric tests against hand-computed answers."""

import numpy as np
import pytest
from cleanroom.metrics import (
    average_precision,
    budget_selection,
    precision_at_budget,
    recall_at_budget,
)


def test_distinct_scores_take_the_top_k() -> None:
    scores = np.arange(10, 0, -1, dtype=float)  # 10, 9, ..., 1
    labels = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    # 20% of 10 cases = 2 slots: the two highest scores. One of two positives caught.
    assert budget_selection(scores, 0.2).tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert recall_at_budget(scores, labels, 0.2) == pytest.approx(0.5)
    assert precision_at_budget(scores, labels, 0.2) == pytest.approx(0.5)


def test_ties_at_the_cutoff_share_the_remaining_slots() -> None:
    scores = np.array([5.0, 5.0, 5.0, 1.0])
    labels = np.array([1, 0, 0, 0])
    # 50% of 4 = 2 slots spread evenly over the three tied cases.
    np.testing.assert_allclose(budget_selection(scores, 0.5), [2 / 3, 2 / 3, 2 / 3, 0])
    assert recall_at_budget(scores, labels, 0.5) == pytest.approx(2 / 3)


def test_fractional_capacity_is_counted_proportionally() -> None:
    scores = np.array([3.0, 2.0, 1.0])
    # 50% of 3 cases = 1.5 slots: all of the first, half of the second.
    np.testing.assert_allclose(budget_selection(scores, 0.5), [1, 0.5, 0])


def test_budget_applies_per_group() -> None:
    # Day "a" has every positive, but only gets its own 2 slots.
    scores = np.array([9, 8, 7, 1, 1, 1, 1, 1, 1, 1] + [0] * 10, dtype=float)
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0] + [0] * 10)
    groups = np.array(["a"] * 10 + ["b"] * 10)
    assert recall_at_budget(scores, labels, 0.2, groups) == pytest.approx(2 / 3)
    # Pooled over both days the 4 slots would catch all three.
    assert recall_at_budget(scores, labels, 0.2) == pytest.approx(1.0)


def test_constant_scores_catch_exactly_the_budget_share() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 1000)
    assert recall_at_budget(np.zeros(1000), labels, 0.05) == pytest.approx(0.05)


def test_perfect_scores_catch_up_to_capacity() -> None:
    labels = np.array([1] * 5 + [0] * 95)
    assert recall_at_budget(labels.astype(float), labels, 0.03) == pytest.approx(3 / 5)
    assert recall_at_budget(labels.astype(float), labels, 0.10) == pytest.approx(1.0)


def test_targets_count_only_selected_rows() -> None:
    # Recall over a subset of positives, e.g. only the first positive per entity.
    scores = np.array([4.0, 3.0, 2.0, 1.0])
    targets = np.array([0, 0, 1, 0])
    assert recall_at_budget(scores, targets, 0.5) == pytest.approx(0.0)
    assert recall_at_budget(scores, targets, 0.75) == pytest.approx(1.0)


def test_bad_inputs_rejected() -> None:
    with pytest.raises(ValueError):
        budget_selection(np.array([1.0, np.nan]), 0.5)
    with pytest.raises(ValueError):
        budget_selection(np.array([1.0]), 1.5)
    with pytest.raises(ValueError):
        recall_at_budget(np.array([1.0, 2.0]), np.array([0, 0]), 0.5)


def test_average_precision_hand_computed() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    labels = np.array([1, 0, 1, 0])
    # Positives at ranks 1 and 3: (1/1 + 2/3) / 2.
    assert average_precision(scores, labels) == pytest.approx((1 + 2 / 3) / 2)


def test_average_precision_with_ties() -> None:
    scores = np.array([0.9, 0.9, 0.1])
    labels = np.array([1, 0, 1])
    # Threshold 0.9: P=1/2, R=1/2. Threshold 0.1: P=2/3, R=1.
    assert average_precision(scores, labels) == pytest.approx(0.5 * 0.5 + 0.5 * 2 / 3)


def test_average_precision_of_random_scores_is_near_base_rate() -> None:
    rng = np.random.default_rng(1)
    labels = (rng.random(20000) < 0.1).astype(int)
    assert average_precision(rng.random(20000), labels) == pytest.approx(0.1, abs=0.01)
