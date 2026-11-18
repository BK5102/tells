import numpy as np
import pytest
from cleanroom.events import (
    caught_within,
    false_alarms_per_unit,
    match_alarms,
    persistent_alarms,
    threshold_for_alarm_rate,
)

# --- matching alarms to intervals --------------------------------------------------


def test_match_alarms_hand_computed() -> None:
    # Events [10, 20], [30, 35], [50, 60]; alarms at 5, 12, 15, 37, 70.
    # Event 1: first alarm in [10, 22] is 12 -> delay 2.
    # Event 2: window [30, 37] with tolerance 2; alarm 37 -> delay 7.
    # Event 3: window [50, 62]; no alarm -> missed.
    # Alarms 5 (before any event) and 70 are false; 15 matches event 1.
    m = match_alarms([5, 12, 15, 37, 70], [(10, 20), (30, 35), (50, 60)], tolerance=2)
    assert m.detected.tolist() == [True, True, False]
    np.testing.assert_allclose(m.delay, [2, 7, np.nan])
    assert m.alarm_matched.tolist() == [False, True, True, True, False]
    assert m.n_false_alarms == 2
    assert m.detection_rate == pytest.approx(2 / 3)


def test_match_alarms_without_tolerance_and_unsorted_alarms() -> None:
    m = match_alarms([37, 12, 20], [(10, 20), (30, 35)])
    assert m.detected.tolist() == [True, False]
    assert m.delay[0] == 2
    assert m.alarm_matched.tolist() == [False, True, True]  # 20 is on the end, inclusive


def test_match_alarms_overlapping_intervals_share_an_alarm() -> None:
    m = match_alarms([25], [(0, 30), (20, 40)])
    assert m.detected.tolist() == [True, True]
    np.testing.assert_allclose(m.delay, [25, 5])
    assert m.alarm_matched.tolist() == [True]


def test_match_alarms_against_brute_force() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        alarms = rng.uniform(0, 100, rng.integers(0, 15))
        s = rng.uniform(0, 90, rng.integers(1, 6))
        iv = np.c_[s, s + rng.uniform(0, 10, len(s))]
        tol = float(rng.uniform(0, 3))
        m = match_alarms(alarms, iv, tolerance=tol)
        for i, (a, b) in enumerate(iv):
            hits = alarms[(alarms >= a) & (alarms <= b + tol)]
            assert m.detected[i] == bool(len(hits))
            if len(hits):
                assert m.delay[i] == pytest.approx(hits.min() - a)
        for j, x in enumerate(alarms):
            assert m.alarm_matched[j] == bool(((iv[:, 0] <= x) & (x <= iv[:, 1] + tol)).any())


def test_match_alarms_edge_cases() -> None:
    m = match_alarms([], [(1, 2)])
    assert m.detected.tolist() == [False] and np.isnan(m.delay[0])
    m = match_alarms([1.0, 2.0], [])
    assert m.alarm_matched.tolist() == [False, False]
    with pytest.raises(ValueError, match="start <= end"):
        match_alarms([1], [(5, 2)])
    with pytest.raises(ValueError, match="tolerance"):
        match_alarms([1], [(1, 2)], tolerance=-1)


# --- caught within and false alarm rate -------------------------------------------------


def test_caught_within_hand_computed() -> None:
    delays = [0, 2, 5, np.nan]
    assert caught_within(delays, 2) == pytest.approx(0.5)
    assert caught_within(delays, 5) == pytest.approx(0.75)
    assert caught_within(delays, 1e9) == pytest.approx(0.75)  # a miss is never caught


def test_false_alarms_per_unit_hand_computed() -> None:
    # 3 false alarms in 72 hours is 1 per 24 hours.
    assert false_alarms_per_unit(3, 72, unit=24) == pytest.approx(1.0)
    assert false_alarms_per_unit(np.array([1.0, 5.0, 9.0]), 72, unit=24) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        false_alarms_per_unit(1, 0)


# --- persistence rule and threshold choice ------------------------------------------------


def _brute_alarms(s: np.ndarray, threshold: float, k: int) -> list[int]:
    out, run = [], 0
    for i, v in enumerate(s):
        run = run + 1 if v > threshold else 0
        if run == k:
            out.append(i)
    return out


def test_persistent_alarms_hand_computed() -> None:
    s = [0, 5, 5, 5, 0, 5, 5, 0, 5, 5, 5, 5]
    # k = 2: runs of exceedances at 1-3, 5-6, 8-11 fire on their 2nd sample.
    assert persistent_alarms(s, 1, 2).tolist() == [2, 6, 9]
    # k = 3: the run at 5-6 is too short.
    assert persistent_alarms(s, 1, 3).tolist() == [3, 10]
    # k = 1: one alarm per run, at its start; equal to threshold does not exceed.
    assert persistent_alarms(s, 1, 1).tolist() == [1, 5, 8]
    assert persistent_alarms(s, 5, 1).tolist() == []


def test_persistent_alarms_runs_do_not_cross_groups() -> None:
    s = [5, 5, 5, 5]
    g = ["a", "a", "b", "b"]
    assert persistent_alarms(s, 1, 2).tolist() == [1]
    assert persistent_alarms(s, 1, 2, groups=g).tolist() == [1, 3]
    assert persistent_alarms(s, 1, 3, groups=g).tolist() == []


def test_threshold_hand_computed() -> None:
    # Times 0..9 (span 9). Scores below; k = 1.
    s = np.array([1, 3, 2, 3, 1, 4, 1, 1, 2, 1], dtype=float)
    t = np.arange(10, dtype=float)
    # Candidate counts: > 4: 0; > 3: 1 (at 5); > 2: 3 (1, 3, 5); > 1: 3 (runs 1-3, 5, 8).
    r = threshold_for_alarm_rate(s, t, target_rate_per_unit=1 / 9, persistence_k=1)
    assert (r.threshold, r.n_alarms) == (3.0, 1)
    assert r.alarm_rate == pytest.approx(1 / 9) and r.total_time == 9
    r = threshold_for_alarm_rate(s, t, target_rate_per_unit=3 / 9, persistence_k=1)
    assert (r.threshold, r.n_alarms) == (1.0, 3)
    r = threshold_for_alarm_rate(s, t, target_rate_per_unit=0)
    assert (r.threshold, r.n_alarms) == (4.0, 0)


def test_threshold_respects_a_non_monotone_rate() -> None:
    # k = 1. Threshold 1: one run 0-2 -> 1 alarm. Threshold 3: 5 > 3 at 0 and 2,
    # split by 4 -> 2 alarms. Threshold 5 (max): 0 alarms. With a target of one
    # alarm per window, 3 breaks it, so the answer must be 5, not 1.
    s = np.array([5, 4, 5, 1], dtype=float)
    r = threshold_for_alarm_rate(s, np.arange(4.0), target_rate_per_unit=1, total_time=1)
    assert r.threshold == 5.0 and r.n_alarms == 0


def test_threshold_with_persistence_and_unit() -> None:
    s = np.array([0, 5, 5, 0, 6, 0, 4, 4, 4, 0], dtype=float)
    t = np.arange(10, dtype=float) * 0.5  # half-hour samples, span 4.5 hours
    # k = 2: > 4: run 1-2 only -> 1 alarm; > 0: runs 1-2 and 6-8 -> 2 alarms.
    r = threshold_for_alarm_rate(s, t, target_rate_per_unit=1 / 4.5, persistence_k=2)
    assert (r.threshold, r.n_alarms) == (4.0, 1)
    r = threshold_for_alarm_rate(s, t, target_rate_per_unit=24 * 2 / 4.5, persistence_k=2, unit=24)
    assert (r.threshold, r.n_alarms) == (0.0, 2)
    assert r.alarm_rate == pytest.approx(24 * 2 / 4.5)


def test_threshold_matches_brute_force_and_the_alarm_rule() -> None:
    rng = np.random.default_rng(7)
    for _ in range(40):
        n = int(rng.integers(5, 60))
        s = np.round(rng.normal(size=n), 1)
        t = np.cumsum(rng.uniform(0.5, 1.5, n))
        k = int(rng.integers(1, 4))
        target = float(rng.uniform(0, 0.4))
        r = threshold_for_alarm_rate(s, t, target, persistence_k=k)
        span = t[-1] - t[0]
        cands = np.unique(s)
        rate = {c: len(_brute_alarms(s, c, k)) / span for c in cands}
        safe = [c for c in cands if all(rate[d] <= target for d in cands if d >= c)]
        assert r.threshold == min(safe)
        assert r.n_alarms == len(_brute_alarms(s, r.threshold, k))
        assert r.n_alarms == len(persistent_alarms(s, r.threshold, k))
        assert r.alarm_rate <= target


def test_threshold_with_groups_sums_spans() -> None:
    s = np.array([0, 5, 0, 0, 5, 0], dtype=float)
    t = np.array([0, 1, 2, 0, 1, 2], dtype=float)
    g = np.array([1, 1, 1, 2, 2, 2])
    r = threshold_for_alarm_rate(s, t, target_rate_per_unit=0.5, groups=g)
    assert r.total_time == 4 and r.n_alarms == 2 and r.threshold == 0.0
    with pytest.raises(ValueError, match="increasing"):
        threshold_for_alarm_rate(s, t, 0.5)  # without groups the times go backwards


def test_threshold_input_checks() -> None:
    with pytest.raises(ValueError):
        threshold_for_alarm_rate([1, 2], [0, 1], 0.1, persistence_k=0)
    with pytest.raises(ValueError, match="zero length"):
        threshold_for_alarm_rate([1], [0], 0.1)
    with pytest.raises(ValueError):
        threshold_for_alarm_rate([1, np.nan], [0, 1], 0.1)
