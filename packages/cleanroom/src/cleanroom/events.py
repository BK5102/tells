"""Scoring alarms against known event intervals, and setting an alarm threshold fairly.

A detector watches a score over time and raises alarms. The truth is a list
of intervals (start, end) during which an event was under way. Questions a
reader asks: how many events were caught, how long after they began, and how
many alarms were raised when nothing was happening.

Times are plain numbers on one scale (for example hours since the start);
convert datetimes before calling. Delays and tolerances use the same scale.

Alarm rule used throughout (`persistent_alarms`): the score must exceed the
threshold (strictly) on `persistence_k` consecutive samples. The alarm fires
once, on the k-th consecutive exceedance, and fires again only after the
score has dropped to or below the threshold and climbed back for another k
samples. Like a smoke detector that waits for several readings before it
sounds, and sounds once per fire rather than once per reading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _times(x: npt.ArrayLike, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.isnan(a).any():
        raise ValueError(f"{name} contains NaN")
    return a


@dataclass(frozen=True)
class AlarmMatch:
    detected: np.ndarray  # one bool per interval
    delay: np.ndarray  # first matching alarm minus interval start; NaN when missed
    alarm_matched: np.ndarray  # one bool per alarm: it fell in some interval's window

    @property
    def n_false_alarms(self) -> int:
        return int((~self.alarm_matched).sum())

    @property
    def detection_rate(self) -> float:
        return float(self.detected.mean()) if self.detected.size else float("nan")


def match_alarms(
    alarm_times: npt.ArrayLike, intervals: npt.ArrayLike, tolerance: float = 0.0
) -> AlarmMatch:
    """Match alarms to event intervals.

    An alarm at time a matches the interval (start, end) when
    start <= a <= end + tolerance. Alarms before an event starts never count
    for it (they could not have seen it); `tolerance` still counts an alarm
    that arrives shortly after the event ended. An interval is detected by its
    first matching alarm, and its delay is that alarm's time minus the start.
    Alarms that match no interval are false alarms. One alarm may match
    several overlapping intervals.
    """
    alarms = _times(alarm_times, "alarm_times")
    iv = (
        np.asarray(intervals, dtype=float).reshape(-1, 2)
        if np.size(intervals)
        else np.empty((0, 2))
    )
    if np.isnan(iv).any():
        raise ValueError("intervals contain NaN")
    if (iv[:, 0] > iv[:, 1]).any():
        raise ValueError("every interval must have start <= end")
    if tolerance < 0:
        raise ValueError(f"tolerance must not be negative, got {tolerance}")
    starts, stops = iv[:, 0], iv[:, 1] + tolerance

    sorted_alarms = np.sort(alarms)
    first = np.searchsorted(sorted_alarms, starts, side="left")
    has_next = first < len(sorted_alarms)
    padded = np.r_[sorted_alarms, np.inf]  # first == len(alarms) means "no later alarm"
    first_time = padded[first]
    detected = has_next & (first_time <= stops)
    delay = np.where(detected, first_time - starts, np.nan)

    # An alarm is matched if some interval starting at or before it is still open.
    order = np.argsort(starts, kind="stable")
    open_until = np.maximum.accumulate(stops[order]) if len(order) else np.empty(0)
    last_started = np.searchsorted(starts[order], alarms, side="right") - 1
    matched = np.zeros(len(alarms), dtype=bool)
    ok = last_started >= 0
    matched[ok] = open_until[last_started[ok]] >= alarms[ok]
    return AlarmMatch(detected=detected, delay=delay, alarm_matched=matched)


def caught_within(delays: npt.ArrayLike, horizon: float) -> float:
    """Share of events detected no later than `horizon` after they began.

    A missed event has delay NaN and is never caught.
    """
    d = np.asarray(delays, dtype=float)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("delays must be a non-empty one-dimensional array")
    if horizon < 0:
        raise ValueError(f"horizon must not be negative, got {horizon}")
    return float(np.mean(np.nan_to_num(d, nan=np.inf) <= horizon))


def false_alarms_per_unit(
    false_alarms: int | npt.ArrayLike, total_time: float, unit: float = 1.0
) -> float:
    """False alarms per `unit` of time (for example per 24 hours) over `total_time` watched.

    `false_alarms` is a count or the false alarm times themselves (for
    example `alarms[~match.alarm_matched]`).
    """
    count = int(false_alarms) if np.ndim(false_alarms) == 0 else int(np.size(false_alarms))
    if total_time <= 0 or unit <= 0:
        raise ValueError("total_time and unit must be positive")
    if count < 0:
        raise ValueError("the false alarm count must not be negative")
    return count / total_time * unit


def _group_order(n: int, groups: npt.ArrayLike | None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Row positions of each group, in input order within the group."""
    if groups is None:
        return np.arange(n), [np.arange(n)]
    g = np.asarray(groups)
    if g.shape != (n,):
        raise ValueError(f"groups must have shape ({n},)")
    order = np.argsort(g, kind="stable")
    _, starts = np.unique(g[order], return_index=True)
    return order, np.split(order, starts[1:])


def _rolling_min(s: np.ndarray, k: int) -> np.ndarray:
    """m[j] = min(s[j : j + k]), the minimum of the k samples ending at position j + k - 1."""
    return np.lib.stride_tricks.sliding_window_view(s, k).min(axis=1)


def persistent_alarms(
    scores: npt.ArrayLike,
    threshold: float,
    persistence_k: int = 1,
    groups: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Positions (into `scores`) where an alarm fires under the module's persistence rule.

    Rows must be in time order within each group; runs never cross groups.
    """
    s = _times(scores, "scores")
    k = _check_k(persistence_k)
    out = []
    for rows in _group_order(len(s), groups)[1]:
        if len(rows) < k:
            continue
        above = _rolling_min(s[rows], k) > threshold
        fires = above & ~np.r_[False, above[:-1]]
        out.append(rows[np.flatnonzero(fires) + k - 1])
    return np.sort(np.concatenate(out)) if out else np.empty(0, dtype=np.int64)


def _check_k(k: int) -> int:
    if not isinstance(k, int | np.integer) or k < 1:
        raise ValueError(f"persistence_k must be a positive integer, got {k!r}")
    return int(k)


@dataclass(frozen=True)
class AlarmThreshold:
    threshold: float  # alarm when score > threshold on persistence_k consecutive samples
    n_alarms: int  # alarms this threshold raises on the calibration window
    alarm_rate: float  # n_alarms per unit of time
    total_time: float
    unit: float
    persistence_k: int

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def threshold_for_alarm_rate(
    scores: npt.ArrayLike,
    times: npt.ArrayLike,
    target_rate_per_unit: float,
    persistence_k: int = 1,
    unit: float = 1.0,
    groups: npt.ArrayLike | None = None,
    total_time: float | None = None,
) -> AlarmThreshold:
    """Choose an alarm threshold on a calibration window so alarms stay within a target rate.

    Pass scores from a calibration window that comes before the evaluation
    window (and ideally holds no events), so the threshold is fixed before
    any evaluated outcome is seen. The threshold is always one of the
    calibration scores. Because one alarm fires per run of exceedances,
    raising the threshold can split one run into two and so raise the count;
    the rate is therefore not monotone in the threshold. The result is the
    lowest threshold at which the rate is at most the target for it and for
    every higher candidate, so small upward changes cannot break the target.

    `total_time` defaults to the summed span (last minus first time) of each
    group in the window; pass it explicitly when the window is known to be
    longer. `unit` converts the rate, for example unit=24 with hourly times
    gives alarms per day.
    """
    s = _times(scores, "scores")
    t = _times(times, "times")
    if s.shape != t.shape or s.size == 0:
        raise ValueError("scores and times must be non-empty and the same length")
    k = _check_k(persistence_k)
    if target_rate_per_unit < 0 or unit <= 0:
        raise ValueError("target_rate_per_unit must be >= 0 and unit > 0")

    lows, highs = [], []
    span = 0.0
    for rows in _group_order(len(s), groups)[1]:
        gt = t[rows]
        if (np.diff(gt) <= 0).any():
            raise ValueError("times must be strictly increasing within each group")
        span += float(gt[-1] - gt[0])
        if len(rows) < k:
            continue
        m = _rolling_min(s[rows], k)
        # The run that reaches k at window j fires for thresholds in [m[j-1], m[j]).
        lows.append(np.r_[-np.inf, m[:-1]])
        highs.append(m)
    total = span if total_time is None else float(total_time)
    if total <= 0:
        raise ValueError("the calibration window has zero length; pass total_time")

    candidates = np.unique(s)
    if lows:
        lo, hi = np.concatenate(lows), np.concatenate(highs)
        live = lo < hi
        lo, hi = np.sort(lo[live]), np.sort(hi[live])
        counts = np.searchsorted(lo, candidates, side="right") - np.searchsorted(
            hi, candidates, side="right"
        )
    else:
        counts = np.zeros(len(candidates), dtype=np.int64)
    rates = counts / total * unit
    # Highest candidate always gives zero alarms; walk down while the target holds.
    bad = np.flatnonzero(rates > target_rate_per_unit)
    pick = bad[-1] + 1 if len(bad) else 0
    return AlarmThreshold(
        threshold=float(candidates[pick]),
        n_alarms=int(counts[pick]),
        alarm_rate=float(rates[pick]),
        total_time=total,
        unit=float(unit),
        persistence_k=k,
    )
