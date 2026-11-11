"""Time-ordered splits: learn from the past, leave a gap, evaluate on the future.

A `Split` is an ordered list of half-open time windows, train first and test
last, with an optional validation window between them. Consecutive windows
must be separated by at least `min_gap`. Rows that fall in a gap, or outside
every window, are dropped.

The split decides which rows are *evaluated* in each window. Inputs built
from history (for example rolling statistics) may still read earlier rows;
keeping those inputs past-only is a separate, separately tested guarantee.

Several splits (rolling origins, a named holdout period) are expressed as a
list of `Split` objects, so callers that handle one split handle many.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, TypeVar

import polars as pl

Frame = TypeVar("Frame", pl.DataFrame, pl.LazyFrame)


class SplitError(ValueError):
    """The split configuration itself is invalid."""


class LeakError(AssertionError):
    """Data assigned to the windows breaks the time order or the minimum gap."""


def _to_datetime(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise SplitError(f"cannot read {value!r} as a date")


@dataclass(frozen=True)
class Window:
    """Rows with start <= time < end. `start=None` means from the beginning of the data."""

    start: datetime | None
    end: datetime

    def __post_init__(self) -> None:
        if self.start is not None and self.start >= self.end:
            raise SplitError(f"window start {self.start} must be before end {self.end}")

    def mask(self, time_col: str) -> pl.Expr:
        inside = pl.col(time_col) < self.end
        if self.start is not None:
            inside = inside & (pl.col(time_col) >= self.start)
        return inside

    def to_dict(self) -> dict[str, str | None]:
        return {
            "start": None if self.start is None else self.start.isoformat(),
            "end": self.end.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Window:
        end = _to_datetime(d["end"])
        if end is None:
            raise SplitError("window end is required")
        return cls(_to_datetime(d.get("start")), end)


@dataclass(frozen=True)
class Split:
    train: Window
    test: Window
    valid: Window | None = None
    min_gap: timedelta = field(default=timedelta(0))

    def __post_init__(self) -> None:
        windows = self.windows()
        for name, w in windows[1:]:
            if w.start is None:
                raise SplitError(f"only the first window may be open-ended; {name} is not")
        for (prev_name, prev), (name, cur) in zip(windows, windows[1:], strict=False):
            assert cur.start is not None
            if prev.end > cur.start:
                raise SplitError(f"{prev_name} must end before {name} starts")
            if cur.start - prev.end < self.min_gap:
                raise SplitError(
                    f"gap between {prev_name} and {name} is {cur.start - prev.end}, "
                    f"smaller than min_gap {self.min_gap}"
                )

    def windows(self) -> list[tuple[str, Window]]:
        """Windows in time order, as (name, window) pairs."""
        out = [("train", self.train)]
        if self.valid is not None:
            out.append(("valid", self.valid))
        out.append(("test", self.test))
        return out

    def apply(self, df: Frame, time_col: str) -> dict[str, Frame]:
        """Assign rows to windows. Returns {"train": ..., ["valid": ...,] "test": ...}."""
        return {name: df.filter(w.mask(time_col)) for name, w in self.windows()}

    def check(self, parts: dict[str, pl.DataFrame], time_col: str) -> None:
        """Verify on the actual data that windows are time-ordered and gapped.

        Raises LeakError if any row of a later part is not at least `min_gap`
        after every row of an earlier part.
        """
        names = [n for n, _ in self.windows()]
        if list(parts) != names:
            raise LeakError(f"expected parts {names}, got {list(parts)}")
        bounds = {n: (p[time_col].min(), p[time_col].max()) for n, p in parts.items() if p.height}
        present = [n for n in names if n in bounds]
        for earlier, later in zip(present, present[1:], strict=False):
            latest_earlier = bounds[earlier][1]
            earliest_later = bounds[later][0]
            if not latest_earlier < earliest_later:
                raise LeakError(
                    f"{earlier} has a row at {latest_earlier}, not before {later}'s "
                    f"first row at {earliest_later}"
                )
            if earliest_later - latest_earlier < self.min_gap:
                raise LeakError(
                    f"observed gap between {earlier} and {later} is "
                    f"{earliest_later - latest_earlier}, smaller than {self.min_gap}"
                )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"min_gap_days": self.min_gap / timedelta(days=1)}
        for name, w in self.windows():
            d[name] = w.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Split:
        return cls(
            train=Window.from_dict(d["train"]),
            test=Window.from_dict(d["test"]),
            valid=Window.from_dict(d["valid"]) if d.get("valid") else None,
            min_gap=timedelta(days=d.get("min_gap_days", 0)),
        )
