"""TSFM adapters. Real models need the optional `tsfm` extra; `fake` never does."""

from __future__ import annotations

from tells.models.tsfm.base import TSFMAdapter


def make_adapter(name: str, **kwargs) -> TSFMAdapter:
    if name == "fake":
        from tells.models.tsfm.fake import FakeTSFM

        return FakeTSFM()
    if name == "ttm":
        from tells.models.tsfm.ttm import TTMAdapter

        return TTMAdapter(**kwargs)
    if name == "chronos2":
        from tells.models.tsfm.chronos2 import Chronos2Adapter

        return Chronos2Adapter(**kwargs)
    raise ValueError(f"unknown TSFM {name!r}")
