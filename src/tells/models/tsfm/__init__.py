"""TSFM adapters. Real models need an optional extra (`tsfm`, or `moment`); `fake` never does."""

from __future__ import annotations

from tells.models.tsfm.base import TSFMAdapter

KNOWN_MODELS = ("fake", "ttm", "ttm_recent", "chronos2", "moment", "chronos_bolt")


def make_adapter(name: str, **kwargs) -> TSFMAdapter:
    if name == "fake":
        from tells.models.tsfm.fake import FakeTSFM

        return FakeTSFM()
    if name == "ttm":
        from tells.models.tsfm.ttm import TTMAdapter

        return TTMAdapter(**kwargs)
    if name == "ttm_recent":  # v0.2 idea 1: TTM, most recent patch only (methodology M45)
        from tells.models.tsfm.ttm_recent import TTMPooledAdapter

        return TTMPooledAdapter(pooling="last", **kwargs)
    if name == "chronos2":
        from tells.models.tsfm.chronos2 import Chronos2Adapter

        return Chronos2Adapter(**kwargs)
    if name == "moment":  # v0.2 idea 2 (methodology M46)
        from tells.models.tsfm.moment import MomentAdapter

        return MomentAdapter(**kwargs)
    if name == "chronos_bolt":  # v0.2: replaces MOMENT on CPU cost
        from tells.models.tsfm.chronos_bolt import ChronosBoltAdapter

        return ChronosBoltAdapter(**kwargs)
    raise ValueError(f"unknown TSFM {name!r}")
