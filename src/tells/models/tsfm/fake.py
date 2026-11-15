"""A tiny deterministic stand-in for a TSFM, for tests and CI (no PyTorch needed).

Its "embedding" is simple statistics of each channel and its "forecast" is the
channel mean. It lets every part of the TSFM pipeline (cache, features,
contestants C and D, leak checks) be tested quickly and for free.
"""

from __future__ import annotations

import numpy as np

from tells.models.tsfm.base import CHANNEL_INDEX, FORECAST_CHANNELS, TSFMOutput


class FakeTSFM:
    name = "fake"

    @property
    def version(self) -> str:
        return "fake-1"

    @property
    def dim(self) -> int:
        return 6

    def _run(self, seqs: list[np.ndarray]) -> TSFMOutput:
        emb = np.stack([np.concatenate([s.mean(axis=1), s[:, -1]]) for s in seqs]).astype(
            np.float32
        )
        fc = np.stack(
            [[s[CHANNEL_INDEX[c]].mean() for c in FORECAST_CHANNELS] for s in seqs]
        ).astype(np.float32)
        return TSFMOutput(emb, fc)
