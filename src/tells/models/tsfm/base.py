"""One interface for every time series foundation model (TSFM).

An adapter takes account histories (one (n_channels, length) array per
account-day, from `tells.features.sequences`) and returns, for each:

- an embedding: a fixed-length summary of the history (methodology M32), and
- a forecast of the next event's channels (log amount, log gap), which the
  surprise score compares with what actually happened (methodology M33).

Empty histories (an account's first day) are not sent to the model; they get
NaN outputs, and `has_history = 0` tells the tree model why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tells.features.sequences import CHANNELS

FORECAST_CHANNELS = ("log_amount", "log_gap_minutes")


@dataclass(frozen=True)
class TSFMOutput:
    embeddings: np.ndarray  # (n, dim) float32, NaN rows for empty histories
    forecasts: np.ndarray  # (n, len(FORECAST_CHANNELS)) float32, NaN for empty histories


class TSFMAdapter(Protocol):
    name: str  # short id used in feature names and cache keys, e.g. "ttm"
    revision: str  # exact model weights (Hugging Face commit)

    @property
    def version(self) -> str: ...  # model id plus package version, recorded in provenance

    @property
    def dim(self) -> int: ...

    def _run(self, seqs: list[np.ndarray]) -> TSFMOutput: ...  # non-empty histories only


def run_adapter(adapter: TSFMAdapter, seqs: list[np.ndarray]) -> TSFMOutput:
    """Run the model on non-empty histories and put NaN rows back for empty ones."""
    n = len(seqs)
    emb = np.full((n, adapter.dim), np.nan, dtype=np.float32)
    fc = np.full((n, len(FORECAST_CHANNELS)), np.nan, dtype=np.float32)
    keep = [i for i, s in enumerate(seqs) if s.shape[1] > 0]
    if keep:
        out = adapter._run([seqs[i] for i in keep])
        emb[keep] = out.embeddings
        fc[keep] = out.forecasts
    return TSFMOutput(emb, fc)


CHANNEL_INDEX = {c: i for i, c in enumerate(CHANNELS)}
