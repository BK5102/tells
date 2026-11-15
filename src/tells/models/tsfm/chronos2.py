"""Chronos-2 adapter (amazon/chronos-2, Apache-2.0). Needs the `tsfm` extra.

Embedding: Chronos-2 returns, per account-day, one 768-number vector per
16-step patch for each channel, followed by a [REG] token and a masked
future-patch token (order checked in chronos/chronos2/model.py). Following
Savvidi et al. 2026, the history patches are averaged per channel and the
three channel averages are concatenated: 3 x 768 = 2,304 numbers.

Forecast: the median of the next step for each forecast channel, from
`predict_quantiles(prediction_length=1)`.

On CPU this costs about 66 ms per AMLWorld account-day (short histories) and
about 280 ms per TabFormer account-day (256 events), measured 2026-11-14.
"""

from __future__ import annotations

from importlib import metadata

import numpy as np

from tells.models.tsfm.base import CHANNEL_INDEX, FORECAST_CHANNELS, TSFMOutput

MODEL_ID = "amazon/chronos-2"
# Exact model weights (Hugging Face commit, checked 2026-11-14).
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
D_MODEL = 768


class Chronos2Adapter:
    name = "chronos2"

    def __init__(self, context: int = 256, batch_size: int = 256, threads: int | None = None):
        import torch
        from chronos import Chronos2Pipeline

        if threads:
            torch.set_num_threads(threads)
        self.context = context
        self.batch_size = batch_size
        self.revision = REVISION
        self._pipe = Chronos2Pipeline.from_pretrained(MODEL_ID, revision=REVISION, device_map="cpu")

    @property
    def version(self) -> str:
        return f"{MODEL_ID}@chronos-forecasting-{metadata.version('chronos-forecasting')}"

    @property
    def dim(self) -> int:
        return len(CHANNEL_INDEX) * D_MODEL

    def _run(self, seqs: list[np.ndarray]) -> TSFMOutput:
        order = np.argsort([s.shape[1] for s in seqs], kind="stable")  # length buckets
        sorted_seqs = [seqs[i][:, -self.context :] for i in order]
        emb_list, _ = self._pipe.embed(sorted_seqs, batch_size=self.batch_size)
        quants, _ = self._pipe.predict_quantiles(
            sorted_seqs, prediction_length=1, quantile_levels=[0.5], batch_size=self.batch_size
        )
        emb = np.empty((len(seqs), self.dim), dtype=np.float32)
        fc = np.empty((len(seqs), len(FORECAST_CHANNELS)), dtype=np.float32)
        for pos, i in enumerate(order):
            e = emb_list[pos].float().numpy()  # (channels, patches + 2, 768)
            emb[i] = e[:, :-2, :].mean(axis=1).reshape(-1)
            q = quants[pos].float().numpy()  # (channels, 1, 1)
            fc[i] = [q[CHANNEL_INDEX[c], 0, 0] for c in FORECAST_CHANNELS]
        return TSFMOutput(emb, fc)
