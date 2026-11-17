"""Chronos-Bolt-tiny adapter (amazon/chronos-bolt-tiny, Apache-2.0, about 9M parameters).

Needs the `tsfm` extra. v0.2 uses it in place of MOMENT, which cost about
143 ms per account-day on CPU.

Chronos-Bolt is univariate, so each of the three channels of an account-day is
sent as its own series. The encoder returns one 256-number vector per 16-step
patch, then one [REG] token appended at the end (chronos/chronos_bolt.py,
`ChronosBoltModelForForecasting.encode`, lines 311-319 in chronos-forecasting
2.3.2). A batch is left-padded with NaN to a common length, and `Patch.forward`
(lines 80-92) left-pads again to a multiple of 16, so a series of L events owns
the last ceil(L / 16) patches before [REG]; the earlier patches are padding.
Like the Chronos-2 adapter, the history patches are averaged per channel and
the three channel averages are concatenated: 3 x 256 = 768 numbers.

Forecast: the median of the next step for log amount and log gap, from
`predict_quantiles(prediction_length=1, quantile_levels=[0.5])`. The
embedding is read from that same call's encoder pass, so each series is
encoded once.
"""

from __future__ import annotations

from importlib import metadata

import numpy as np

from tells.models.tsfm.base import CHANNEL_INDEX, FORECAST_CHANNELS, TSFMOutput

MODEL_ID = "amazon/chronos-bolt-tiny"
# Exact model weights (Hugging Face commit, checked 2026-11-17).
REVISION = "a0e552de83495b5c28c14c71c374f3e33280b340"
D_MODEL = 256
PATCH = 16  # input_patch_size in the model config


def history_patch_mean(emb: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Average each series' own patches, skipping padding patches and the final [REG] token.

    emb: (series, patches + 1, d) from `ChronosBoltPipeline.embed`; lengths: real steps per series.
    """
    n_patches = emb.shape[1] - 1
    out = np.empty((emb.shape[0], emb.shape[2]), dtype=np.float32)
    for i, length in enumerate(lengths):
        own = -(-int(length) // PATCH)  # ceil(length / PATCH)
        out[i] = emb[i, n_patches - own : n_patches].mean(axis=0)
    return out


class ChronosBoltAdapter:
    name = "chronos_bolt"

    def __init__(self, context: int = 256, batch_size: int = 256, threads: int | None = None):
        import torch
        from chronos import BaseChronosPipeline

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.context = context
        self.batch_size = batch_size  # account-days per model call (3 series each)
        self.revision = REVISION
        self._pipe = BaseChronosPipeline.from_pretrained(
            MODEL_ID, revision=REVISION, device_map="cpu"
        )

    @property
    def version(self) -> str:
        return f"{MODEL_ID}@chronos-forecasting-{metadata.version('chronos-forecasting')}"

    @property
    def dim(self) -> int:
        return len(CHANNEL_INDEX) * D_MODEL

    def _embed_and_forecast(self, series) -> tuple[np.ndarray, np.ndarray]:
        """Encoder states and median next step for each series, from one model pass.

        `predict_quantiles` runs the same `encode` that `embed` runs (chronos_bolt.py,
        `forward`, line 344), so a hook on the encoder returns exactly what `embed`
        would, without encoding every series a second time (about half the cost).
        """
        captured = []
        hook = self._pipe.model.encoder.register_forward_hook(
            lambda _m, _i, out: captured.append(out[0].detach())
        )
        try:
            q, _ = self._pipe.predict_quantiles(series, prediction_length=1, quantile_levels=[0.5])
        finally:
            hook.remove()
        (e,) = captured  # (series, patches + 1, 256); q: (series, 1, 1)
        return e.float().cpu().numpy(), q[:, 0, 0].float().numpy()

    def _run(self, seqs: list[np.ndarray]) -> TSFMOutput:
        torch = self._torch
        n_ch = len(CHANNEL_INDEX)
        fc_rows = [CHANNEL_INDEX[c] for c in FORECAST_CHANNELS]
        order = np.argsort([s.shape[1] for s in seqs], kind="stable")  # length buckets
        emb = np.empty((len(seqs), self.dim), dtype=np.float32)
        fc = np.empty((len(seqs), len(FORECAST_CHANNELS)), dtype=np.float32)
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            windows = [seqs[i][:, -self.context :] for i in idx]
            # One univariate series per (account-day, channel); the pipeline left-pads with NaN.
            series = [
                torch.from_numpy(np.ascontiguousarray(w[c])) for w in windows for c in range(n_ch)
            ]
            e, q = self._embed_and_forecast(series)
            lengths = np.array([w.shape[1] for w in windows for _ in range(n_ch)])
            pooled = history_patch_mean(e, lengths)
            emb[idx] = pooled.reshape(len(idx), n_ch * D_MODEL)
            fc[idx] = q.reshape(len(idx), n_ch)[:, fc_rows]
        return TSFMOutput(emb, fc)
