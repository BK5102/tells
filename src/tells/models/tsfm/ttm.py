"""TTM r2 adapter (ibm-granite/granite-timeseries-ttm-r2, Apache-2.0). Needs the `tsfm` extra.

TTM reads a fixed-length window (512 steps here), so shorter histories are
left-padded and the padding is marked as unobserved with `past_observed_mask`
so the model is told it is not real data.

Embedding: the backbone hidden state has shape (channels, patches, 192); the
patches are averaged per channel and concatenated: 3 x 192 = 576 numbers.
Forecast: the model's first predicted step for each forecast channel.

On CPU this costs about 2.3 to 2.9 ms per account-day, measured 2026-11-14,
about 25 to 100 times cheaper than Chronos-2.
"""

from __future__ import annotations

from importlib import metadata

import numpy as np

from tells.models.tsfm.base import CHANNEL_INDEX, FORECAST_CHANNELS, TSFMOutput

MODEL_ID = "ibm-granite/granite-timeseries-ttm-r2"
# Exact model weights (Hugging Face commit of the 512-96 branch, checked 2026-11-14),
# so a rerun can never silently pick up different weights.
REVISION = "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f"
CONTEXT = 512
PREDICTION = 96
D_MODEL = 192


class TTMAdapter:
    name = "ttm"

    def __init__(self, batch_size: int = 1024, threads: int | None = None):
        import torch
        from tsfm_public.toolkit.get_model import get_model

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.batch_size = batch_size
        self.revision = REVISION
        self._model = get_model(
            MODEL_ID,
            context_length=CONTEXT,
            prediction_length=PREDICTION,
            model_revision=REVISION,
        )
        self._model.eval()

    @property
    def version(self) -> str:
        return f"{MODEL_ID}@ctx{CONTEXT}@granite-tsfm-{metadata.version('granite-tsfm')}"

    @property
    def dim(self) -> int:
        return len(CHANNEL_INDEX) * D_MODEL

    def _run(self, seqs: list[np.ndarray]) -> TSFMOutput:
        torch = self._torch
        n_ch = len(CHANNEL_INDEX)
        emb = np.empty((len(seqs), self.dim), dtype=np.float32)
        fc = np.empty((len(seqs), len(FORECAST_CHANNELS)), dtype=np.float32)
        for start in range(0, len(seqs), self.batch_size):
            batch = seqs[start : start + self.batch_size]
            x = np.zeros((len(batch), CONTEXT, n_ch), dtype=np.float32)
            mask = np.zeros((len(batch), CONTEXT, n_ch), dtype=bool)
            for j, s in enumerate(batch):
                length = min(s.shape[1], CONTEXT)
                x[j, CONTEXT - length :, :] = s[:, -length:].T
                mask[j, CONTEXT - length :, :] = True
            with torch.no_grad():
                out = self._model(
                    past_values=torch.from_numpy(x), past_observed_mask=torch.from_numpy(mask)
                )
            hidden = out.backbone_hidden_state.float().numpy()  # (b, channels, patches, 192)
            emb[start : start + len(batch)] = hidden.mean(axis=2).reshape(len(batch), -1)
            pred = out.prediction_outputs.float().numpy()  # (b, 96, channels)
            fc[start : start + len(batch)] = pred[
                :, 0, [CHANNEL_INDEX[c] for c in FORECAST_CHANNELS]
            ]
        return TSFMOutput(emb, fc)
