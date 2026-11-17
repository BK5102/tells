"""MOMENT adapter (AutonLab/MOMENT-1-small, MIT; v0.2 idea 2, methodology M46). Needs the
`moment` extra.

MOMENT reads a fixed 512-step window per channel, cut into 64 patches of 8
steps, with an input mask that marks real steps. A patch only counts as real
when all 8 of its steps are real (momentfm `Masking.convert_seq_to_patch_view`);
other patches are replaced by a learned mask token and hidden from attention.
So a history is left-padded and, to keep its oldest events, the rest of its
oldest patch repeats the oldest real event and is marked real. Without that, a
history of fewer than 8 events (common on AMLWorld) would have no visible patch.

Embedding: `task_name="embedding"` with `reduction="none"` gives a hidden
state of shape (channels, 64 patches, 512). The library's default "mean"
reduction averages the channels together; here the real patches are averaged
per channel and the three channel averages concatenated, 3 x 512 = 1,536
numbers, matching how the TTM and Chronos-2 adapters keep channels apart.

Forecast: none. MOMENT's forecasting head is not pretrained (only the
reconstruction head is), so forecasts are NaN and the surprise features for
this model are all missing; the tree model then ignores them.
"""

from __future__ import annotations

from importlib import metadata

import numpy as np

from tells.models.tsfm.base import CHANNEL_INDEX, FORECAST_CHANNELS, TSFMOutput

MODEL_ID = "AutonLab/MOMENT-1-small"
# Exact model weights (Hugging Face commit, checked 2026-11-17).
REVISION = "411e288267f82cce86296dbe4d6c8bc533cc162f"
CONTEXT = 512
PATCH = 8
D_MODEL = 512


def moment_window(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Left-padded (batch, channels, 512) inputs and (batch, 512) masks of real steps.

    The oldest partial patch is completed by repeating the oldest real event, and
    those repeated steps are marked real so MOMENT does not discard the patch.
    """
    n_ch = len(CHANNEL_INDEX)
    x = np.zeros((len(seqs), n_ch, CONTEXT), dtype=np.float32)
    mask = np.zeros((len(seqs), CONTEXT), dtype=np.float32)
    for j, s in enumerate(seqs):
        length = min(s.shape[1], CONTEXT)
        if length == 0:
            continue
        shown = min(-(-length // PATCH) * PATCH, CONTEXT)  # round up to whole patches
        x[j, :, CONTEXT - length :] = s[:, -length:]
        x[j, :, CONTEXT - shown : CONTEXT - length] = s[:, -length, None]
        mask[j, CONTEXT - shown :] = 1.0
    return x, mask


class MomentAdapter:
    name = "moment"

    def __init__(self, batch_size: int = 256, threads: int | None = None):
        import torch
        from momentfm import MOMENTPipeline

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.batch_size = batch_size
        self.revision = REVISION
        self._model = MOMENTPipeline.from_pretrained(
            MODEL_ID, revision=REVISION, model_kwargs={"task_name": "embedding"}
        )
        self._model.init()
        self._model.eval()

    @property
    def version(self) -> str:
        pkg = metadata.version("momentfm")
        return f"{MODEL_ID}@ctx{CONTEXT}@pool-mean-per-channel@momentfm-{pkg}"

    @property
    def dim(self) -> int:
        return len(CHANNEL_INDEX) * D_MODEL

    def _embed(self, x, mask) -> np.ndarray:
        """Per-channel mean over real patches, channels concatenated: (batch, 3 x 512)."""
        with self._torch.no_grad():
            out = self._model(x_enc=x, input_mask=mask, reduction="none")
        hidden = out.embeddings.float().numpy()  # (b, channels, 64, 512)
        real = mask.numpy().reshape(len(mask), -1, PATCH).min(axis=2)  # 1 if all 8 steps real
        pooled = (hidden * real[:, None, :, None]).sum(axis=2) / real.sum(axis=1)[:, None, None]
        return pooled.reshape(len(mask), -1)

    def _run(self, seqs: list[np.ndarray]) -> TSFMOutput:
        torch = self._torch
        emb = np.empty((len(seqs), self.dim), dtype=np.float32)
        for start in range(0, len(seqs), self.batch_size):
            x, mask = moment_window(seqs[start : start + self.batch_size])
            emb[start : start + len(x)] = self._embed(torch.from_numpy(x), torch.from_numpy(mask))
        fc = np.full((len(seqs), len(FORECAST_CHANNELS)), np.nan, dtype=np.float32)
        return TSFMOutput(emb, fc)
