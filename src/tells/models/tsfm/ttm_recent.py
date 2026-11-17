"""TTM r2 with a choice of pooling (v0.2 idea 1, methodology M45). Needs the `tsfm` extra.

TTM's backbone returns one 192-number hidden state per patch and channel. v0.1
averaged all patches ("mean"). Because the 512-step window is left-padded,
most patches of a short history are padding, and averaging blurs a recent
change in behavior into the account's whole past. "last" keeps only the
hidden state of the most recent patch, per channel, concatenated (3 x 192).

`ttm.py` is left byte-for-byte unchanged on purpose: v0.1's cache records a
checksum of that file and refuses to resume if it changes. This module reuses
TTMAdapter's input handling and wraps the model so its hidden state is cut to
the last patch before TTMAdapter averages it; the average of one patch is that
patch. pooling="mean" is therefore exactly v0.1 (same name, version, numbers).
"""

from __future__ import annotations

from tells.models.tsfm.ttm import REVISION, TTMAdapter

POOLINGS = ("mean", "last")


class _LastPatchOnly:
    """Calls the TTM model and keeps only the most recent patch of its hidden state."""

    def __init__(self, model):
        self.model = model

    def __call__(self, **kwargs):
        out = self.model(**kwargs)
        out.backbone_hidden_state = out.backbone_hidden_state[:, :, -1:, :]
        return out


class TTMPooledAdapter(TTMAdapter):
    def __init__(self, pooling: str = "last", **kwargs):
        _check(pooling)
        super().__init__(**kwargs)
        self._set_pooling(pooling)

    @classmethod
    def _from_model(cls, model, pooling: str = "last", batch_size: int = 1024):
        """Build around an already loaded model (used by tests; skips the download)."""
        _check(pooling)
        import importlib

        self = cls.__new__(cls)
        self._torch = importlib.import_module("torch") if model is not None else None
        self.batch_size = batch_size
        self.revision = REVISION
        self._model = model
        self._set_pooling(pooling)
        return self

    def _set_pooling(self, pooling: str) -> None:
        self.pooling = pooling
        # Feature names and the cache directory start with `name`, so the two
        # poolings can never share columns or cached embeddings.
        self.name = "ttm" if pooling == "mean" else f"ttm_{_SUFFIX[pooling]}"
        if pooling == "last" and self._model is not None:
            self._model = _LastPatchOnly(self._model)

    @property
    def version(self) -> str:
        base = super().version
        return base if self.pooling == "mean" else f"{base}@pool-{self.pooling}"


_SUFFIX = {"last": "recent"}


def _check(pooling: str) -> None:
    if pooling not in POOLINGS:
        raise ValueError(f"unknown TTM pooling {pooling!r}; use one of {POOLINGS}")
