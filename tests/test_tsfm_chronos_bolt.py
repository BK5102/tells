"""Chronos-Bolt-tiny adapter (v0.2, replaces MOMENT on cost).

Tests that need no model weights run everywhere; tests that load the real
model skip unless the `tsfm` extra (chronos-forecasting) is installed.
"""

from importlib import metadata

import numpy as np
import pytest

from tells.features.tsfm_features import feature_names
from tells.models.contestants import make_contestant
from tells.models.tsfm.base import run_adapter

SEQS = [
    np.zeros((3, 0), dtype=np.float32),  # an account's first day: no history
    np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 1.5], [1.0, 1.0, -1.0]], dtype=np.float32),
    np.random.default_rng(0).normal(size=(3, 40)).astype(np.float32),
]


# ---------------------------------------------------------------- no weights needed


def test_history_patch_mean_skips_padding_patches_and_the_reg_token() -> None:
    from tells.models.tsfm.chronos_bolt import PATCH, history_patch_mean

    # Batch padded to 40 steps: 3 patches of 16, then [REG]. A 40-step series
    # fills all 3 patches; a 3-step series fills only the last one.
    emb = np.zeros((2, 4, 5), dtype=np.float32)
    emb[:, :3, :] = np.arange(3, dtype=np.float32)[None, :, None]  # patch i holds value i
    emb[:, 3, :] = 100.0  # [REG] must never be averaged in
    out = history_patch_mean(emb, np.array([40, 3]))
    assert PATCH == 16
    np.testing.assert_allclose(out[0], np.full(5, 1.0))  # mean of patches 0, 1, 2
    np.testing.assert_allclose(out[1], np.full(5, 2.0))  # last patch only
    # A series of exactly 16 steps is one patch; 17 steps spill into a second.
    np.testing.assert_allclose(history_patch_mean(emb, np.array([16, 17]))[:, 0], [2.0, 1.5])


def test_chronos_bolt_feature_names_never_collide() -> None:
    k = 64
    names = {m: set(feature_names(m, k)) for m in ("ttm", "ttm_recent", "chronos2", "chronos_bolt")}
    bolt = names.pop("chronos_bolt")
    assert all(n.startswith("chronos_bolt_") for n in bolt)
    for other in names.values():
        assert not bolt & other
    d = make_contestant("checklist_tsfm:chronos_bolt", {"tsfm": {"pca_components": k}})
    assert d.name == "D_checklist_tsfm_chronos_bolt"
    assert sum(n.startswith("chronos_bolt_pc") for n in d.numeric) == k


def test_make_adapter_knows_chronos_bolt() -> None:
    import tells.models.tsfm as tsfm

    assert "chronos_bolt" in tsfm.KNOWN_MODELS


# ---------------------------------------------------------------- real model


@pytest.fixture(scope="module")
def bolt():
    pytest.importorskip("torch")
    pytest.importorskip("chronos")
    from tells.models.tsfm import make_adapter

    return make_adapter("chronos_bolt")


def test_real_chronos_bolt_shapes_empty_rows_and_determinism(bolt) -> None:
    out = run_adapter(bolt, SEQS)
    assert bolt.name == "chronos_bolt"
    assert bolt.dim == 3 * 256  # d_model 256, one block per channel
    assert out.embeddings.shape == (3, 768) and out.forecasts.shape == (3, 2)
    assert np.isnan(out.embeddings[0]).all() and np.isnan(out.forecasts[0]).all()
    assert np.isfinite(out.embeddings[1:]).all() and np.isfinite(out.forecasts[1:]).all()
    again = run_adapter(bolt, SEQS)
    np.testing.assert_allclose(out.embeddings, again.embeddings, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out.forecasts, again.forecasts, rtol=1e-5, atol=1e-6)


def test_real_chronos_bolt_version_and_revision(bolt) -> None:
    from tells.models.tsfm.chronos_bolt import MODEL_ID, REVISION

    assert MODEL_ID == "amazon/chronos-bolt-tiny"
    assert bolt.revision == REVISION == "a0e552de83495b5c28c14c71c374f3e33280b340"
    assert bolt.version == f"{MODEL_ID}@chronos-forecasting-{metadata.version('chronos-forecasting')}"


def test_real_chronos_bolt_padding_and_sorting_do_not_change_outputs(bolt) -> None:
    # Each history run alone must match the same history run in a mixed-length
    # batch (left NaN padding, length sorting, and unsorting are all invisible).
    rng = np.random.default_rng(1)
    seqs = [rng.normal(size=(3, n)).astype(np.float32) for n in (40, 1, 300, 17, 5)]
    batch = bolt._run(seqs)
    for i, s in enumerate(seqs):
        solo = bolt._run([s])
        np.testing.assert_allclose(batch.embeddings[i], solo.embeddings[0], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(batch.forecasts[i], solo.forecasts[0], rtol=1e-4, atol=1e-4)


def test_real_chronos_bolt_forecast_is_the_median_next_step(bolt) -> None:
    import torch

    from tells.models.tsfm.base import CHANNEL_INDEX, FORECAST_CHANNELS

    s = SEQS[2]
    out = bolt._run([s])
    for j, c in enumerate(FORECAST_CHANNELS):
        q, _ = bolt._pipe.predict_quantiles(
            [torch.from_numpy(s[CHANNEL_INDEX[c]])], prediction_length=1, quantile_levels=[0.5]
        )
        assert out.forecasts[0, j] == pytest.approx(float(q[0, 0, 0]), rel=1e-5, abs=1e-6)


def test_real_chronos_bolt_uses_only_the_last_context_events(bolt) -> None:
    # Events older than the context window must not change anything.
    rng = np.random.default_rng(2)
    recent = rng.normal(size=(3, bolt.context)).astype(np.float32)
    older = rng.normal(size=(3, 50)).astype(np.float32)
    a = bolt._run([recent])
    b = bolt._run([np.concatenate([older, recent], axis=1)])
    np.testing.assert_array_equal(a.embeddings, b.embeddings)
    np.testing.assert_array_equal(a.forecasts, b.forecasts)
