"""v0.2 TSFM variants: TTM with recent-patch pooling, MOMENT, and models without forecasts.

Tests that need no model weights run everywhere; tests that load real weights
skip unless the optional extras are installed (`tsfm` for TTM, `moment` for MOMENT).
"""

from datetime import datetime
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from cleanroom.split import Split, Window

from tells.features.tsfm_features import (
    _cache_dir,
    add_tsfm_features,
    compute_tsfm_cache,
    feature_names,
    fit_pca,
    load_tsfm_table,
)
from tells.models.contestants import make_contestant
from tells.models.tsfm.base import TSFMOutput, run_adapter
from tells.models.tsfm.fake import FakeTSFM

SEQS = [
    np.zeros((3, 0), dtype=np.float32),  # an account's first day: no history
    np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 1.5], [1.0, 1.0, -1.0]], dtype=np.float32),
    np.random.default_rng(0).normal(size=(3, 40)).astype(np.float32),
]


def _quiet(_: str) -> None:
    return None


def _granite_version() -> str:
    try:
        return metadata.version("granite-tsfm")
    except metadata.PackageNotFoundError:
        pytest.skip("granite-tsfm not installed")


# ---------------------------------------------------------------- names and versions


def test_v01_ttm_version_string_is_unchanged() -> None:
    # The v0.1 cache directory is keyed by this exact string; changing it would
    # silently orphan the cached TTM embeddings that v0.1's results came from.
    from tells.models.tsfm.ttm import TTMAdapter

    ver = _granite_version()
    adapter = TTMAdapter.__new__(TTMAdapter)
    assert adapter.name == "ttm"
    assert adapter.version == f"ibm-granite/granite-timeseries-ttm-r2@ctx512@granite-tsfm-{ver}"


def test_ttm_recent_has_its_own_name_version_and_cache(tmp_path: Path) -> None:
    from tells.models.tsfm.ttm import TTMAdapter
    from tells.models.tsfm.ttm_recent import TTMPooledAdapter

    _granite_version()
    v01 = TTMAdapter.__new__(TTMAdapter)
    recent = TTMPooledAdapter._from_model(None, pooling="last")
    mean = TTMPooledAdapter._from_model(None, pooling="mean")
    assert recent.name == "ttm_recent"
    assert "pool-last" in recent.version and recent.version.startswith(v01.version)
    assert mean.name == "ttm" and mean.version == v01.version  # mean is v0.1 exactly
    assert _cache_dir(recent, "sha", 256, tmp_path) != _cache_dir(v01, "sha", 256, tmp_path)
    with pytest.raises(ValueError, match="pooling"):
        TTMPooledAdapter._from_model(None, pooling="max")


def test_new_feature_names_never_collide_with_v01() -> None:
    k = 64
    ttm, recent, moment = (set(feature_names(m, k)) for m in ("ttm", "ttm_recent", "moment"))
    assert not ttm & recent and not ttm & moment and not recent & moment
    assert all(n.startswith("ttm_recent_") for n in recent)
    for model in ("ttm_recent", "moment"):
        d = make_contestant(f"checklist_tsfm:{model}", {"tsfm": {"pca_components": k}})
        assert sum(n.startswith(f"{model}_pc") for n in d.numeric) == k  # k from config


def test_make_adapter_knows_the_new_models() -> None:
    import tells.models.tsfm as tsfm

    with pytest.raises(ValueError, match="unknown"):
        tsfm.make_adapter("nope")
    assert {"ttm", "ttm_recent", "chronos2", "moment"} <= set(tsfm.KNOWN_MODELS)


# ---------------------------------------------------------------- recent pooling


class _FakeTTMModel:
    """Returns a fixed hidden state of shape (batch, channels, patches, 192)."""

    def __init__(self, hidden):
        self.hidden = hidden

    def __call__(self, past_values, past_observed_mask):
        import torch

        b = past_values.shape[0]
        return SimpleNamespace(
            backbone_hidden_state=self.hidden[:b].clone(),
            prediction_outputs=torch.zeros(b, 96, 3),
        )


def test_last_pooling_differs_from_mean_and_uses_only_the_last_patch() -> None:
    torch = pytest.importorskip("torch")
    from tells.models.tsfm.ttm_recent import TTMPooledAdapter

    gen = torch.Generator().manual_seed(0)
    hidden = torch.randn(2, 3, 8, 192, generator=gen)
    seqs = SEQS[1:]
    last = TTMPooledAdapter._from_model(_FakeTTMModel(hidden), pooling="last")._run(seqs)
    mean = TTMPooledAdapter._from_model(_FakeTTMModel(hidden), pooling="mean")._run(seqs)
    np.testing.assert_allclose(last.embeddings, hidden[:, :, -1, :].reshape(2, -1).numpy())
    np.testing.assert_allclose(mean.embeddings, hidden.mean(dim=2).reshape(2, -1).numpy(), 1e-6)
    assert not np.allclose(last.embeddings, mean.embeddings)
    # Changing every earlier patch leaves the last-patch embedding untouched.
    other = hidden.clone()
    other[:, :, :-1, :] += 100.0
    again = TTMPooledAdapter._from_model(_FakeTTMModel(other), pooling="last")._run(seqs)
    np.testing.assert_array_equal(again.embeddings, last.embeddings)


def test_real_ttm_recent_is_the_last_patch_and_mean_matches_v01() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("tsfm_public")
    from tells.models.tsfm import make_adapter
    from tells.models.tsfm.ttm_recent import TTMPooledAdapter

    v01 = make_adapter("ttm")
    recent = make_adapter("ttm_recent")
    assert recent.name == "ttm_recent" and recent.dim == v01.dim
    mean = TTMPooledAdapter(pooling="mean")
    a, b, c = (run_adapter(m, SEQS) for m in (v01, mean, recent))
    np.testing.assert_array_equal(a.embeddings, b.embeddings)  # mean pooling is v0.1 exactly
    np.testing.assert_array_equal(a.forecasts, c.forecasts)  # pooling never touches forecasts
    assert np.isnan(c.embeddings[0]).all() and np.isfinite(c.embeddings[1:]).all()
    assert not np.allclose(a.embeddings[1:], c.embeddings[1:])


# ---------------------------------------------------------------- MOMENT


def test_moment_window_left_pads_and_masks() -> None:
    from tells.models.tsfm.moment import CONTEXT, PATCH, moment_window

    long = np.arange(3 * 600, dtype=np.float32).reshape(3, 600)
    x, mask = moment_window([SEQS[1], SEQS[2], long])
    assert x.shape == (3, 3, CONTEXT) and mask.shape == (3, CONTEXT)
    # 3 events: the most recent 3 steps are real, and the rest of their patch
    # repeats the earliest real event so MOMENT sees one whole patch.
    np.testing.assert_array_equal(x[0, :, -3:], SEQS[1])
    np.testing.assert_array_equal(x[0, :, -PATCH:-3], np.repeat(SEQS[1][:, :1], PATCH - 3, 1))
    assert mask[0, -PATCH:].all() and not mask[0, :-PATCH].any()
    assert (x[0, :, :-PATCH] == 0).all()
    # 40 events fill exactly 5 patches.
    np.testing.assert_array_equal(x[1, :, -40:], SEQS[2])
    assert mask[1].sum() == 40
    # Longer than the window: only the most recent 512 events.
    np.testing.assert_array_equal(x[2], long[:, -CONTEXT:])
    assert mask[2].all()


def _moment():
    pytest.importorskip("torch")
    pytest.importorskip("momentfm")
    from tells.models.tsfm import make_adapter

    return make_adapter("moment")


def test_real_moment_shapes_empty_rows_and_no_forecast() -> None:
    adapter = _moment()
    short = np.array([[2.0], [0.0], [1.0]], dtype=np.float32)  # one event
    seqs = [*SEQS, short]
    out = run_adapter(adapter, seqs)
    assert adapter.dim == 3 * 512  # MOMENT-1-small: d_model 512, one block per channel
    assert out.embeddings.shape == (4, adapter.dim) and out.forecasts.shape == (4, 2)
    assert np.isnan(out.embeddings[0]).all()  # empty history stays empty
    assert np.isfinite(out.embeddings[1:]).all()
    assert np.isnan(out.forecasts).all()  # MOMENT has no forecast in embedding mode
    again = run_adapter(adapter, seqs)
    np.testing.assert_allclose(out.embeddings, again.embeddings, rtol=1e-5, atol=1e-6)
    assert "MOMENT-1-small" in adapter.version and adapter.revision


def test_real_moment_ignores_masked_padding() -> None:
    import torch

    adapter = _moment()
    from tells.models.tsfm.moment import moment_window

    x, mask = moment_window(SEQS[1:])
    noisy = x.copy()
    noisy[mask[:, None, :].repeat(3, 1) == 0] = 99.0
    a = adapter._embed(torch.from_numpy(x), torch.from_numpy(mask))
    b = adapter._embed(torch.from_numpy(noisy), torch.from_numpy(mask))
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_real_moment_loads_every_backbone_weight() -> None:
    # huggingface_hub loads weights non-strictly, so a renamed layer in a newer
    # transformers would silently leave random weights. Check them all.
    adapter = _moment()
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from tells.models.tsfm.moment import MODEL_ID, REVISION

    saved = load_file(hf_hub_download(MODEL_ID, "model.safetensors", revision=REVISION))
    live = adapter._model.state_dict()
    unused = {k for k in saved if k not in live}
    assert unused <= {"head.linear.weight", "head.linear.bias"}  # reconstruction head only
    assert set(live) <= set(saved)
    for k, v in live.items():
        assert (saved[k] == v).all(), k


# ---------------------------------------------------------------- models without forecasts


class _NoForecastTSFM(FakeTSFM):
    name = "noforecast"

    def _run(self, seqs: list[np.ndarray]) -> TSFMOutput:
        out = super()._run(seqs)
        return TSFMOutput(out.embeddings, np.full_like(out.forecasts, np.nan))


def test_pipeline_accepts_a_model_without_forecasts(
    synthetic_tx: pl.DataFrame, tmp_path: Path
) -> None:
    from tells.features.checklist import build_checklist_features

    tx = synthetic_tx.join(build_checklist_features(synthetic_tx), on="tx_id", maintain_order="left")
    keys = tx.select("account_id", day=pl.col("ts").dt.date()).unique()
    cache = compute_tsfm_cache(_NoForecastTSFM(), tx, keys, "sha-n", 16, 500, tmp_path, _quiet)

    def all_days(days: pl.Series) -> np.ndarray:
        return np.ones(len(days), dtype=bool)

    mean, scale, comp, _ = fit_pca(cache, all_days, 2)
    table = load_tsfm_table(cache, "nf", mean, scale, comp, keys)
    out = add_tsfm_features(tx, table, "nf")
    assert out["nf_surprise_amount"].null_count() == out.height
    assert out["nf_surprise_gap"].null_count() == out.height
    has = out["nf_has_history"] == 1
    assert has.any() and out.filter(has)["nf_pc0"].null_count() == 0


def test_full_run_with_a_model_without_forecasts(
    synthetic_tx: pl.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tells.harness as harness
    from tells.config import ExperimentConfig

    monkeypatch.setattr(harness, "make_adapter", lambda name, **kw: _NoForecastTSFM())
    data = tmp_path / "syn.parquet"
    synthetic_tx.write_parquet(data)
    cfg = ExperimentConfig(
        name="nofc",
        dataset="synthetic",
        data=data,
        split=Split(
            train=Window(None, datetime(2015, 1, 1)),
            valid=Window(datetime(2015, 2, 1), datetime(2015, 8, 1)),
            test=Window(datetime(2015, 9, 1), datetime(2016, 1, 1)),
        ),
        contestants=("lgbm_checklist", "checklist_tsfm:nf"),
        model={
            "lgbm_checklist": {"n_estimators": 20, "min_child_samples": 5},
            "tsfm": {"context": 16, "pca_components": 3, "chunk": 500},
        },
        n_boot=20,
        raw={"name": "nofc"},
    )
    s = harness.run_experiment(
        cfg, runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache", tsfm_dir=tmp_path / "tsfm"
    ).summary
    gain = s["contestants"]["D_checklist_tsfm_nf"]["model"]["feature_gain"]
    assert gain["nf_surprise_amount"] == 0 and gain["nf_surprise_gap"] == 0
    assert "nf_pc0" in gain
