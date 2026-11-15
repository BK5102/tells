"""TSFM pipeline with the fake model: leak tests, PCA isolation, cache, contestants C and D."""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
from cleanroom.pastonly import assert_ignores, assert_past_only
from cleanroom.split import Split, Window

from tells.config import ExperimentConfig
from tells.features.checklist import build_checklist_features
from tells.features.sequences import build_sequences, to_model_input
from tells.features.tsfm_features import add_tsfm_features, compute_tsfm_cache, fit_pca
from tells.harness import run_experiment
from tells.models.contestants import make_contestant, tsfm_models_needed
from tells.models.tsfm.base import FORECAST_CHANNELS, run_adapter
from tells.models.tsfm.fake import FakeTSFM


def _quiet(_: str) -> None:
    return None


def _raw_tsfm_features(tx: pl.DataFrame) -> pl.DataFrame:
    """Checklist + fake-TSFM features per transaction, without PCA (tested separately)."""
    tx = tx.join(build_checklist_features(tx), on="tx_id", maintain_order="left")
    keys = tx.select("account_id", day=pl.col("ts").dt.date()).unique()
    seqs = build_sequences(tx, keys, context=16)
    out = run_adapter(FakeTSFM(), to_model_input(seqs))
    forecasts = {
        f"fake_fc_{c}": pl.Series(out.forecasts[:, j]).fill_nan(None)
        for j, c in enumerate(FORECAST_CHANNELS)
    }
    table = seqs.select(
        "account_id", "day", fake_has_history=(pl.col("length") > 0).cast(pl.Int8)
    ).with_columns(fake_pc0=pl.Series(out.embeddings[:, 0]).fill_nan(None), **forecasts)
    feats = add_tsfm_features(tx, table, "fake")
    return feats.select(
        "tx_id", "fake_pc0", "fake_surprise_amount", "fake_surprise_gap", "fake_has_history"
    )


def test_tsfm_features_are_past_only(synthetic_tx: pl.DataFrame) -> None:
    assert_past_only(_raw_tsfm_features, synthetic_tx, "ts", "tx_id", n_probes=30)
    assert_ignores(_raw_tsfm_features, synthetic_tx, ["label"], "tx_id")


def test_pca_uses_training_days_only(synthetic_tx: pl.DataFrame, tmp_path: Path) -> None:
    keys = synthetic_tx.select("account_id", day=pl.col("ts").dt.date()).unique()

    def train(days: pl.Series) -> np.ndarray:
        return (days < date(2015, 1, 1)).to_numpy()

    a = compute_tsfm_cache(FakeTSFM(), synthetic_tx, keys, "sha-a", 16, 500, tmp_path, _quiet)
    # Change every later transaction wildly: the PCA must not move.
    later = pl.col("ts") >= datetime(2015, 1, 1)
    changed = synthetic_tx.with_columns(
        amount=pl.when(later).then(pl.col("amount") * 1000).otherwise(pl.col("amount"))
    )
    b = compute_tsfm_cache(FakeTSFM(), changed, keys, "sha-b", 16, 500, tmp_path, _quiet)
    mean_a, comp_a, _ = fit_pca(a, train, 3)
    mean_b, comp_b, _ = fit_pca(b, train, 3)
    np.testing.assert_allclose(mean_a, mean_b, rtol=1e-3)
    np.testing.assert_allclose(np.abs(comp_a), np.abs(comp_b), atol=1e-3)


def test_cache_resumes_and_is_keyed_by_data(synthetic_tx: pl.DataFrame, tmp_path: Path) -> None:
    keys = synthetic_tx.select("account_id", day=pl.col("ts").dt.date()).unique()
    logs: list[str] = []
    first = compute_tsfm_cache(
        FakeTSFM(), synthetic_tx, keys, "sha-x", 16, 1000, tmp_path, logs.append
    )
    n_chunks = len(logs)
    again = compute_tsfm_cache(
        FakeTSFM(), synthetic_tx, keys, "sha-x", 16, 1000, tmp_path, logs.append
    )
    assert again == first and len(logs) == n_chunks  # nothing recomputed
    other = compute_tsfm_cache(
        FakeTSFM(), synthetic_tx, keys, "sha-y", 16, 1000, tmp_path, logs.append
    )
    assert other != first


def test_contestant_keys() -> None:
    keys = (
        "lazy_amount",
        "lgbm_checklist",
        "tsfm_only:ttm",
        "checklist_tsfm:ttm",
        "checklist_tsfm:chronos2",
    )
    assert tsfm_models_needed(keys) == ["ttm", "chronos2"]
    c = make_contestant("tsfm_only:ttm", {"tsfm": {"pca_components": 4}})
    assert c.name == "C_tsfm_only_ttm" and len(c.numeric) == 4 + 3
    d = make_contestant("checklist_tsfm:ttm", {"lgbm_checklist": {"n_estimators": 9}})
    b = make_contestant("lgbm_checklist", {"lgbm_checklist": {"n_estimators": 9}})
    assert d.params == b.params  # same settings, so only the inputs differ
    assert set(b.numeric) < set(d.numeric)


def test_full_run_with_all_four_contestants(synthetic_tx: pl.DataFrame, tmp_path: Path) -> None:
    data = tmp_path / "syn.parquet"
    synthetic_tx.write_parquet(data)
    cfg = ExperimentConfig(
        name="syn4",
        dataset="synthetic",
        data=data,
        split=Split(
            train=Window(None, datetime(2015, 1, 1)),
            valid=Window(datetime(2015, 2, 1), datetime(2015, 8, 1)),
            test=Window(datetime(2015, 9, 1), datetime(2016, 1, 1)),
        ),
        contestants=("lazy_amount", "lgbm_checklist", "tsfm_only:fake", "checklist_tsfm:fake"),
        model={
            "lgbm_checklist": {"n_estimators": 30, "min_child_samples": 5},
            "tsfm": {"context": 16, "pca_components": 3, "chunk": 500},
        },
        n_boot=50,
        raw={"name": "syn4"},
    )
    s = run_experiment(
        cfg,
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        tsfm_dir=tmp_path / "tsfm",
    ).summary
    expected = {"A_lazy_amount", "B_checklist_lgbm", "C_tsfm_only_fake", "D_checklist_tsfm_fake"}
    assert set(s["contestants"]) == expected
    assert s["tsfm"]["fake"]["pca_components"] == 3
    assert set(s["comparisons"]["results"]) == expected - {"B_checklist_lgbm"}
    assert s["comparisons"]["n_comparisons"] == 3
    assert "fake_pc0" in s["contestants"]["D_checklist_tsfm_fake"]["model"]["feature_gain"]
