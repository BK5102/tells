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
    mean_a, _, comp_a, _ = fit_pca(a, train, 3)
    mean_b, _, comp_b, _ = fit_pca(b, train, 3)
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
    d_exp = s["contestants"]["D_checklist_tsfm_fake"]["explanation"]
    assert 0 <= d_exp["tsfm_share_of_attribution"] <= 1
    assert s["contestants"]["B_checklist_lgbm"]["explanation"]["tsfm_share_of_attribution"] == 0
    assert s["contestants"]["C_tsfm_only_fake"]["explanation"]["tsfm_share_of_attribution"] > 0.99
    assert all(len(e["top_reasons"]) == 3 for e in d_exp["examples"])


def test_cache_refuses_a_different_model_revision(
    synthetic_tx: pl.DataFrame, tmp_path: Path
) -> None:
    import pytest

    keys = synthetic_tx.select("account_id", day=pl.col("ts").dt.date()).unique().head(50)
    compute_tsfm_cache(FakeTSFM(), synthetic_tx, keys, "sha-r", 16, 1000, tmp_path, _quiet)
    other = FakeTSFM()
    other.revision = "different-weights"
    with pytest.raises(ValueError, match="revision"):
        compute_tsfm_cache(other, synthetic_tx, keys, "sha-r", 16, 1000, tmp_path, _quiet)


def test_report_numbers_come_from_the_results_file(
    synthetic_tx: pl.DataFrame, tmp_path: Path
) -> None:
    from tells.report import budget_table, cost_table, headline_table

    data = tmp_path / "syn.parquet"
    synthetic_tx.write_parquet(data)
    cfg = ExperimentConfig(
        name="rep",
        dataset="synthetic",
        data=data,
        split=Split(
            train=Window(None, datetime(2015, 1, 1)),
            valid=Window(datetime(2015, 2, 1), datetime(2015, 8, 1)),
            test=Window(datetime(2015, 9, 1), datetime(2016, 1, 1)),
        ),
        contestants=("lazy_amount", "lgbm_checklist", "checklist_tsfm:fake"),
        model={
            "lgbm_checklist": {"n_estimators": 20, "min_child_samples": 5},
            "tsfm": {"context": 16, "pca_components": 3, "chunk": 500},
        },
        n_boot=30,
        raw={"name": "rep"},
    )
    run = run_experiment(
        cfg, runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache", tsfm_dir=tmp_path / "tsfm"
    )
    s = run.summary
    text = headline_table(s) + budget_table(s) + cost_table(s)
    for name, c in s["contestants"].items():
        assert name in text
        assert f"{100 * c['headline']:.1f}%" in text  # the exact stored value, formatted by code


def test_standardized_pca_is_not_dominated_by_one_large_number(tmp_path: Path) -> None:
    from tells.features.tsfm_features import fit_pca

    rng = np.random.default_rng(0)
    part = tmp_path / "cache" / "part_00000"
    part.mkdir(parents=True)
    n = 2000
    emb = rng.normal(size=(n, 6))
    emb[:, 0] *= 500  # one number on a far larger scale, as seen in TTM's output
    np.save(part / "emb.npy", emb.astype(np.float16))
    np.save(part / "fc.npy", np.zeros((n, 2), dtype=np.float32))
    pl.DataFrame(
        {
            "account_id": [str(i) for i in range(n)],
            "day": [date(2020, 1, 1)] * n,
            "has_history": [1] * n,
        }
    ).with_columns(pl.col("has_history").cast(pl.Int8)).write_parquet(part / "keys.parquet")
    (part / "done").write_text("ok")

    def all_days(days: pl.Series) -> np.ndarray:
        return np.ones(len(days), dtype=bool)

    _, _, _, raw_share = fit_pca(tmp_path / "cache", all_days, 1, standardize=False)
    _, _, _, std_share = fit_pca(tmp_path / "cache", all_days, 1, standardize=True)
    assert raw_share > 0.99  # the large number takes everything
    assert std_share < 0.3  # after scaling, no single direction dominates


def test_cache_refuses_a_different_chunk_size(synthetic_tx: pl.DataFrame, tmp_path: Path) -> None:
    # Regression: resume goes by chunk index, so a new chunk size on an
    # existing cache would mix rows from two different slicings.
    import pytest

    keys = synthetic_tx.select("account_id", day=pl.col("ts").dt.date()).unique().head(60)
    compute_tsfm_cache(FakeTSFM(), synthetic_tx, keys, "sha-c", 16, 20, tmp_path, _quiet)
    with pytest.raises(ValueError, match="delete"):
        compute_tsfm_cache(FakeTSFM(), synthetic_tx, keys, "sha-c", 16, 30, tmp_path, _quiet)


def test_cache_refuses_a_different_set_of_keys(synthetic_tx: pl.DataFrame, tmp_path: Path) -> None:
    import pytest

    keys = synthetic_tx.select("account_id", day=pl.col("ts").dt.date()).unique().sort("day")
    compute_tsfm_cache(FakeTSFM(), synthetic_tx, keys.head(60), "sha-k", 16, 20, tmp_path, _quiet)
    with pytest.raises(ValueError, match="delete"):
        compute_tsfm_cache(
            FakeTSFM(), synthetic_tx, keys.slice(5, 60), "sha-k", 16, 20, tmp_path, _quiet
        )


def test_cache_accepts_an_old_meta_without_the_new_fields(
    synthetic_tx: pl.DataFrame, tmp_path: Path
) -> None:
    # Caches built before 2026-11-16 have no keys or adapter checksum; they must stay usable.
    import json

    keys = synthetic_tx.select("account_id", day=pl.col("ts").dt.date()).unique().head(60)
    out = compute_tsfm_cache(FakeTSFM(), synthetic_tx, keys, "sha-o", 16, 20, tmp_path, _quiet)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    for field in ("keys_sha256", "adapter_code_sha256"):
        meta.pop(field)
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    again = compute_tsfm_cache(FakeTSFM(), synthetic_tx, keys, "sha-o", 16, 20, tmp_path, _quiet)
    assert again == out


def test_loaded_table_must_be_unique_and_complete(
    synthetic_tx: pl.DataFrame, tmp_path: Path
) -> None:
    import shutil

    import pytest

    from tells.features.tsfm_features import load_tsfm_table

    keys = synthetic_tx.select("account_id", day=pl.col("ts").dt.date()).unique().head(60)
    out = compute_tsfm_cache(FakeTSFM(), synthetic_tx, keys, "sha-d", 16, 20, tmp_path, _quiet)

    def all_days(days: pl.Series) -> np.ndarray:
        return np.ones(len(days), dtype=bool)

    mean, scale, comp, _ = fit_pca(out, all_days, 2)
    assert load_tsfm_table(out, "fake", mean, scale, comp, keys).height == keys.height
    # A stray copy of a chunk duplicates its account-days.
    shutil.copytree(out / "part_00000", out / "part_00099")
    with pytest.raises(ValueError, match="duplicate"):
        load_tsfm_table(out, "fake", mean, scale, comp, keys)
    shutil.rmtree(out / "part_00099")
    shutil.rmtree(out / "part_00001")  # a missing chunk loses account-days
    with pytest.raises(ValueError, match="missing"):
        load_tsfm_table(out, "fake", mean, scale, comp, keys)
