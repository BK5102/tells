import json
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest
from cleanroom.split import Split, Window

from tells.config import ExperimentConfig
from tells.harness import ScoringLockedError, run_experiment, split_data
from tells.models.contestants import LazyAmount

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "tabformer_tiny.csv"


def _config(tmp_path: Path) -> ExperimentConfig:
    from tells.data.tabformer import load_tabformer

    data = tmp_path / "tiny.parquet"
    load_tabformer(FIXTURE_CSV).write_parquet(data)
    split = Split(
        train=Window(None, datetime(2002, 9, 2)),
        valid=Window(datetime(2002, 9, 2), datetime(2002, 12, 1)),
        test=Window(datetime(2003, 1, 1), datetime(2004, 1, 1)),
    )
    return ExperimentConfig(
        name="tiny", dataset="tabformer", data=data, split=split, contestants=("lazy_amount",)
    )


def test_lazy_amount_ranks_by_amount() -> None:
    rows = pl.DataFrame({"amount": [5.0, 100.0, -3.0]})
    assert LazyAmount().score(rows).to_list() == [5.0, 100.0, -3.0]


def test_test_scoring_refused_before_lock(tmp_path: Path) -> None:
    with pytest.raises(ScoringLockedError):
        run_experiment(_config(tmp_path), part="test", runs_dir=tmp_path / "runs")


def test_split_hides_test_labels(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    parts = split_data(cfg, pl.read_parquet(cfg.data))
    assert "label" not in parts["test"].columns
    assert "label" in parts["train"].columns


def test_run_writes_results_and_cases(tmp_path: Path) -> None:
    result = run_experiment(
        _config(tmp_path), part="valid", runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache"
    )
    saved = json.loads((result.run_dir / "results.json").read_text(encoding="utf-8"))
    a = saved["contestants"]["A_lazy_amount"]
    assert a["transactions"]["n_transactions"] == 4
    assert a["cases"]["n_cases"] == 3
    assert set(a["cases"]["by_budget"]) == {"0.005", "0.01", "0.02", "0.05"}
    cases = pl.read_parquet(result.run_dir / "cases_A_lazy_amount.parquet")
    assert cases.height == 3
    assert "score" in cases.columns


def test_lgbm_run_records_provenance_index_and_cache(tmp_path: Path, synthetic_tx) -> None:
    data = tmp_path / "synthetic.parquet"
    synthetic_tx.write_parquet(data)
    cfg = ExperimentConfig(
        name="syn",
        dataset="synthetic",
        data=data,
        split=Split(
            train=Window(None, datetime(2015, 1, 1)),
            valid=Window(datetime(2015, 2, 1), datetime(2015, 8, 1)),
            test=Window(datetime(2015, 9, 1), datetime(2016, 1, 1)),
        ),
        contestants=("lazy_amount", "lgbm_checklist"),
        model={"lgbm_checklist": {"n_estimators": 30, "min_child_samples": 5}},
        raw={"name": "syn", "train_end": date(2015, 1, 1)},  # YAML gives dates
    )
    runs, cache = tmp_path / "runs", tmp_path / "cache"
    first = run_experiment(cfg, runs_dir=runs, cache_dir=cache)
    prov = first.summary["provenance"]
    assert len(prov["config_hash"]) == 64
    assert len(prov["data"]["transactions"]["sha256"]) == 64
    b = first.summary["contestants"]["B_checklist_lgbm"]
    assert b["model"]["library"] == "lightgbm"
    assert "calibration" in b["transactions"]
    assert len(list(cache.glob("checklist_*.parquet"))) == 1

    run_experiment(cfg, runs_dir=runs, cache_dir=cache)  # reuses the cache
    assert len(list(cache.glob("checklist_*.parquet"))) == 1
    lines = (runs / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert set(json.loads(lines[0])["recall_at_1pct"]) == {"A_lazy_amount", "B_checklist_lgbm"}


def test_two_runs_in_the_same_second_get_separate_folders(tmp_path: Path, monkeypatch) -> None:
    # Regression: CI runs fast enough that two runs can start in the same
    # second, and the run folder name used to be second-resolution only.
    import tells.harness as harness

    frozen = datetime(2026, 1, 1, 12, 0, 0, tzinfo=harness.UTC)
    monkeypatch.setattr(harness, "_now", lambda: frozen)
    cfg = _config(tmp_path)
    runs, cache = tmp_path / "runs", tmp_path / "cache"
    first = run_experiment(cfg, runs_dir=runs, cache_dir=cache)
    second = run_experiment(cfg, runs_dir=runs, cache_dir=cache)
    assert first.run_dir != second.run_dir
    assert (first.run_dir / "results.json").exists() and (second.run_dir / "results.json").exists()
