import json
from datetime import datetime
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
    result = run_experiment(_config(tmp_path), part="valid", runs_dir=tmp_path / "runs")
    saved = json.loads((result.run_dir / "results.json").read_text(encoding="utf-8"))
    assert saved["contestants"]["A_lazy_amount"]["n_transactions"] == 4
    cases = pl.read_parquet(result.run_dir / "cases_A_lazy_amount.parquet")
    assert cases.height == 3
    assert "score" in cases.columns
