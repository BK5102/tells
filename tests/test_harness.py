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
        run_experiment(
            _config(tmp_path), part="test", runs_dir=tmp_path / "runs", locks_dir=tmp_path / "locks"
        )


def test_test_scoring_after_lock_is_recorded(tmp_path: Path) -> None:
    from cleanroom.lock import seal

    cfg = _config(tmp_path)
    cfg = ExperimentConfig(**{**cfg.__dict__, "raw": {"name": "tiny", "v": 1}})
    lock = seal(cfg.raw, tmp_path / "locks")
    # tmp_path is not a git repo, so the lock cannot be tracked; unit tests only.
    kw = dict(
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        locks_dir=tmp_path / "locks",
        allow_untracked_lock_for_tests=True,
    )
    first = run_experiment(cfg, part="test", **kw)
    second = run_experiment(cfg, part="test", **kw)
    assert first.summary["lock"]["test_run_number"] == 1
    assert second.summary["lock"]["test_run_number"] == 2
    # Each run records "started" before scoring and "finished" after.
    assert [u["status"] for u in lock.uses()] == ["started", "finished"] * 2
    assert first.summary["positives"]["test"] >= 0  # labels available only after the lock


def test_crashed_and_legacy_test_runs_still_count(tmp_path: Path) -> None:
    from cleanroom.lock import seal

    cfg = _config(tmp_path)
    cfg = ExperimentConfig(**{**cfg.__dict__, "raw": {"name": "tiny", "v": 2}})
    lock = seal(cfg.raw, tmp_path / "locks")
    lock.record_use("old-run", git_commit="abc")  # written before statuses existed
    lock.record_use("crashed-run", status="started", git_commit="abc")  # never finished
    run = run_experiment(
        cfg,
        part="test",
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        locks_dir=tmp_path / "locks",
        allow_untracked_lock_for_tests=True,
    )
    assert run.summary["lock"]["test_run_number"] == 3


def test_test_scoring_refused_with_an_untracked_lock_or_dirty_code(
    tmp_path: Path, monkeypatch
) -> None:
    # Regression: an uncommitted lock proves nothing about when the plan was fixed.
    from cleanroom.lock import seal

    import tells.harness as harness

    cfg = _config(tmp_path)
    cfg = ExperimentConfig(**{**cfg.__dict__, "raw": {"name": "tiny", "v": 3}})
    lock = seal(cfg.raw, tmp_path / "locks")
    kw = dict(runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache", locks_dir=lock.path.parent)
    monkeypatch.setattr(harness, "_git_state", lambda: {"commit": "a" * 40, "dirty": False})
    monkeypatch.setattr(harness, "_is_tracked", lambda path: False)
    with pytest.raises(ScoringLockedError, match="not committed"):
        run_experiment(cfg, part="test", **kw)
    monkeypatch.setattr(harness, "_is_tracked", lambda path: True)
    monkeypatch.setattr(harness, "_git_state", lambda: {"commit": "a" * 40, "dirty": True})
    with pytest.raises(ScoringLockedError, match="uncommitted"):
        run_experiment(cfg, part="test", **kw)
    monkeypatch.setattr(harness, "_git_state", lambda: {"commit": "unknown", "dirty": None})
    with pytest.raises(ScoringLockedError, match="uncommitted"):
        run_experiment(cfg, part="test", **kw)
    assert lock.uses() == []  # nothing was scored


def test_test_scoring_refused_when_the_data_changed_since_the_lock(tmp_path: Path) -> None:
    from cleanroom.lock import seal

    cfg = _config(tmp_path)
    cfg = ExperimentConfig(**{**cfg.__dict__, "raw": {"name": "tiny", "v": 4}})
    seal(cfg.raw, tmp_path / "locks", data_sha256="0" * 64)
    with pytest.raises(ScoringLockedError, match="data"):
        run_experiment(
            cfg,
            part="test",
            runs_dir=tmp_path / "runs",
            cache_dir=tmp_path / "cache",
            locks_dir=tmp_path / "locks",
            allow_untracked_lock_for_tests=True,
        )


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
    assert set(a["cases"]["by_budget"]) == {"0.001", "0.005", "0.01", "0.02", "0.05"}
    assert "headline" in a
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
        n_boot=50,
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
    assert first.summary["canary"]["ok"] is True
    comp = first.summary["comparisons"]
    assert comp["reference"] == "B_checklist_lgbm"
    a_vs_b = comp["results"]["A_lazy_amount"]
    assert a_vs_b["ci_low"] <= a_vs_b["diff"] <= a_vs_b["ci_high"]
    assert {"p_value", "p_holm", "p_bh"} <= set(a_vs_b)

    run_experiment(cfg, runs_dir=runs, cache_dir=cache)  # reuses the cache
    assert len(list(cache.glob("checklist_*.parquet"))) == 1
    lines = (runs / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert set(json.loads(lines[0])["headline_value"]) == {"A_lazy_amount", "B_checklist_lgbm"}


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


def test_label_permutation_control_is_consistent_with_chance(tmp_path: Path, synthetic_tx) -> None:
    data = tmp_path / "synthetic.parquet"
    synthetic_tx.write_parquet(data)
    cfg = ExperimentConfig(
        name="perm",
        dataset="synthetic",
        data=data,
        split=Split(
            train=Window(None, datetime(2015, 1, 1)),
            valid=Window(datetime(2015, 2, 1), datetime(2015, 8, 1)),
            test=Window(datetime(2015, 9, 1), datetime(2016, 1, 1)),
        ),
        contestants=("lgbm_checklist",),
        model={"lgbm_checklist": {"n_estimators": 30, "min_child_samples": 5}},
        headline_budget=0.05,
        budgets=(0.05,),
        n_boot=100,
        controls=("label_permutation",),
        raw={"name": "perm"},
    )
    s = run_experiment(cfg, runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache").summary
    control = s["controls"]["label_permutation"]
    real = s["contestants"]["B_checklist_lgbm"]["headline"]
    assert control["consistent_with_chance"] is True
    assert control["permuted_headline"] < real  # the real model learned something the noise did not


def test_report_history_never_writes_the_feature_cache(tmp_path: Path, synthetic_tx) -> None:
    # Regression: the report read a subset of columns and could save an
    # incomplete feature cache that later runs would reuse.
    from tells.report import history_table

    data = tmp_path / "synthetic.parquet"
    synthetic_tx.write_parquet(data)
    cfg = ExperimentConfig(
        name="hist",
        dataset="synthetic",
        data=data,
        split=Split(
            train=Window(None, datetime(2015, 1, 1)),
            valid=Window(datetime(2015, 2, 1), datetime(2015, 8, 1)),
            test=Window(datetime(2015, 9, 1), datetime(2016, 1, 1)),
        ),
        contestants=("lazy_amount",),
        n_boot=20,
        raw={"name": "hist", "data": str(data)},
    )
    run = run_experiment(cfg, runs_dir=tmp_path / "runs", cache_dir=tmp_path / "cache")
    s = run.summary
    empty = tmp_path / "report_cache"
    computed = history_table(run.run_dir, s, cache_dir=empty)
    assert not empty.exists()  # computed in memory, nothing written
    cached = history_table(run.run_dir, s, cache_dir=tmp_path / "cache")
    assert computed == cached  # the run's own cache gives the same table

    synthetic_tx.head(100).write_parquet(data)  # the data changed since the run
    with pytest.raises(ValueError, match="checksum"):
        history_table(run.run_dir, s, cache_dir=empty)


def test_canary_reads_no_rows_from_the_test_window(synthetic_tx, monkeypatch) -> None:
    # Regression: the canary chose accounts by label over the whole history,
    # so test-window labels shaped which accounts it used.
    import tells.harness as harness

    seen: list[pl.DataFrame] = []
    real = harness.run_canary

    def spy(builder, frame, **kw):
        seen.append(frame)
        return real(builder, frame, **kw)

    monkeypatch.setattr(harness, "run_canary", spy)
    cutoff = datetime(2015, 9, 1)
    # Positives only inside the test window must not make an account "positive".
    later_only = synthetic_tx.with_columns(
        label=pl.when(pl.col("ts") >= cutoff).then(1).otherwise(pl.col("label"))
    )
    harness.canary_check(later_only, seed=0, before=cutoff)
    assert seen[0]["ts"].max() < cutoff
    before = later_only.filter(pl.col("ts") < cutoff)
    positive_before = set(before.filter(pl.col("label") == 1)["account_id"])
    chosen = set(seen[0]["account_id"])
    assert len(chosen & positive_before) == harness.CANARY_ACCOUNTS // 2
