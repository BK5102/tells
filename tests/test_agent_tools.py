"""Guardrail tests for the agent's tool layer: every forbidden action is denied."""

import json
from pathlib import Path

import pytest

from tells.agent.tools import Toolbox, ToolDenied

RUN_ID = "20261115T020923Z_amlworld_valid_5152975d_4396a0"


def _summary() -> dict:
    budget = {"first_day_recall": 0.3, "recall": 0.4, "precision": 0.2}
    contestant = {
        "headline": 0.3,
        "cases": {"by_budget": {"0.005": budget}, "case_pr_auc": 0.2},
        "transactions": {"transaction_pr_auc": 0.4},
        "cost": {"fit_seconds": 1.0},
    }
    return {
        "run_id": RUN_ID,
        "config_name": "amlworld",
        "part": "valid",
        "headline": {"metric": "first_day_recall", "budget": 0.005},
        "contestants": {"B_checklist_lgbm": contestant},
    }


@pytest.fixture()
def box(tmp_path: Path) -> Toolbox:
    configs = tmp_path / "configs"
    configs.mkdir()
    real = Path(__file__).parents[1] / "configs" / "amlworld_small.yaml"
    (configs / "amlworld.yaml").write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    run = tmp_path / "runs" / RUN_ID
    run.mkdir(parents=True)
    (run / "results.json").write_text(json.dumps(_summary()), encoding="utf-8")
    (run / "cases_B_checklist_lgbm.parquet").write_bytes(b"rows with labels")
    return Toolbox(configs_dir=configs, runs_dir=tmp_path / "runs")


def test_only_three_tools_exist(box: Toolbox) -> None:
    assert [s["name"] for s in box.specs()] == ["run_experiment", "load_results", "plot"]
    for name in ("shell", "read_file", "fetch_url", "write_file"):
        with pytest.raises(ToolDenied, match="allow-list"):
            box.call(name, {})


def test_path_traversal_and_bad_ids_are_denied(box: Toolbox) -> None:
    for bad in ("../../data/raw", "x/../../etc", RUN_ID + "/../..", "cases_B_checklist_lgbm"):
        with pytest.raises(ToolDenied):
            box.call("load_results", {"run_id": bad})


def test_unknown_config_and_extra_arguments_are_denied(box: Toolbox) -> None:
    with pytest.raises(ToolDenied, match="unknown config"):
        box.call("run_experiment", {"config": "../secret"})
    with pytest.raises(ToolDenied, match="unexpected arguments"):
        box.call("load_results", {"run_id": RUN_ID, "path": "data/raw"})


def test_results_never_include_rows_or_labels(box: Toolbox) -> None:
    view = box.call("load_results", {"run_id": RUN_ID})
    text = json.dumps(view)
    assert "rows with labels" not in text and "parquet" not in text
    assert view["contestants"]["B_checklist_lgbm"]["headline"] == 0.3


def test_peek_at_test_labels_is_denied(box: Toolbox) -> None:
    # The real harness raises ScoringLockedError without a lock; the tool turns it into a denial.
    from tells.harness import ScoringLockedError

    def locked_runner(cfg, part):
        raise ScoringLockedError("no lock")

    box.runner = locked_runner
    with pytest.raises(ToolDenied, match="test scoring refused"):
        box.call("run_experiment", {"config": "amlworld", "part": "test"})


def test_plot_returns_chart_data_and_every_call_is_logged(box: Toolbox) -> None:
    chart = box.call("plot", {"run_id": RUN_ID, "metric": "recall"})
    assert chart["x"] == [0.005] and chart["series"]["B_checklist_lgbm"] == [0.4]
    with pytest.raises(ToolDenied):
        box.call("plot", {"run_id": RUN_ID, "metric": "label"})
    assert [c["tool"] for c in box.calls] == ["plot", "plot"]
