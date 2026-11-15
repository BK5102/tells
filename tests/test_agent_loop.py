"""Agent loop, Critic, and Reporter with a hand-written cassette (no API key, no cost).

The cassette below is written by hand to mimic Claude's response shape. It is
not a recording of the real API; recorded cassettes are added once a key is
available (`CassetteLLM(mode="record", backend=AnthropicLLM())`).
"""

import json
from pathlib import Path

import pytest

from tells.agent.critic import review
from tells.agent.llm import CassetteLLM, request_digest
from tells.agent.reporter import build_report
from tells.agent.runner import SYSTEM, run_agent
from tells.agent.tools import Toolbox

RUN_ID = "20261115T020923Z_amlworld_valid_5152975d_4396a0"


def _summary() -> dict:
    budget = {"first_day_recall": 0.3, "recall": 0.4, "precision": 0.2}

    def contestant(h: float) -> dict:
        return {
            "headline": h,
            "cases": {"by_budget": {"0.005": budget}, "case_pr_auc": 0.2},
            "transactions": {"transaction_pr_auc": 0.4},
            "cost": {"fit_seconds": 1.0, "score_seconds": 1.0, "score_ms_per_1k_transactions": 1.0},
        }

    return {
        "run_id": RUN_ID,
        "config_name": "amlworld",
        "part": "valid",
        "headline": {"metric": "first_day_recall", "budget": 0.005},
        "canary": {"ok": True},
        "controls": {
            "label_permutation": {
                "permuted_headline": 0.006,
                "random_headline": 0.005,
                "difference": 0.001,
                "ci_low": -0.004,
                "ci_high": 0.006,
                "consistent_with_chance": True,
            }
        },
        "provenance": {"git": {"commit": "abc1234", "dirty": False}, "config_hash": "f" * 64},
        "contestants": {
            "B_checklist_lgbm": contestant(0.3),
            "D_checklist_tsfm_ttm": contestant(0.35),
        },
        "comparisons": {
            "reference": "B_checklist_lgbm",
            "results": {
                "D_checklist_tsfm_ttm": {
                    "diff": 0.05,
                    "ci_low": 0.01,
                    "ci_high": 0.09,
                    "p_holm": 0.02,
                },
            },
        },
    }


FINAL = (
    "The checklist-plus-TSFM model (D_checklist_tsfm_ttm) ranked newly bad accounts higher "
    f"than the bank-style model in run {RUN_ID}; see the headline table. This rests on one "
    "validation day and does not show test-period performance.\n"
    + json.dumps(
        {
            "runs": [RUN_ID],
            "claims": [
                {
                    "contestant": "D_checklist_tsfm_ttm",
                    "metric": "first_day_recall",
                    "run_id": RUN_ID,
                    "direction": "better",
                },
            ],
        }
    )
)


def _cassette(tmp_path: Path, final_text: str = FINAL) -> Path:
    responses = [
        {"stop_reason": "tool_use", "content": [
            {"type": "text", "text": "I will read the existing run first."},
            {"type": "tool_use", "id": "t1", "name": "load_results", "input": {"run_id": RUN_ID}},
        ]},
        {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "t2", "name": "read_file", "input": {"path": "data/raw"}},
        ]},
        {"stop_reason": "end_turn", "content": [{"type": "text", "text": final_text}]},
    ]  # fmt: skip
    path = tmp_path / "cassette.json"
    path.write_text(
        json.dumps([{"request_digest": "any", "response": r} for r in responses]), encoding="utf-8"
    )
    return path


@pytest.fixture()
def box(tmp_path: Path) -> Toolbox:
    configs = tmp_path / "configs"
    configs.mkdir()
    real = Path(__file__).parents[1] / "configs" / "amlworld_small.yaml"
    (configs / "amlworld.yaml").write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    run = tmp_path / "runs" / RUN_ID
    run.mkdir(parents=True)
    (run / "results.json").write_text(json.dumps(_summary()), encoding="utf-8")
    return Toolbox(configs_dir=configs, runs_dir=tmp_path / "runs")


def test_loop_executes_allowed_tools_and_denies_others(box: Toolbox, tmp_path: Path) -> None:
    result = run_agent("Does the TSFM help on AMLWorld?", box, CassetteLLM(_cassette(tmp_path)))
    assert [c["tool"] for c in result.tool_calls] == ["load_results", "read_file"]
    assert result.denied and "allow-list" in result.denied[0]["reason"]
    assert result.final_text == FINAL


def test_tool_budget_stops_the_loop(box: Toolbox, tmp_path: Path) -> None:
    result = run_agent("q", box, CassetteLLM(_cassette(tmp_path)), max_tool_calls=1)
    assert result.stopped_by_budget
    assert len(result.tool_calls) == 1


def test_critic_passes_a_good_answer(box: Toolbox) -> None:
    report = review(FINAL, box)
    assert report.passed, report.checks


def test_critic_rejects_typed_numbers(box: Toolbox) -> None:
    text = FINAL.replace("see the headline table", "it caught 35% of them")
    report = review(text, box)
    assert not report.passed
    assert any(c["check"] == "no typed numbers" and not c["ok"] for c in report.checks)


def test_critic_rejects_a_claim_the_interval_does_not_support(box: Toolbox) -> None:
    report = review(FINAL.replace('"better"', '"worse"'), box)
    assert not report.passed


def test_report_puts_numbers_from_files_next_to_the_narrative(box: Toolbox, tmp_path: Path) -> None:
    result = run_agent("Does the TSFM help?", box, CassetteLLM(_cassette(tmp_path)))
    text = build_report(result, review(result.final_text, box), box)
    assert "draft for human review" in text
    assert "35.0%" in text  # headline of D, formatted by code from results.json
    assert "denied `read_file`" in text


def test_replay_rejects_a_changed_prompt(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    recorded = request_digest(SYSTEM, [{"role": "user", "content": "original"}], [])
    path.write_text(
        json.dumps(
            [{"request_digest": recorded, "response": {"stop_reason": "end_turn", "content": []}}]
        ),
        encoding="utf-8",
    )
    llm = CassetteLLM(path)
    with pytest.raises(RuntimeError, match="differs from the recording"):
        llm.create(system=SYSTEM, messages=[{"role": "user", "content": "changed"}], tools=[])
