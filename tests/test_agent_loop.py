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
            "metric": "first_day_recall",  # as written by compare_to_reference
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
    f"than the bank-style model in run {RUN_ID}; see the headline table. This rests on a single "
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


def _rewrite_run(box: Toolbox, change) -> None:
    path = box.runs_dir / RUN_ID / "results.json"
    s = json.loads(path.read_text(encoding="utf-8"))
    change(s)
    path.write_text(json.dumps(s), encoding="utf-8")


def test_narrative_is_cut_at_the_summary_line_not_inside_it(box: Toolbox, tmp_path: Path) -> None:
    # Regression: cutting at the last "{" landed inside the nested claims, so
    # the start of the JSON summary ended up in the narrative.
    from tells.agent.critic import split_tail

    prose, tail = split_tail(FINAL)
    assert prose == FINAL.splitlines()[0]
    assert tail is not None and tail["runs"] == [RUN_ID]
    result = run_agent("q", box, CassetteLLM(_cassette(tmp_path)))
    narrative = build_report(result, review(result.final_text, box), box).split("## Numbers")[0]
    assert '"claims"' not in narrative and '"runs"' not in narrative


def test_critic_allows_contestant_names_with_capitals_and_digits(box: Toolbox) -> None:
    # Regression: only lowercase names were stripped, so an honest answer
    # naming D_checklist_tsfm_chronos2 failed the typed-number check.
    def add_chronos(s: dict) -> None:
        s["contestants"]["D_checklist_tsfm_chronos2"] = s["contestants"]["D_checklist_tsfm_ttm"]

    _rewrite_run(box, add_chronos)
    text = FINAL.replace(
        "(D_checklist_tsfm_ttm)", "(D_checklist_tsfm_ttm, like D_checklist_tsfm_chronos2)"
    )
    report = review(text, box)
    assert report.passed, report.checks


@pytest.mark.parametrize(
    "phrase", ["about two thirds of them", "roughly Half", "it doubled", "a 35 point gain"]
)
def test_critic_rejects_spelled_out_and_typed_numbers(box: Toolbox, phrase: str) -> None:
    report = review(FINAL.replace("see the headline table", phrase), box)
    assert any(c["check"] == "no typed numbers" and not c["ok"] for c in report.checks)


def test_critic_rejects_a_claim_on_a_metric_the_run_did_not_test(box: Toolbox) -> None:
    report = review(FINAL.replace('"metric": "first_day_recall"', '"metric": "recall"'), box)
    failed = [c for c in report.checks if not c["ok"]]
    assert [c["check"] for c in failed] == [f"claim metric: D_checklist_tsfm_ttm in {RUN_ID}"]
    assert "only tested 'first_day_recall'" in failed[0]["detail"]


def test_critic_needs_the_holm_corrected_p_value_for_better(box: Toolbox) -> None:
    # The interval alone excludes zero here, but after correcting for the
    # number of comparisons the difference is not significant.
    def weaken(s: dict) -> None:
        s["comparisons"]["results"]["D_checklist_tsfm_ttm"]["p_holm"] = 0.2

    _rewrite_run(box, weaken)
    assert not review(FINAL, box).passed
    assert review(FINAL.replace('"better"', '"no clear difference"'), box).passed


def test_a_tool_that_crashes_is_reported_to_the_model(box: Toolbox, tmp_path: Path) -> None:
    # Regression: only ToolDenied was caught, so any other error ended the whole agent run.
    class Broken(Toolbox):
        def call(self, name, args):
            raise ValueError("bad results file")

    broken = Broken(configs_dir=box.configs_dir, runs_dir=box.runs_dir)
    llm = CassetteLLM(_cassette(tmp_path))
    result = run_agent("q", broken, llm)
    assert result.final_text == FINAL  # the loop carried on to the final answer
    assert result.errors[0] == {"tool": "load_results", "error": "ValueError: bad results file"}
    assert len(result.errors) == 2 and result.denied == []
