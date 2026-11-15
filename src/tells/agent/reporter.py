"""The Reporter: one Markdown document from an agent run, for a person to review.

Numbers come only from `tells.report` (code reading result files); the model's
narrative is included as written, after the Critic has confirmed it contains
no typed numbers. The document says it is a draft for human review.
"""

from __future__ import annotations

from tells.agent.critic import CriticReport, _parse_tail_json
from tells.agent.runner import AgentResult
from tells.agent.tools import Toolbox
from tells.report import render


def build_report(result: AgentResult, critic: CriticReport, toolbox: Toolbox) -> str:
    tail = _parse_tail_json(result.final_text) or {}
    prose = result.final_text
    if tail:
        prose = prose[: prose.rstrip().rfind("{")].rstrip()
    lines = [
        "# Agent report (draft for human review)",
        "",
        f"**Question:** {result.question}",
        "",
        f"**Critic:** {'all checks passed' if critic.passed else 'CHECKS FAILED, do not use'}.",
        "",
        "## Narrative (written by the model; numbers are not allowed here)",
        "",
        prose or "_The model returned no narrative._",
        "",
        "## Numbers (inserted by code from the result files)",
        "",
    ]
    for run_id in tail.get("runs", []):
        try:
            toolbox._summary(run_id)
        except Exception:  # noqa: BLE001 - a missing run is already a failed critic check
            lines += [f"_Run {run_id} could not be loaded._", ""]
            continue
        lines += [render(toolbox.runs_dir / run_id, with_history=False), ""]
    lines += ["## Critic checks", "", "| Check | Result | Detail |", "|---|---|---|"]
    for c in critic.checks:
        lines.append(f"| {c['check']} | {'pass' if c['ok'] else 'FAIL'} | {c['detail']} |")
    lines += ["", "## Tool calls", ""]
    for call in result.tool_calls:
        lines.append(f"- `{call['tool']}` {call['input']}")
    for d in result.denied:
        lines.append(f"- denied `{d['tool']}`: {d['reason']}")
    if result.stopped_by_budget:
        lines.append("- stopped by the tool-call budget")
    return "\n".join(lines) + "\n"
