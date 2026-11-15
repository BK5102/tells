"""Planner and Runner: turn a research question into tool calls, under a hard budget.

The model plans and calls tools; this loop executes each call only through
the `Toolbox` allow-list, returns denials as errors the model can read, and
stops at a fixed number of tool calls. The model's final message must refer to
numbers by run id and metric name only; the Critic checks that (methodology M40).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tells.agent.llm import LLM
from tells.agent.tools import Toolbox, ToolDenied

SYSTEM = """You are the lab assistant for Tells, an evaluation of whether time series \
foundation models (TSFMs) help a bank's fraud model catch newly bad accounts at a fixed \
daily review budget. You do not decide who is fraudulent; you run experiments and \
summarize them.

You can only use three tools: run_experiment (a config by name, validation period unless \
the user explicitly asks for the sealed test period), load_results (summary numbers of a \
finished run), and plot (chart data). Prefer load_results on existing runs before \
starting new ones, because runs take minutes to hours.

Rules for your final answer:
- Never write a number yourself. Refer to results only by run id and metric name, for \
example "headline of D_checklist_tsfm_ttm in run <run_id>". A report generator inserts \
every number from the result files.
- Say what the evidence does and does not show, including results where the TSFM did not \
help, and name anything you could not test.
- End with a JSON object on its own line: {"runs": [<run ids you relied on>], \
"claims": [{"contestant": ..., "metric": ..., "run_id": ..., "direction": \
"better"|"worse"|"no clear difference"}]}"""


@dataclass
class AgentResult:
    question: str
    final_text: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    denied: list[dict[str, Any]] = field(default_factory=list)
    stopped_by_budget: bool = False
    turns: int = 0


def run_agent(
    question: str, toolbox: Toolbox, llm: LLM, max_tool_calls: int = 10, max_turns: int = 12
) -> AgentResult:
    tools = toolbox.specs()
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    result = AgentResult(question=question, final_text="")
    while result.turns < max_turns:
        result.turns += 1
        response = llm.create(system=SYSTEM, messages=messages, tools=tools)
        # Pass the content back unchanged (thinking blocks included).
        messages.append({"role": "assistant", "content": response["content"]})
        if response["stop_reason"] == "pause_turn":
            continue
        uses = [b for b in response["content"] if b.get("type") == "tool_use"]
        if response["stop_reason"] != "tool_use" or not uses:
            result.final_text = "\n".join(
                b["text"] for b in response["content"] if b.get("type") == "text"
            )
            return result
        tool_results = []
        for block in uses:
            if len(result.tool_calls) >= max_tool_calls:
                result.stopped_by_budget = True
                content, is_error = "Tool-call budget used up. Write your final answer now.", True
            else:
                result.tool_calls.append({"tool": block["name"], "input": block["input"]})
                try:
                    content = json.dumps(toolbox.call(block["name"], dict(block["input"])))
                    is_error = False
                except ToolDenied as err:
                    result.denied.append({"tool": block["name"], "reason": str(err)})
                    content, is_error = f"Denied: {err}", True
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": content,
                    "is_error": is_error,
                }  # fmt: skip
            )
        messages.append({"role": "user", "content": tool_results})
    result.stopped_by_budget = True
    return result
