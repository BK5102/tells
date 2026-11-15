"""The Critic: deterministic checks on an agent's answer and the runs it relied on.

Checks run before any language-model review (CLAUDE.md agent rules):

1. The final text contains no numbers typed by the model (rule 4). Run ids and
   config names are allowed; any other digit sequence is a failure.
2. Every run the answer relies on exists and passed its canary.
3. Every run's label-permutation control, if present, is consistent with chance.
4. Any test-period run was scored under a lock.
5. Every claim names a contestant and run that exist, and its direction
   ("better", "worse", "no clear difference") matches the run's bootstrap
   interval against the reference.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from tells.agent.tools import RUN_ID, Toolbox, ToolDenied

_RUN_ID_ANYWHERE = re.compile(
    r"[0-9]{8}T[0-9]{6}Z_[a-z0-9_]+_(?:valid|test)_[0-9a-f]{8}_[0-9a-f]{6}"
)
_DIGITS = re.compile(r"\d")


@dataclass
class CriticReport:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "ok": ok, "detail": detail})
        self.passed = self.passed and ok


def _parse_tail_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


def _direction(r: dict[str, Any]) -> str:
    if r["ci_low"] > 0:
        return "better"
    if r["ci_high"] < 0:
        return "worse"
    return "no clear difference"


def review(final_text: str, toolbox: Toolbox) -> CriticReport:
    report = CriticReport(passed=True)
    tail = _parse_tail_json(final_text)
    prose = final_text
    if tail is not None:
        prose = final_text[: final_text.rstrip().rfind("{")]
    stripped = _RUN_ID_ANYWHERE.sub("", prose)
    stripped = re.sub(r"\b[a-z]+_[a-z0-9_]*\d[a-z0-9_]*\b", "", stripped)  # names like ttm_pc0
    typed = _DIGITS.findall(stripped)
    report.add("no typed numbers", not typed, f"{len(typed)} digits outside ids" if typed else "")
    report.add("structured summary present", tail is not None)
    if tail is None:
        return report

    runs = tail.get("runs", [])
    summaries: dict[str, dict[str, Any]] = {}
    for run_id in runs:
        try:
            summaries[run_id] = toolbox._summary(run_id)  # read-only, same validation as the tool
        except ToolDenied as err:
            report.add(f"run exists: {run_id}", False, str(err))
            continue
        s = summaries[run_id]
        report.add(f"canary passed: {run_id}", bool(s.get("canary", {}).get("ok")))
        control = (s.get("controls") or {}).get("label_permutation")
        if control is not None:
            report.add(
                f"label-permutation control consistent with chance: {run_id}",
                bool(control["consistent_with_chance"]),
            )
        if s.get("part") == "test":
            report.add(f"test run was locked: {run_id}", bool(s.get("lock")))

    for claim in tail.get("claims", []):
        run_id, name = claim.get("run_id"), claim.get("contestant")
        s = summaries.get(run_id)
        if s is None or not RUN_ID.match(str(run_id)):
            report.add(f"claim run known: {name}", False, f"run {run_id!r} not in 'runs'")
            continue
        comp = (s.get("comparisons") or {}).get("results", {}).get(name)
        if comp is None:
            report.add(f"claim contestant known: {name}", False, "no comparison for it")
            continue
        actual = _direction(comp)
        report.add(
            f"claim direction: {name} in {run_id}",
            claim.get("direction") == actual,
            f"claimed {claim.get('direction')!r}, interval says {actual!r}",
        )
    return report
