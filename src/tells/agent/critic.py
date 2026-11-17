"""The Critic: deterministic checks on an agent's answer and the runs it relied on.

Checks run before any language-model review (CLAUDE.md agent rules):

1. The final text contains no numbers typed by the model (rule 4): no digits
   and no number words ("two", "half", "doubled", "percent"). Exact known
   identifiers (run ids, contestant and feature names from the runs relied
   on, config names) are removed first, so names like
   D_checklist_tsfm_chronos2 do not count as numbers.
2. Every run the answer relies on exists and passed its canary.
3. Every run's label-permutation control, if present, is consistent with chance.
4. Any test-period run was scored under a lock.
5. Every claim names a contestant and run that exist, uses the run's tested
   metric, and its direction ("better", "worse", "no clear difference")
   matches the bootstrap interval and the Holm-corrected p-value.
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
# Spelled-out quantities are typed numbers too.
NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "hundreds", "thousand", "thousands", "million", "millions", "billion",
    "half", "halves", "halved", "third", "thirds", "quarter", "quarters",
    "double", "doubled", "triple", "tripled", "twice", "percent", "percentage",
)  # fmt: skip
_NUMBER_WORD = re.compile(r"\b(?:" + "|".join(NUMBER_WORDS) + r")\b", re.IGNORECASE)
ALPHA = 0.05  # Holm-corrected significance level for a "better" or "worse" claim


@dataclass
class CriticReport:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "ok": ok, "detail": detail})
        self.passed = self.passed and ok


def split_tail(text: str) -> tuple[str, dict[str, Any] | None]:
    """(narrative, summary): the summary is the last line that is a JSON object.

    The narrative is everything before the start of that line. Cutting at the
    last "{" instead would land inside the nested claims and leave part of the
    JSON in the narrative.
    """
    offset = len(text)
    for line in reversed(text.splitlines(keepends=True)):
        offset -= len(line)
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                return text[:offset].rstrip(), json.loads(stripped)
            except json.JSONDecodeError:
                return text, None
        return text, None
    return text, None


def _parse_tail_json(text: str) -> dict[str, Any] | None:
    return split_tail(text)[1]


def _direction(r: dict[str, Any]) -> str:
    # Uses the Holm-corrected p-value, so a claim accounts for how many
    # contestants were compared (CLAUDE.md rule 11).
    p = r.get("p_holm", 1.0)
    if r["ci_low"] > 0 and p < ALPHA:
        return "better"
    if r["ci_high"] < 0 and p < ALPHA:
        return "worse"
    return "no clear difference"


def _known_identifiers(summaries: dict[str, dict[str, Any]], toolbox: Toolbox) -> set[str]:
    names = set(summaries) | set(toolbox.config_names())
    for s in summaries.values():
        names.add(str(s.get("config_name", "")))
        for name, c in (s.get("contestants") or {}).items():
            names.add(name)
            exp = c.get("explanation") or {}
            names.update(exp.get("mean_abs_shap") or {})
            names.update(((c.get("model") or {}).get("feature_gain")) or {})
    return {n for n in names if n}


def _typed_numbers(prose: str, identifiers: set[str]) -> list[str]:
    """Digits and number words left after removing known identifiers."""
    stripped = _RUN_ID_ANYWHERE.sub(" ", prose)
    # Longest first, so a name is not cut by a shorter name it contains.
    for name in sorted(identifiers, key=len, reverse=True):
        stripped = re.sub(
            r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])",
            " ",
            stripped,
            flags=re.IGNORECASE,
        )
    return _DIGITS.findall(stripped) + [m.lower() for m in _NUMBER_WORD.findall(stripped)]


def review(final_text: str, toolbox: Toolbox) -> CriticReport:
    report = CriticReport(passed=True)
    prose, tail = split_tail(final_text)
    runs = (tail or {}).get("runs", [])
    summaries: dict[str, dict[str, Any]] = {}
    run_checks = CriticReport(passed=True)
    for run_id in runs:
        try:
            summaries[run_id] = toolbox._summary(run_id)  # read-only, same validation as the tool
        except ToolDenied as err:
            run_checks.add(f"run exists: {run_id}", False, str(err))
            continue
        s = summaries[run_id]
        run_checks.add(f"canary passed: {run_id}", bool(s.get("canary", {}).get("ok")))
        control = (s.get("controls") or {}).get("label_permutation")
        if control is not None:
            run_checks.add(
                f"label-permutation control consistent with chance: {run_id}",
                bool(control["consistent_with_chance"]),
            )
        if s.get("part") == "test":
            run_checks.add(f"test run was locked: {run_id}", bool(s.get("lock")))

    typed = _typed_numbers(prose, _known_identifiers(summaries, toolbox))
    detail = ""
    if typed:
        words = sorted({t for t in typed if not t.isdigit()})
        detail = f"{len(typed)} numbers outside identifiers" + (
            f" (words: {', '.join(words)})" if words else ""
        )
    report.add("no typed numbers", not typed, detail)
    report.add("structured summary present", tail is not None)
    if tail is None:
        return report
    for c in run_checks.checks:
        report.add(c["check"], c["ok"], c["detail"])

    for claim in tail.get("claims", []):
        run_id, name = claim.get("run_id"), claim.get("contestant")
        s = summaries.get(run_id)
        if s is None or not RUN_ID.match(str(run_id)):
            report.add(f"claim run known: {name}", False, f"run {run_id!r} not in 'runs'")
            continue
        comparisons = s.get("comparisons") or {}
        comp = comparisons.get("results", {}).get(name)
        if comp is None:
            report.add(f"claim contestant known: {name}", False, "no comparison for it")
            continue
        tested = comparisons.get("metric")
        if claim.get("metric") != tested:
            report.add(
                f"claim metric: {name} in {run_id}",
                False,
                f"claimed {claim.get('metric')!r}, but the run only tested {tested!r}",
            )
            continue
        actual = _direction(comp)
        report.add(
            f"claim direction: {name} in {run_id}",
            claim.get("direction") == actual,
            f"claimed {claim.get('direction')!r}, interval and Holm p-value say {actual!r}",
        )
    return report
