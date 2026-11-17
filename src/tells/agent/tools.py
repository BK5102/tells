"""The agent's tool layer: three tools, nothing else (CLAUDE.md agent safety rules).

The agent never gets a shell, file paths, or network access. It may only:

- `run_experiment(config, part)`: run a config from `configs/` by name. The
  test period is denied unless the Toolbox was built with `allow_test=True`
  (`tells agent --allow-test`), and even then goes through the harness, which
  refuses without a committed lock.
- `load_results(run_id)`: read a finished run's summary. Raw rows and case
  files are never returned, so labels cannot be read through this tool.
- `plot(run_id, metric)`: chart data (budgets on x, one series per contestant).

Every argument is checked against an allow-list or a strict pattern before
anything runs. Anything else raises `ToolDenied`. Methodology M39.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RUN_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[a-z0-9_]+_(valid|test)_[0-9a-f]{8}_[0-9a-f]{6}$")
CONFIG_NAME = re.compile(r"^[a-z0-9_]+$")
METRICS = ("first_day_recall", "recall", "precision")
SUMMARY_KEYS = (
    "run_id", "config_name", "part", "headline", "canary", "controls", "comparisons",
    "rows", "positives", "tsfm", "lock",
)  # fmt: skip


class ToolDenied(PermissionError):
    """The agent asked for something outside its allow-list."""


def _contestant_view(c: dict[str, Any]) -> dict[str, Any]:
    """Numbers about a contestant, never rows."""
    out = {
        "headline": c.get("headline"),
        "by_budget": c["cases"]["by_budget"],
        "case_pr_auc": c["cases"]["case_pr_auc"],
        "transaction_pr_auc": c["transactions"]["transaction_pr_auc"],
        "cost": c["cost"],
    }
    if "by_history" in c["cases"]:
        out["by_history"] = c["cases"]["by_history"]
    exp = c.get("explanation")
    if exp:
        top = list(exp["mean_abs_shap"].items())[:10]
        out["top_features"] = top
        out["tsfm_share_of_attribution"] = exp["tsfm_share_of_attribution"]
    return out


@dataclass
class Toolbox:
    configs_dir: Path = Path("configs")
    runs_dir: Path = Path("artifacts/runs")
    runner: Callable[..., Any] | None = None  # injected so tests need no real runs
    # The sealed test period is spent by a person's decision, not the agent's:
    # off unless `tells agent --allow-test` is given.
    allow_test: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)  # audit trail of every call

    def parts(self) -> list[str]:
        return ["valid", "test"] if self.allow_test else ["valid"]

    # ----- allow-list -----
    def specs(self) -> list[dict[str, Any]]:
        """Tool descriptions in the JSON-schema shape language-model APIs expect."""
        if self.allow_test:
            description = (
                "Run an experiment config by name on the validation or test period. "
                "Test scoring is refused unless the config has been sealed."
            )
        else:
            description = "Run an experiment config by name on the validation period."
        return [
            {
                "name": "run_experiment",
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "config": {"type": "string", "enum": self.config_names()},
                        "part": {"type": "string", "enum": self.parts()},
                    },
                    "required": ["config"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "load_results",
                "description": "Read the summary numbers of a finished run.",
                "input_schema": {
                    "type": "object",
                    "properties": {"run_id": {"type": "string"}},
                    "required": ["run_id"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "plot",
                "description": "Chart data for one metric across review budgets.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "run_id": {"type": "string"},
                        "metric": {"type": "string", "enum": list(METRICS)},
                    },
                    "required": ["run_id", "metric"],
                    "additionalProperties": False,
                },
            },
        ]

    def config_names(self) -> list[str]:
        return sorted(p.stem for p in self.configs_dir.glob("*.yaml"))

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The only way the agent touches anything."""
        self.calls.append({"tool": name, "args": dict(args)})
        handlers = {
            "run_experiment": self._run_experiment,
            "load_results": self._load_results,
            "plot": self._plot,
        }
        if name not in handlers:
            raise ToolDenied(f"tool {name!r} is not on the allow-list {sorted(handlers)}")
        spec = next(s for s in self.specs() if s["name"] == name)
        allowed = set(spec["input_schema"]["properties"])
        extra = set(args) - allowed
        if extra:
            raise ToolDenied(f"unexpected arguments {sorted(extra)} for {name}")
        return handlers[name](**args)

    # ----- tools -----
    def _run_experiment(self, config: str, part: str = "valid") -> dict[str, Any]:
        if not CONFIG_NAME.match(config) or config not in self.config_names():
            raise ToolDenied(f"unknown config {config!r}; choose from {self.config_names()}")
        if part == "test" and not self.allow_test:
            raise ToolDenied(
                "test scoring refused: the agent may score the test period only when a person "
                "starts it with --allow-test"
            )
        if part not in self.parts():
            raise ToolDenied(f"part must be one of {self.parts()}")
        from tells.config import load_config
        from tells.harness import ScoringLockedError, run_experiment

        runner = self.runner or run_experiment
        try:
            result = runner(load_config(self.configs_dir / f"{config}.yaml"), part=part)
        except ScoringLockedError as err:
            raise ToolDenied(f"test scoring refused: {err}") from err
        return self._view(result.summary)

    def _summary(self, run_id: str) -> dict[str, Any]:
        if not RUN_ID.match(run_id):
            raise ToolDenied(f"run_id {run_id!r} is not a valid run id")
        path = self.runs_dir / run_id / "results.json"
        if not path.exists():
            raise ToolDenied(f"no finished run {run_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _view(self, s: dict[str, Any]) -> dict[str, Any]:
        out = {k: s.get(k) for k in SUMMARY_KEYS}
        out["contestants"] = {n: _contestant_view(c) for n, c in s["contestants"].items()}
        return out

    def _load_results(self, run_id: str) -> dict[str, Any]:
        return self._view(self._summary(run_id))

    def _plot(self, run_id: str, metric: str) -> dict[str, Any]:
        if metric not in METRICS:
            raise ToolDenied(f"metric must be one of {METRICS}")
        s = self._summary(run_id)
        budgets = list(next(iter(s["contestants"].values()))["cases"]["by_budget"])
        return {
            "x_label": "daily review budget (share of cases)",
            "y_label": metric,
            "x": [float(b) for b in budgets],
            "series": {
                n: [c["cases"]["by_budget"][b][metric] for b in budgets]
                for n, c in s["contestants"].items()
            },
        }
