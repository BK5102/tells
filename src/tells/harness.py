"""The scoring harness: split, fit, score, build cases, measure, save.

Test scoring is refused until the sealed-envelope lock exists (Day 9), and
test labels are dropped the moment the data is split, so no code path below
can read them early.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from tells.cases import build_cases
from tells.config import ExperimentConfig
from tells.data.schema import validate_transactions
from tells.models.contestants import make_contestant

RUNS_DIR = Path("artifacts/runs")
EVAL_PARTS = ("valid",)  # "test" joins this list only through the lock


class ScoringLockedError(PermissionError):
    """Raised when anything asks for test-period scores before the lock exists."""


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    summary: dict[str, Any]


def split_data(cfg: ExperimentConfig, tx: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Split with the Cleanroom, verify the split on the data, and hide test labels."""
    parts = cfg.split.apply(tx, "ts")
    cfg.split.check(parts, "ts")
    parts["test"] = parts["test"].drop("label")
    return parts


def run_experiment(
    cfg: ExperimentConfig, part: str = "valid", runs_dir: Path = RUNS_DIR
) -> RunResult:
    if part == "test":
        raise ScoringLockedError("test scoring needs the sealed-envelope lock (PLAN.md Day 9)")
    if part not in EVAL_PARTS:
        raise ValueError(f"part must be one of {EVAL_PARTS}, got {part!r}")

    tx = pl.read_parquet(cfg.data)
    validate_transactions(tx)
    parts = split_data(cfg, tx)
    eval_rows = parts[part]

    started = datetime.now(UTC)
    run_dir = runs_dir / f"{started:%Y%m%dT%H%M%SZ}_{cfg.name}_{part}"
    run_dir.mkdir(parents=True, exist_ok=False)

    contestants: dict[str, Any] = {}
    for key in cfg.contestants:
        contestant = make_contestant(key, cfg.model)
        contestant.fit(parts["train"])
        scored = eval_rows.with_columns(score=contestant.score(eval_rows))
        cases = build_cases(scored, score_col="score")
        cases.write_parquet(run_dir / f"cases_{contestant.name}.parquet")
        contestants[contestant.name] = {
            "n_transactions": scored.height,
            "n_cases": cases.height,
            "n_positive_cases": int(cases["label"].sum()),
        }

    summary = {
        "config_name": cfg.name,
        "part": part,
        "started_utc": started.isoformat(),
        "rows": {name: p.height for name, p in parts.items()},
        "contestants": contestants,
    }
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return RunResult(run_dir, summary)
