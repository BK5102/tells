"""The scoring harness: featurize, split, fit, score, build cases, measure, save.

Test scoring is refused until the sealed-envelope lock exists (Day 9), and
test labels are dropped the moment the data is split, so no code path below
can read them early.

Every run writes `results.json` (numbers plus provenance) to its own folder
and appends one line to `artifacts/runs/index.jsonl`, so every run stays on
record, including the ones where nothing helped.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from cleanroom.provenance import config_hash, file_sha256, stamp

import tells.features.checklist as checklist
from tells.cases import build_cases
from tells.config import ExperimentConfig
from tells.data.schema import validate_transactions
from tells.evaluate import case_metrics, transaction_metrics
from tells.models.contestants import make_contestant

RUNS_DIR = Path("artifacts/runs")
FEATURE_CACHE = Path("data/interim/features")
EVAL_PARTS = ("valid",)  # "test" joins this list only through the lock
PACKAGES = ("tells", "cleanroom", "polars", "numpy", "lightgbm")


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


def checklist_features(
    tx: pl.DataFrame, data_sha256: str, cache_dir: Path = FEATURE_CACHE
) -> pl.DataFrame:
    """Build checklist features, cached by data checksum and feature-code checksum.

    Changing either the data or `features/checklist.py` gives a new cache key,
    so stale features are never reused.
    """
    code_sha = file_sha256(checklist.__file__)
    key = hashlib.sha256(f"{data_sha256}:{code_sha}".encode()).hexdigest()[:16]
    path = cache_dir / f"checklist_{key}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    feats = checklist.build_checklist_features(tx)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feats.write_parquet(path)
    return feats


def run_experiment(
    cfg: ExperimentConfig,
    part: str = "valid",
    runs_dir: Path = RUNS_DIR,
    cache_dir: Path = FEATURE_CACHE,
) -> RunResult:
    if part == "test":
        raise ScoringLockedError("test scoring needs the sealed-envelope lock (PLAN.md Day 9)")
    if part not in EVAL_PARTS:
        raise ValueError(f"part must be one of {EVAL_PARTS}, got {part!r}")

    started = datetime.now(UTC)
    data_sha = file_sha256(cfg.data)
    provenance = stamp(
        cfg.raw, {"transactions": cfg.data}, PACKAGES, data_checksums={"transactions": data_sha}
    )

    tx = pl.read_parquet(cfg.data)
    validate_transactions(tx)
    # Features are built on the full history before splitting, so evaluated
    # rows can use their own past. They are past-only (tested), and labels are
    # dropped inside the builder.
    feats = checklist_features(tx, data_sha, cache_dir)
    tx = tx.join(feats, on="tx_id", how="left", maintain_order="left")
    parts = split_data(cfg, tx)
    eval_rows = parts[part]

    run_id = f"{started:%Y%m%dT%H%M%SZ}_{cfg.name}_{part}_{config_hash(cfg.raw)[:8]}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    contestants: dict[str, Any] = {}
    for key in cfg.contestants:
        contestant = make_contestant(key, cfg.model)
        t0 = time.perf_counter()
        contestant.fit(parts["train"])
        t1 = time.perf_counter()
        scores = contestant.score(eval_rows)
        t2 = time.perf_counter()
        scored = eval_rows.with_columns(score=scores)
        cases = build_cases(scored, score_col="score")
        cases.write_parquet(run_dir / f"cases_{contestant.name}.parquet")
        contestants[contestant.name] = {
            "model": contestant.describe(),
            "cost": {
                "fit_seconds": t1 - t0,
                "score_seconds": t2 - t1,
                "score_ms_per_1k_transactions": 1e6 * (t2 - t1) / max(eval_rows.height, 1),
            },
            "transactions": transaction_metrics(scored, contestant.outputs_probability),
            "cases": case_metrics(cases, cfg.budgets),
        }

    summary = {
        "run_id": run_id,
        "config_name": cfg.name,
        "part": part,
        "provenance": provenance,
        "config": cfg.raw,
        "rows": {name: p.height for name, p in parts.items()},
        "positives": {n: int(p["label"].sum()) for n, p in parts.items() if "label" in p.columns},
        "contestants": contestants,
    }
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    _append_index(runs_dir, summary)
    return RunResult(run_dir, summary)


def _append_index(runs_dir: Path, summary: dict[str, Any]) -> None:
    """One line per run with headline numbers, for quick review and for the agent later."""
    line = {
        "run_id": summary["run_id"],
        "config_hash": summary["provenance"]["config_hash"],
        "git_commit": summary["provenance"]["git"]["commit"],
        "part": summary["part"],
        "recall_at_1pct": {
            name: c["cases"]["by_budget"].get("0.01", {}).get("recall")
            for name, c in summary["contestants"].items()
        },
    }
    with (runs_dir / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
