"""The scoring harness: featurize, split, fit, score, build cases, measure, save.

Test scoring needs a sealed-envelope lock for exactly this config
(`cleanroom.lock`). Without one, test labels are dropped the moment the data
is split, so no code path below can read them. Every test run is recorded
against its lock, so the number of test runs can always be reported.

Every run writes `results.json` (numbers plus provenance) to its own folder
and appends one line to `artifacts/runs/index.jsonl`, so every run stays on
record, including the ones where nothing helped.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from cleanroom.canary import run_canary
from cleanroom.lock import LockMissingError, require_lock
from cleanroom.provenance import config_hash, file_sha256, stamp

import tells.features.checklist as checklist
from tells.cases import build_cases
from tells.config import ExperimentConfig
from tells.data.sampling import sample_accounts
from tells.data.schema import validate_transactions
from tells.evaluate import case_metrics, compare_to_reference, headline, transaction_metrics
from tells.features.tsfm_features import (
    TSFM_CACHE,
    add_tsfm_features,
    compute_tsfm_cache,
    fit_pca,
    load_tsfm_table,
)
from tells.models.contestants import make_contestant, tsfm_models_needed
from tells.models.tsfm import make_adapter

RUNS_DIR = Path("artifacts/runs")
FEATURE_CACHE = Path("data/interim/features")
LOCKS_DIR = Path("locks")  # tracked in git: the commit history dates each lock
EVAL_PARTS = ("valid", "test")
PACKAGES = (
    "tells",
    "cleanroom",
    "polars",
    "numpy",
    "lightgbm",
    "torch",
    "chronos-forecasting",
    "granite-tsfm",
)


def _now() -> datetime:
    return datetime.now(UTC)


class ScoringLockedError(PermissionError):
    """Raised when anything asks for test-period scores before the lock exists."""


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    summary: dict[str, Any]


def split_data(
    cfg: ExperimentConfig, tx: pl.DataFrame, reveal_test_labels: bool = False
) -> dict[str, pl.DataFrame]:
    """Split with the Cleanroom, verify the split on the data, and hide test labels.

    Only `run_experiment`, after `require_lock` succeeds, passes
    reveal_test_labels=True.
    """
    parts = cfg.split.apply(tx, "ts")
    cfg.split.check(parts, "ts")
    if not reveal_test_labels:
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


CANARY_ACCOUNTS = 8


class LeakCheckFailedError(AssertionError):
    """The feature builder failed its own past-only checks on this data."""


def canary_check(tx: pl.DataFrame, seed: int = 0) -> dict[str, bool]:
    """Plant two leaks on a few accounts and require the leak checks to catch both.

    Runs on every experiment, on this experiment's own data, so a run can only
    report numbers if its leak protection was shown to work on that data.
    """
    # Half the accounts have a positive label somewhere, so the outcome canary
    # has something to leak (choosing them by label is fine: this is a test of
    # the leak checks, not a model).
    positive = tx.filter(pl.col("label") == 1)["account_id"].unique().to_list()
    negative = tx.filter(~pl.col("account_id").is_in(positive))["account_id"].unique()
    half = CANARY_ACCOUNTS // 2
    accounts = sample_accounts(positive, half, seed=seed) + sample_accounts(
        negative, half, seed=seed
    )
    sub = tx.filter(pl.col("account_id").is_in(accounts))
    report = run_canary(
        checklist.build_checklist_features,
        sub,
        time_col="ts",
        id_col="tx_id",
        group_col="account_id",
        value_col="amount",
        label_col="label",
        n_probes=10,
        seed=seed,
    )
    if not report.honest_builder_clean:
        raise LeakCheckFailedError(f"checklist features failed the past-only check: {report}")
    return report.to_dict()


def tsfm_features(
    cfg: ExperimentConfig, tx: pl.DataFrame, data_sha: str, root: Path = TSFM_CACHE
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Add TSFM columns for every model the contestants need (methodology M32 to M34)."""
    settings = dict(cfg.model.get("tsfm", {}))
    context = int(settings.pop("context", 256))
    k = int(settings.pop("pca_components", 16))
    chunk = int(settings.pop("chunk", 20_000))
    adapter_kwargs = settings.pop("adapter", {})
    windows = cfg.split.windows()
    in_any = pl.lit(False)
    for _, w in windows:
        in_any = in_any | w.mask("ts")
    keys = tx.filter(in_any).select("account_id", day=pl.col("ts").dt.date()).unique()
    train = dict(windows)["train"]

    def train_days(days: pl.Series) -> np.ndarray:
        inside = days < train.end.date()
        if train.start is not None:
            inside = inside & (days >= train.start.date())
        return inside.to_numpy()

    info: dict[str, Any] = {}
    for model in tsfm_models_needed(cfg.contestants):
        adapter = make_adapter(model, **adapter_kwargs.get(model, {}))
        t0 = time.perf_counter()
        cache = compute_tsfm_cache(adapter, tx, keys, data_sha, context, chunk, root)
        mean, components, explained = fit_pca(cache, train_days, k)
        tx = add_tsfm_features(tx, load_tsfm_table(cache, model, mean, components), model)
        info[model] = {
            "version": adapter.version,
            "revision": adapter.revision,
            "cache": str(cache),
            "context": context,
            "pca_components": k,
            "pca_variance_explained": explained,
            "n_account_days": keys.height,
            "seconds_including_cache": time.perf_counter() - t0,
        }
    return tx, info


def run_experiment(
    cfg: ExperimentConfig,
    part: str = "valid",
    runs_dir: Path = RUNS_DIR,
    cache_dir: Path = FEATURE_CACHE,
    locks_dir: Path = LOCKS_DIR,
    tsfm_dir: Path = TSFM_CACHE,
) -> RunResult:
    if part not in EVAL_PARTS:
        raise ValueError(f"part must be one of {EVAL_PARTS}, got {part!r}")
    lock = None
    if part == "test":
        try:
            lock = require_lock(cfg.raw, locks_dir)
        except LockMissingError as err:
            raise ScoringLockedError(f"{err}. Run `uv run tells lock <config>` first.") from err

    started = _now()
    data_sha = file_sha256(cfg.data)
    provenance = stamp(
        cfg.raw, {"transactions": cfg.data}, PACKAGES, data_checksums={"transactions": data_sha}
    )

    tx = pl.read_parquet(cfg.data)
    validate_transactions(tx)
    canary = canary_check(tx, cfg.seed)
    # Features are built on the full history before splitting, so evaluated
    # rows can use their own past. They are past-only (tested), and labels are
    # dropped inside the builder.
    feats = checklist_features(tx, data_sha, cache_dir)
    tx = tx.join(feats, on="tx_id", how="left", maintain_order="left")
    tx, tsfm_info = tsfm_features(cfg, tx, data_sha, tsfm_dir)
    # Accounts already labeled positive before the evaluation window starts.
    # Used only to define "first fraud day" in the answer key, never as a model
    # input, and it reads no labels from the evaluated window or later.
    eval_start = dict(cfg.split.windows())[part].start
    known_positive = set(
        tx.filter((pl.col("ts") < eval_start) & (pl.col("label") == 1))["account_id"].unique()
    )
    parts = split_data(cfg, tx, reveal_test_labels=lock is not None)
    eval_rows = parts[part]

    # The random suffix keeps two runs that start in the same second apart.
    # exist_ok=False below still refuses to overwrite an existing run.
    run_id = (
        f"{started:%Y%m%dT%H%M%SZ}_{cfg.name}_{part}_{config_hash(cfg.raw)[:8]}"
        f"_{uuid.uuid4().hex[:6]}"
    )
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    contestants: dict[str, Any] = {}
    cases_by_name: dict[str, pl.DataFrame] = {}
    reference_name = None
    for key in cfg.contestants:
        contestant = make_contestant(key, cfg.model)
        t0 = time.perf_counter()
        contestant.fit(parts["train"])
        t1 = time.perf_counter()
        scores = contestant.score(eval_rows)
        t2 = time.perf_counter()
        scored = eval_rows.with_columns(score=scores)
        cases = build_cases(scored, score_col="score", known_positive_accounts=known_positive)
        cases.write_parquet(run_dir / f"cases_{contestant.name}.parquet")
        cases_by_name[contestant.name] = cases
        if key == cfg.reference:
            reference_name = contestant.name
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
        contestants[contestant.name]["headline"] = headline(
            contestants[contestant.name]["cases"], cfg.headline_metric, cfg.headline_budget
        )

    comparisons = None
    if reference_name is not None and len(cases_by_name) > 1:
        comparisons = compare_to_reference(
            cases_by_name,
            reference_name,
            cfg.headline_metric,
            cfg.headline_budget,
            cfg.n_boot,
            cfg.seed,
        )

    summary = {
        "run_id": run_id,
        "config_name": cfg.name,
        "part": part,
        "provenance": provenance,
        "config": cfg.raw,
        "headline": {"metric": cfg.headline_metric, "budget": cfg.headline_budget},
        "canary": canary,
        "tsfm": tsfm_info,
        "known_positive_accounts_before_window": len(known_positive),
        "rows": {name: p.height for name, p in parts.items()},
        "positives": {n: int(p["label"].sum()) for n, p in parts.items() if "label" in p.columns},
        "contestants": contestants,
        "comparisons": comparisons,
    }
    if lock is not None:
        previous = len(lock.uses())
        lock.record_use(run_id, git_commit=provenance["git"]["commit"])
        summary["lock"] = {
            "config_hash": lock.config_hash,
            "lock_file": str(lock.path),
            "test_run_number": previous + 1,
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
        "headline": summary["headline"],
        "headline_value": {name: c["headline"] for name, c in summary["contestants"].items()},
    }
    with (runs_dir / "index.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
