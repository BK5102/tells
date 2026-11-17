"""Turn a finished run folder into Markdown tables. Every number comes from the run's files.

This is the deterministic half of the Reporter (PLAN.md Day 24, CLAUDE.md rule
4): tables are produced by code from `results.json` and the saved case files,
so no number in a report is ever typed by a person or a language model.
Analyses added after a run (for example the history-length split) are
recomputed from the saved cases and the data, not from memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from tells.evaluate import history_slices


def load_run(run_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "results.json").read_text(encoding="utf-8"))


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _pts(x: float) -> str:
    return f"{100 * x:+.1f}"


def headline_table(s: dict[str, Any]) -> str:
    head = s["headline"]
    comp = (s.get("comparisons") or {}).get("results", {})
    ref = (s.get("comparisons") or {}).get("reference")
    lines = [
        f"Headline: {head['metric']} at a {_pct(head['budget'])} daily budget. "
        f"Differences are contestant minus {ref}, in percentage points, with a 95% "
        "account-bootstrap interval and Holm-corrected p-value.",
        "",
        "| Contestant | Headline | Difference vs reference | 95% interval | p (Holm) |",
        "|---|---|---|---|---|",
    ]
    for name, c in s["contestants"].items():
        r = comp.get(name)
        if r is None:
            lines.append(f"| {name} | {_pct(c['headline'])} | reference | | |")
        else:
            lines.append(
                f"| {name} | {_pct(c['headline'])} | {_pts(r['diff'])} | "
                f"[{_pts(r['ci_low'])}, {_pts(r['ci_high'])}] | {r['p_holm']:.3f} |"
            )
    return "\n".join(lines)


def budget_table(s: dict[str, Any], metric: str = "first_day_recall") -> str:
    budgets = list(next(iter(s["contestants"].values()))["cases"]["by_budget"])
    lines = [
        f"{metric} at every budget:",
        "",
        "| Contestant | " + " | ".join(_pct(float(b)) for b in budgets) + " |",
        "|---|" + "---|" * len(budgets),
    ]
    for name, c in s["contestants"].items():
        vals = [_pct(c["cases"]["by_budget"][b][metric]) for b in budgets]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def cost_table(s: dict[str, Any]) -> str:
    lines = [
        "| Contestant | Fit (s) | Score (s) | Score ms per 1,000 transactions "
        "| TSFM share of attribution |",
        "|---|---|---|---|---|",
    ]
    for name, c in s["contestants"].items():
        exp = c.get("explanation") or {}
        share = exp.get("tsfm_share_of_attribution")
        lines.append(
            f"| {name} | {c['cost']['fit_seconds']:.0f} | {c['cost']['score_seconds']:.0f} | "
            f"{c['cost']['score_ms_per_1k_transactions']:.1f} | {_pct(share)} |"
        )
    for model, info in (s.get("tsfm") or {}).items():
        per = 1000 * info["seconds_including_cache"] / max(info["n_account_days"], 1)
        lines.append("")
        lines.append(
            f"TSFM `{model}` ({info['version']}): {info['n_account_days']:,} account-days, "
            f"{info['seconds_including_cache']:.0f} s including cache reads "
            f"({per:.2f} ms per account-day this run; cached chunks cost almost nothing), "
            f"PCA {info['pca_components']} components keep "
            f"{_pct(info['pca_variance_explained'])} of the variance."
        )
    return "\n".join(lines)


def history_table(run_dir: str | Path, s: dict[str, Any], cache_dir: Path | None = None) -> str:
    """First-fraud-day catch by history length, recomputed from saved cases and data.

    Reads the run's feature cache if it exists; otherwise builds the features in
    memory from every column and saves nothing, so a report can never leave an
    incomplete cache behind. Refuses if the data file changed since the run.
    """
    from cleanroom.provenance import file_sha256

    from tells.harness import FEATURE_CACHE, checklist_features

    data = Path(s["config"]["data"])
    data_sha = file_sha256(data)
    recorded = s["provenance"]["data"]["transactions"]["sha256"]
    if data_sha != recorded:
        raise ValueError(
            f"{data} checksum {data_sha[:12]} differs from the run's {recorded[:12]}; "
            "the data changed since the run, so the history split cannot be recomputed"
        )
    tx = pl.read_parquet(data)
    feats = checklist_features(
        tx, data_sha, cache_dir if cache_dir is not None else FEATURE_CACHE, write=False
    ).select("tx_id", "n_prior")
    hist = (
        tx.select("tx_id", "account_id", day=pl.col("ts").dt.date())
        .join(feats, on="tx_id")
        .group_by("account_id", "day")
        .agg(history_len=pl.col("n_prior").min())
    )
    budget = str(s["headline"]["budget"])
    lines = [
        f"First-fraud-day catch at the headline budget ({_pct(float(budget))}), split by history "
        "before that day (fewer than 16 events versus 16 or more):",
        "",
        "| Contestant | Short history | Long history |",
        "|---|---|---|",
    ]
    counts = None
    for name in s["contestants"]:
        cases = pl.read_parquet(Path(run_dir) / f"cases_{name}.parquet")
        if "history_len" not in cases.columns:
            cases = cases.join(hist, on=["account_id", "day"], how="left")
        h = history_slices(cases, (float(budget),))
        short, long_ = h["short_history"], h["long_history"]
        counts = (short["n_first_positive_cases"], long_["n_first_positive_cases"])
        lines.append(
            f"| {name} | {_pct(short['by_budget'][budget]['first_day_recall'])} | "
            f"{_pct(long_['by_budget'][budget]['first_day_recall'])} |"
        )
    if counts:
        lines.append("")
        lines.append(f"New bad accounts: {counts[0]} with short history, {counts[1]} with long.")
    return "\n".join(lines)


def render(run_dir: str | Path, with_history: bool = True) -> str:
    s = load_run(run_dir)
    prov = s["provenance"]
    control = (s.get("controls") or {}).get("label_permutation")
    parts = [
        f"# Run `{s['run_id']}`",
        "",
        f"Config `{s['config_name']}`, part `{s['part']}`, commit `{prov['git']['commit'][:7]}`"
        f"{' (uncommitted changes)' if prov['git']['dirty'] else ''}, "
        f"config hash `{prov['config_hash'][:12]}`. Canary: "
        f"{'passed' if s.get('canary', {}).get('ok') else 'NOT PASSED'}.",
        "",
        headline_table(s),
        "",
        budget_table(s),
        "",
        cost_table(s),
    ]
    if control:
        parts += [
            "",
            "Label-permutation control (reference model trained on shuffled labels): headline "
            f"{_pct(control['permuted_headline'])} against {_pct(control['random_headline'])} for "
            f"random ranking, difference {_pts(control['difference'])} points "
            f"[{_pts(control['ci_low'])}, {_pts(control['ci_high'])}]: "
            + (
                "consistent with chance."
                if control["consistent_with_chance"]
                else "NOT consistent with chance."
            ),  # fmt: skip
        ]
    if with_history:
        parts += ["", history_table(run_dir, s)]
    return "\n".join(parts) + "\n"
