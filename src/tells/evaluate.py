"""Turn scored cases and transactions into the numbers the report uses."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from cleanroom.bootstrap import paired_bootstrap
from cleanroom.calibration import brier_score, expected_calibration_error, reliability_table
from cleanroom.metrics import (
    average_precision,
    budget_selection,
    precision_at_budget,
    recall_at_budget,
)
from cleanroom.stats import benjamini_hochberg, holm


def _defined(fn, *args):
    """Return None when a metric is undefined (for example, no positives in the period).

    Reporting "undefined" is honest; inventing a 0 would not be.
    """
    try:
        return fn(*args)
    except ValueError as err:
        if "undefined" in str(err):
            return None
        raise


def case_metrics(cases: pl.DataFrame, budgets: tuple[float, ...]) -> dict[str, Any]:
    """Recall, precision, and first-fraud-day catch at each daily review budget."""
    scores = cases["score"].to_numpy()
    labels = cases["label"].to_numpy()
    first = cases["is_first_positive"].cast(pl.Int8).to_numpy()
    days = cases["day"].to_numpy()
    out: dict[str, Any] = {
        "n_cases": cases.height,
        "n_positive_cases": int(labels.sum()),
        "n_first_positive_cases": int(first.sum()),
        "case_pr_auc": _defined(average_precision, scores, labels),
        "by_budget": {},
    }
    for b in budgets:
        out["by_budget"][str(b)] = {
            "recall": _defined(recall_at_budget, scores, labels, b, days),
            "precision": precision_at_budget(scores, labels, b, days),
            "first_day_recall": _defined(recall_at_budget, scores, first, b, days),
        }
    return out


def transaction_metrics(scored: pl.DataFrame, outputs_probability: bool) -> dict[str, Any]:
    """Transaction-level PR-AUC, plus calibration when scores are probabilities.

    Calibration is measured per transaction because that is where the model's
    probabilities live; a case score is a maximum, not a probability.
    """
    if "source_tx" in scored.columns:
        # One transfer appears in two account histories (sender and receiver).
        # Count it once, with the higher of its two scores, as published
        # per-transaction results do.
        scored = scored.group_by("source_tx").agg(pl.col("score").max(), pl.col("label").max())
    scores = scored["score"].to_numpy()
    labels = scored["label"].to_numpy()
    out: dict[str, Any] = {
        "n_transactions": scored.height,
        "n_positive_transactions": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "transaction_pr_auc": _defined(average_precision, scores, labels),
    }
    if outputs_probability and labels.sum() > 0:
        out["calibration"] = {
            "brier": brier_score(scores, labels),
            "brier_of_base_rate": brier_score([labels.mean()] * len(labels), labels),
            "ece_quantile_10": expected_calibration_error(scores, labels),
            "reliability_quantile_10": reliability_table(scores, labels).to_dicts(),
        }
    return out


def headline(case_results: dict[str, Any], metric: str, budget: float) -> float | None:
    """The single number a run is judged by (fixed in the config before results)."""
    return case_results["by_budget"][str(budget)][metric]


def compare_to_reference(
    cases_by_contestant: dict[str, pl.DataFrame],
    reference: str,
    metric: str,
    budget: float,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Paired account-level bootstrap of every contestant against the reference.

    Only the pre-registered headline metric is tested formally; every other
    number is descriptive. Accounts are resampled whole because one account's
    cases on different days are not independent. Holm and Benjamini-Hochberg
    corrections are applied across all comparisons in the run.
    """
    ref = cases_by_contestant[reference]
    keys = ref.select("account_id", "day")
    clusters = ref["account_id"].to_numpy()
    days = ref["day"].to_numpy()
    target_col = "is_first_positive" if metric == "first_day_recall" else "label"
    target = ref[target_col].cast(pl.Int8).to_numpy().astype(float)

    # Each day's review queue is formed once, on the real cases. The bootstrap
    # then resamples which accounts are counted. Re-ranking inside each
    # resample would make duplicated accounts compete for the same slots and
    # bias every resample downward (found and fixed 2026-11-14, methodology M30).
    def shares(cases: pl.DataFrame) -> np.ndarray:
        return budget_selection(cases["score"].to_numpy(), budget, days)

    def value(share: np.ndarray, idx: np.ndarray) -> float:
        if metric == "precision":
            denom = share[idx].sum()
            return float((share[idx] * target[idx]).sum() / denom) if denom else float("nan")
        denom = target[idx].sum()
        return float((share[idx] * target[idx]).sum() / denom) if denom else float("nan")

    ref_share = shares(ref)
    results: dict[str, Any] = {}
    for name, cases in cases_by_contestant.items():
        if name == reference:
            continue
        if not cases.select("account_id", "day").equals(keys):
            raise ValueError(f"cases for {name} are not aligned with {reference}")
        share = shares(cases)
        r = paired_bootstrap(
            lambda idx, s=share: (value(ref_share, idx), value(s, idx)),
            clusters,
            n_boot=n_boot,
            seed=seed,
        )
        d = r.to_dict()
        d["reference_value"] = d.pop("a")
        d["contestant_value"] = d.pop("b")
        results[name] = d

    names = list(results)
    if names:
        p = [results[n]["p_value"] for n in names]
        for n, ph, pb in zip(names, holm(p), benjamini_hochberg(p), strict=True):
            results[n]["p_holm"] = float(ph)
            results[n]["p_bh"] = float(pb)
    return {
        "reference": reference,
        "metric": metric,
        "budget": budget,
        "difference_is": "contestant minus reference",
        "cluster": "account_id",
        "n_comparisons": len(names),
        "results": results,
    }
