"""Turn scored cases and transactions into the numbers the report uses."""

from __future__ import annotations

from typing import Any

import polars as pl
from cleanroom.calibration import brier_score, expected_calibration_error, reliability_table
from cleanroom.metrics import average_precision, precision_at_budget, recall_at_budget


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
        "case_pr_auc": average_precision(scores, labels),
        "by_budget": {},
    }
    for b in budgets:
        out["by_budget"][str(b)] = {
            "recall": recall_at_budget(scores, labels, b, days),
            "precision": precision_at_budget(scores, labels, b, days),
            "first_day_recall": recall_at_budget(scores, first, b, days) if first.sum() else None,
        }
    return out


def transaction_metrics(scored: pl.DataFrame, outputs_probability: bool) -> dict[str, Any]:
    """Transaction-level PR-AUC, plus calibration when scores are probabilities.

    Calibration is measured per transaction because that is where the model's
    probabilities live; a case score is a maximum, not a probability.
    """
    scores = scored["score"].to_numpy()
    labels = scored["label"].to_numpy()
    out: dict[str, Any] = {
        "n_transactions": scored.height,
        "n_positive_transactions": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "transaction_pr_auc": average_precision(scores, labels),
    }
    if outputs_probability:
        out["calibration"] = {
            "brier": brier_score(scores, labels),
            "brier_of_base_rate": brier_score([labels.mean()] * len(labels), labels),
            "ece_quantile_10": expected_calibration_error(scores, labels),
            "reliability_quantile_10": reliability_table(scores, labels).to_dicts(),
        }
    return out
