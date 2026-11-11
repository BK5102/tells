"""Turn scored cases and transactions into the numbers the report uses."""

from __future__ import annotations

from typing import Any

import polars as pl
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


def transaction_metrics(scored: pl.DataFrame) -> dict[str, Any]:
    """Transaction-level PR-AUC, comparable with published per-transaction results."""
    return {
        "n_transactions": scored.height,
        "n_positive_transactions": int(scored["label"].sum()),
        "transaction_pr_auc": average_precision(
            scored["score"].to_numpy(), scored["label"].to_numpy()
        ),
    }
