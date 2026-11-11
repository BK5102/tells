"""Shared test data: a small synthetic transaction set in the standard schema."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from tells.data.schema import SORT_KEYS, TRANSACTION_SCHEMA


def make_synthetic_tx(
    n_accounts: int = 40, n_per_account: int = 120, seed: int = 0
) -> pl.DataFrame:
    """Normal spending plus fraud bursts that are big, at night, at new merchants.

    Times are rounded to the hour so many transactions share a timestamp,
    which is exactly where past-only bugs hide.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2014, 1, 1)
    rows = []
    tx_id = 0
    for a in range(n_accounts):
        hours = np.sort(rng.integers(0, 2 * 365 * 24, n_per_account))
        fraud_from = rng.integers(n_per_account // 2, n_per_account) if a % 3 == 0 else None
        for i, h in enumerate(hours):
            is_fraud = fraud_from is not None and fraud_from <= i < fraud_from + 4
            ts = start + timedelta(hours=int(h))
            if is_fraud:
                ts = ts.replace(hour=3)
            rows.append(
                {
                    "tx_id": tx_id,
                    "account_id": f"acct{a}",
                    "ts": ts,
                    "amount": float(rng.uniform(900, 2000) if is_fraud else rng.gamma(2, 30)),
                    "counterparty": f"new{tx_id}" if is_fraud else f"m{rng.integers(0, 8)}",
                    "category": str(rng.choice(["5411", "5812", "5300"])),
                    "channel": "Online Transaction" if is_fraud else "Chip Transaction",
                    "label": int(is_fraud),
                    "errors": None if rng.random() > 0.05 else "Bad PIN,",
                }
            )
            tx_id += 1
    df = pl.DataFrame(rows, schema={**TRANSACTION_SCHEMA, "errors": pl.Utf8})
    return df.sort(SORT_KEYS)


@pytest.fixture(scope="session")
def synthetic_tx() -> pl.DataFrame:
    return make_synthetic_tx()
