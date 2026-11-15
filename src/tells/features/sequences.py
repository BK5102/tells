"""Account histories as model-ready sequences for a time series foundation model.

One sequence per account-day case: the account's events strictly before
midnight of that day, oldest first, at most `context` of the most recent ones
(methodology M32). Three parallel channels, the same for both datasets:

- log_amount: log(1 + |amount in USD|). Logs tame amounts that span cents to millions.
- log_gap_minutes: log(1 + minutes since the account's previous event); 0 for the first.
  Rhythm (how often an account acts) is a core tell.
- direction: +1 outgoing, -1 incoming, 0 internal (card data is all +1).

Labels are never read. Cutting at midnight is stricter than the checklist's
per-transaction rule, so it cannot leak; tests/test_sequences.py proves it.
"""

from __future__ import annotations

from datetime import datetime, time

import numpy as np
import polars as pl

CHANNELS = ("log_amount", "log_gap_minutes", "direction")
DEFAULT_CONTEXT = 256


def _events(tx: pl.DataFrame) -> pl.DataFrame:
    direction = pl.col("direction") if "direction" in tx.columns else pl.lit(1)
    return (
        tx.select("account_id", "ts", "tx_id", "amount", direction=direction)
        .sort("account_id", "ts", "tx_id")
        .with_columns(
            log_amount=pl.col("amount").abs().log1p(),
            log_gap_minutes=(
                (pl.col("ts") - pl.col("ts").shift(1).over("account_id")).dt.total_seconds() / 60
            )
            .fill_null(0.0)
            .log1p(),
            direction=pl.col("direction").cast(pl.Float64),
        )
    )


class SequenceIndex:
    """Index an account's events once, then cut many account-day sequences from it."""

    def __init__(self, tx: pl.DataFrame) -> None:
        ev = _events(tx)
        self._by_account = {
            acct: (group["ts"].to_numpy(), {c: group[c].to_numpy() for c in CHANNELS})
            for (acct,), group in ev.partition_by("account_id", as_dict=True).items()
        }

    def build(self, keys: pl.DataFrame, context: int = DEFAULT_CONTEXT) -> pl.DataFrame:
        """One row per (account_id, day) in `keys`: length plus one list column per channel."""
        out: dict[str, list] = {"account_id": [], "day": [], "length": []}
        out.update({c: [] for c in CHANNELS})
        empty = (np.array([], dtype="datetime64[us]"), {})
        for acct, day in keys.select("account_id", "day").iter_rows():
            cutoff = np.datetime64(datetime.combine(day, time.min), "us")
            ts, chans = self._by_account.get(acct, empty)
            end = int(np.searchsorted(ts, cutoff, side="left"))  # strictly before midnight
            start = max(0, end - context)
            out["account_id"].append(acct)
            out["day"].append(day)
            out["length"].append(end - start)
            for c in CHANNELS:
                out[c].append(chans[c][start:end].tolist() if end > start else [])
        schema = {
            "account_id": pl.Utf8,
            "day": pl.Date,
            "length": pl.Int64,
            **{c: pl.List(pl.Float64) for c in CHANNELS},
        }
        return pl.DataFrame(out, schema=schema)


def build_sequences(
    tx: pl.DataFrame, keys: pl.DataFrame, context: int = DEFAULT_CONTEXT
) -> pl.DataFrame:
    """One row per (account_id, day) in `keys`: length plus one list column per channel."""
    return SequenceIndex(tx).build(keys, context)


def sequence_summary(seqs: pl.DataFrame) -> pl.DataFrame:
    """Small numeric summary of each sequence (used by tests and as a sanity view)."""
    return seqs.select(
        "account_id",
        "day",
        seq_len="length",
        seq_amount_sum=pl.col("log_amount").list.sum().fill_null(0.0),
        seq_last_gap=pl.col("log_gap_minutes").list.last(),
    )


def to_model_input(seqs: pl.DataFrame) -> list[np.ndarray]:
    """One (n_channels, length) float32 array per row, for TSFM `embed` calls.

    Empty histories become an empty array; the adapter decides how to embed them.
    """
    return [
        np.asarray([row[c] for c in CHANNELS], dtype=np.float32)
        for row in seqs.select(CHANNELS).iter_rows(named=True)
    ]
