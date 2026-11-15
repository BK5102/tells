"""Loader for AMLWorld HI-Small, IBM's synthetic anti-money-laundering data.

Source: Altman et al., NeurIPS 2023 Datasets and Benchmarks
(https://arxiv.org/abs/2306.16424), Kaggle "IBM Transactions for Anti Money
Laundering". Measured facts and traps are in docs/data-fit.md.

Choices made here, and why (see docs/methodology.md for the full reasoning):

- One raw row is one transfer between two accounts. A transfer belongs to the
  history of both the sender and the receiver, so it becomes two events: an
  outgoing one for the sender (direction +1) and an incoming one for the
  receiver (direction -1). A transfer from an account to itself becomes one
  event (direction 0), so it is not double counted.
- Rows from 2022-09-11 on are dropped. After that date the file is about 60%
  laundering, a generator artifact that a time split would otherwise turn into
  a free "late date means laundering" signal.
- Amounts come in 15 currencies, so they are converted to US dollars using the
  median exchange rate seen in cross-currency transfers on the first day only.
  The first day is inside every training window, so no later data is used.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from tells.data.schema import SORT_KEYS, TRANSACTION_SCHEMA

RAW_FILE = Path("data/raw/amlworld_hf_mirror/HI-Small_Trans.csv")
TRUNCATE_AT = datetime(2022, 9, 11)
USD = "US Dollar"

# The raw header repeats "Account", so columns are named by position.
_RAW_NAMES = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received",
    "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering",
]  # fmt: skip


def _scan_raw(path: str | Path) -> pl.LazyFrame:
    return (
        pl.scan_csv(
            path,
            has_header=False,
            skip_rows=1,
            new_columns=_RAW_NAMES,
            schema={c: pl.Utf8 for c in _RAW_NAMES},
        )
        .with_row_index("source_tx")
        .select(
            pl.col("source_tx").cast(pl.Int64),
            ts=pl.col("Timestamp").str.to_datetime("%Y/%m/%d %H:%M", time_unit="us"),
            sender=pl.format("{}_{}", pl.col("From Bank"), pl.col("Account")),
            receiver=pl.format("{}_{}", pl.col("To Bank"), pl.col("Account.1")),
            paid=pl.col("Amount Paid").cast(pl.Float64),
            pay_cur=pl.col("Payment Currency"),
            received=pl.col("Amount Received").cast(pl.Float64),
            rec_cur=pl.col("Receiving Currency"),
            fmt=pl.col("Payment Format"),
            label=pl.col("Is Laundering").cast(pl.Int8),
        )
    )


def usd_rates(raw: pl.DataFrame) -> dict[str, float]:
    """USD per unit of each currency, from first-day cross-currency transfers (median)."""
    first_day = raw["ts"].min().date()
    day1 = raw.filter(pl.col("ts").dt.date() == first_day)
    to_usd = day1.filter((pl.col("rec_cur") == USD) & (pl.col("pay_cur") != USD)).select(
        cur="pay_cur", rate=pl.col("received") / pl.col("paid")
    )
    from_usd = day1.filter((pl.col("pay_cur") == USD) & (pl.col("rec_cur") != USD)).select(
        cur="rec_cur", rate=pl.col("paid") / pl.col("received")
    )
    rates = pl.concat([to_usd, from_usd]).group_by("cur").agg(pl.col("rate").median())
    out = dict(zip(rates["cur"], rates["rate"], strict=True))
    out[USD] = 1.0
    return out


def load_amlworld(path: str | Path = RAW_FILE) -> pl.DataFrame:
    """Load AMLWorld HI-Small as account events in the standard schema, sorted."""
    raw = _scan_raw(path).filter(pl.col("ts") < TRUNCATE_AT).collect()
    rates = usd_rates(raw)
    missing = set(raw["pay_cur"].unique()) - set(rates)
    if missing:
        raise ValueError(f"no first-day exchange rate for {sorted(missing)}")
    raw = raw.with_columns(
        amount=pl.col("paid") * pl.col("pay_cur").replace_strict(rates, return_dtype=pl.Float64),
        is_self=pl.col("sender") == pl.col("receiver"),
    )
    common = ["source_tx", "ts", "amount", "label", pl.col("fmt").alias("category")]
    sent = raw.select(
        *common,
        account_id="sender",
        counterparty="receiver",
        channel=pl.when(pl.col("is_self")).then(pl.lit("self")).otherwise(pl.lit("sent")),
        direction=pl.when(pl.col("is_self")).then(0).otherwise(1).cast(pl.Int8),
        side=pl.lit(0, pl.Int64),
    )
    received = raw.filter(~pl.col("is_self")).select(
        *common,
        account_id="receiver",
        counterparty="sender",
        channel=pl.lit("received"),
        direction=pl.lit(-1, pl.Int8),
        side=pl.lit(1, pl.Int64),
    )
    events = pl.concat([sent, received]).with_columns(
        tx_id=pl.col("source_tx") * 2 + pl.col("side")
    )
    cols = [*TRANSACTION_SCHEMA, "source_tx", "direction"]
    return events.select(cols).cast(TRANSACTION_SCHEMA).sort(SORT_KEYS)
