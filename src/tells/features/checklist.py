"""Checklist features: the red flags a bank's fraud model typically uses.

Two kinds of feature:

- Facts about the transaction itself (amount, hour, channel). These are known
  at the moment the transaction is scored, so they are allowed.
- History of the same account, built ONLY from transactions with a strictly
  earlier timestamp. Transactions that share a timestamp are treated as
  simultaneous, so none of them sees the others.

Money-movement features (money in versus out, bursts of new counterparties)
use the optional `direction` column (+1 outgoing, -1 incoming, 0 internal).
Card data has no incoming side, so it is treated as all outgoing.

Labels are dropped before anything is computed. Both guarantees are tested
with `cleanroom.pastonly` in tests/test_checklist_leaks.py.
"""

from __future__ import annotations

import polars as pl

WINDOWS = {"1h": "1h", "24h": "24h", "7d": "7d", "30d": "30d"}

CURRENT_COLUMNS = [
    "amount", "log_abs_amount", "is_refund", "hour", "weekday", "is_online", "is_chip", "has_error",
    "direction",
]  # fmt: skip
HISTORY_COLUMNS = [
    "n_prior", "secs_since_prev",
    "n_tx_1h", "n_tx_24h", "n_tx_7d", "n_tx_30d",
    "sum_amount_24h", "sum_amount_7d", "mean_amount_30d", "std_amount_30d", "max_amount_30d",
    "amount_vs_mean_30d", "amount_z_30d", "n_errors_7d", "share_online_30d",
    "is_new_merchant", "is_new_category",
    "n_new_counterparties_24h", "n_new_counterparties_7d",
    "in_amount_24h", "out_amount_24h", "in_amount_7d", "out_amount_7d", "n_in_24h", "n_out_24h",
]  # fmt: skip
FEATURE_COLUMNS = CURRENT_COLUMNS + HISTORY_COLUMNS


def _current(tx: pl.DataFrame) -> pl.DataFrame:
    errors = (
        (pl.col("errors").is_not_null() & (pl.col("errors") != ""))
        if "errors" in tx.columns
        else pl.lit(False)
    )
    direction = pl.col("direction") if "direction" in tx.columns else pl.lit(1)
    return tx.with_columns(
        direction=direction.cast(pl.Int8),
        log_abs_amount=pl.col("amount").abs().log1p(),
        is_refund=(pl.col("amount") < 0).cast(pl.Int8),
        hour=pl.col("ts").dt.hour().cast(pl.Int8),
        weekday=pl.col("ts").dt.weekday().cast(pl.Int8),
        is_online=(pl.col("channel") == "Online Transaction").fill_null(False).cast(pl.Int8),
        is_chip=(pl.col("channel") == "Chip Transaction").fill_null(False).cast(pl.Int8),
        has_error=errors.cast(pl.Int8),
    )


def _history(tx: pl.DataFrame) -> pl.DataFrame:
    """Per (account, timestamp) history from strictly earlier timestamps."""
    per_ts = (
        tx.group_by("account_id", "ts")
        .agg(
            n=pl.len(),
            s=pl.col("amount").sum(),
            s2=(pl.col("amount") ** 2).sum(),
            mx=pl.col("amount").max(),
            n_err=pl.col("has_error").cast(pl.Int64).sum(),
            n_onl=pl.col("is_online").cast(pl.Int64).sum(),
            n_new=pl.col("is_new_merchant").cast(pl.Int64).sum(),
            n_in=(pl.col("direction") == -1).cast(pl.Int64).sum(),
            n_out=(pl.col("direction") == 1).cast(pl.Int64).sum(),
            amt_in=pl.col("amount").filter(pl.col("direction") == -1).sum(),
            amt_out=pl.col("amount").filter(pl.col("direction") == 1).sum(),
        )
        .sort("account_id", "ts")
    )
    hist = per_ts.with_columns(
        n_prior=(pl.col("n").cum_sum() - pl.col("n")).over("account_id"),
        prev_ts=pl.col("ts").shift(1).over("account_id"),
    ).select("account_id", "ts", "n_prior", "prev_ts")

    for name, period in WINDOWS.items():
        # closed="left": the window is [t - period, t), so time t itself is excluded.
        rolled = per_ts.rolling("ts", period=period, closed="left", group_by="account_id").agg(
            pl.col("n").sum().alias(f"n_{name}"),
            pl.col("s").sum().alias(f"s_{name}"),
            pl.col("s2").sum().alias(f"s2_{name}"),
            pl.col("mx").max().alias(f"mx_{name}"),
            pl.col("n_err").sum().alias(f"err_{name}"),
            pl.col("n_onl").sum().alias(f"onl_{name}"),
            pl.col("n_new").sum().alias(f"new_{name}"),
            pl.col("n_in").sum().alias(f"nin_{name}"),
            pl.col("n_out").sum().alias(f"nout_{name}"),
            pl.col("amt_in").sum().alias(f"ain_{name}"),
            pl.col("amt_out").sum().alias(f"aout_{name}"),
        )
        hist = hist.join(rolled, on=["account_id", "ts"], how="left")

    n30 = pl.col("n_30d").fill_null(0)
    mean30 = pl.when(n30 > 0).then(pl.col("s_30d") / n30)
    var30 = pl.col("s2_30d") / n30 - (pl.col("s_30d") / n30) ** 2
    std30 = pl.when(n30 > 1).then(var30.clip(lower_bound=0).sqrt())
    return hist.with_columns(
        secs_since_prev=(pl.col("ts") - pl.col("prev_ts")).dt.total_seconds().cast(pl.Float64),
        n_tx_1h=pl.col("n_1h").fill_null(0),
        n_tx_24h=pl.col("n_24h").fill_null(0),
        n_tx_7d=pl.col("n_7d").fill_null(0),
        n_tx_30d=n30,
        sum_amount_24h=pl.col("s_24h").fill_null(0.0),
        sum_amount_7d=pl.col("s_7d").fill_null(0.0),
        mean_amount_30d=mean30,
        std_amount_30d=std30,
        max_amount_30d=pl.col("mx_30d"),
        n_errors_7d=pl.col("err_7d").fill_null(0),
        share_online_30d=pl.when(n30 > 0).then(pl.col("onl_30d") / n30),
        n_new_counterparties_24h=pl.col("new_24h").fill_null(0),
        n_new_counterparties_7d=pl.col("new_7d").fill_null(0),
        in_amount_24h=pl.col("ain_24h").fill_null(0.0),
        out_amount_24h=pl.col("aout_24h").fill_null(0.0),
        in_amount_7d=pl.col("ain_7d").fill_null(0.0),
        out_amount_7d=pl.col("aout_7d").fill_null(0.0),
        n_in_24h=pl.col("nin_24h").fill_null(0),
        n_out_24h=pl.col("nout_24h").fill_null(0),
    )


def build_checklist_features(tx: pl.DataFrame) -> pl.DataFrame:
    """Return one row per transaction: tx_id plus FEATURE_COLUMNS, in input order."""
    tx = _current(tx.drop("label", strict=False))
    tx = tx.with_columns(
        is_new_merchant=(
            pl.col("ts") == pl.col("ts").min().over("account_id", "counterparty")
        ).cast(pl.Int8),
        is_new_category=(pl.col("ts") == pl.col("ts").min().over("account_id", "category")).cast(
            pl.Int8
        ),
    )
    hist = _history(tx)
    out = tx.join(hist, on=["account_id", "ts"], how="left", maintain_order="left")
    out = out.with_columns(
        amount_vs_mean_30d=pl.when(pl.col("mean_amount_30d") > 0).then(
            pl.col("amount") / pl.col("mean_amount_30d")
        ),
        amount_z_30d=pl.when(pl.col("std_amount_30d") > 0).then(
            (pl.col("amount") - pl.col("mean_amount_30d")) / pl.col("std_amount_30d")
        ),
    )
    return out.select("tx_id", *FEATURE_COLUMNS)
