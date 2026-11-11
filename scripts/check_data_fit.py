"""Check whether a fraud dataset fits the Tells idea.

Tells needs each account to have a real history (a time series) before the
fraud happens, and a long enough time span for a train / gap / test split.
This script measures exactly that, so dataset choices rest on numbers, not on
dataset descriptions.

Usage (no project install needed):
    uv run --with polars python scripts/check_data_fit.py tabformer data/raw/tabformer/card_transaction.v1.csv
    uv run --with polars python scripts/check_data_fit.py amlworld  data/raw/amlworld/HI-Small_Trans.csv
    uv run --with polars python scripts/check_data_fit.py sparkov   data/raw/sparkov/fraudTrain.csv data/raw/sparkov/fraudTest.csv
    uv run --with polars python scripts/check_data_fit.py paysim    data/raw/paysim/PS_20174392719_1491204439457_log.csv
    uv run --with polars python scripts/check_data_fit.py ieee      data/raw/ieee/train_transaction.csv
"""

from __future__ import annotations

import sys
from datetime import datetime

import polars as pl

# Chronos-2 reads sequences in patches of 16 steps, so 16 prior events is a
# rough floor for "the model has something to read".
MIN_HISTORY = 16


def load_tabformer(paths: list[str]) -> pl.LazyFrame:
    lf = pl.scan_csv(paths[0])
    return lf.select(
        account=pl.format("{}_{}", pl.col("User"), pl.col("Card")),
        ts=pl.datetime(pl.col("Year"), pl.col("Month"), pl.col("Day"))
        + pl.col("Time").str.to_time("%H:%M").cast(pl.Duration("us")),
        label=(pl.col("Is Fraud?") == "Yes").cast(pl.Int8),
    )


def load_amlworld(paths: list[str]) -> pl.LazyFrame:
    # Columns: Timestamp, From Bank, Account, To Bank, Account.1, ...,
    # Is Laundering. A transaction belongs to the history of both the sender
    # and the receiver, so each row becomes two account events.
    # The raw header repeats "Account", so name columns by position.
    names = [
        "Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received",
        "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering",
    ]  # fmt: skip
    lf = pl.scan_csv(paths[0], has_header=False, skip_rows=1, new_columns=names, schema_overrides={
        "From Bank": pl.Utf8, "To Bank": pl.Utf8, "Account": pl.Utf8, "Account.1": pl.Utf8,
    })  # fmt: skip
    ts = pl.col("Timestamp").str.to_datetime("%Y/%m/%d %H:%M")
    sent = lf.select(
        account=pl.format("{}_{}", pl.col("From Bank"), pl.col("Account")),
        ts=ts,
        label=pl.col("Is Laundering").cast(pl.Int8),
    )
    received = lf.select(
        account=pl.format("{}_{}", pl.col("To Bank"), pl.col("Account.1")),
        ts=ts,
        label=pl.col("Is Laundering").cast(pl.Int8),
    )
    return pl.concat([sent, received])


def load_sparkov(paths: list[str]) -> pl.LazyFrame:
    lf = pl.concat([pl.scan_csv(p) for p in paths])
    return lf.select(
        account=pl.col("cc_num").cast(pl.Utf8),
        ts=pl.col("trans_date_trans_time").str.to_datetime("%Y-%m-%d %H:%M:%S"),
        label=pl.col("is_fraud").cast(pl.Int8),
    )


def load_paysim(paths: list[str]) -> pl.LazyFrame:
    # step = 1 hour of simulated time. Fraud is attributed to the originator.
    lf = pl.scan_csv(paths[0])
    return lf.select(
        account=pl.col("nameOrig"),
        ts=pl.lit(datetime(2000, 1, 1)) + pl.duration(hours=pl.col("step")),
        label=pl.col("isFraud").cast(pl.Int8),
    )


def load_ieee(paths: list[str]) -> pl.LazyFrame:
    # No account id exists. Common reconstruction: card1 + addr1 + the card's
    # implied start day (day - D1). TransactionDT is seconds from an unknown
    # reference date; the reference used here is arbitrary.
    lf = pl.scan_csv(paths[0], infer_schema_length=10000)
    day = (pl.col("TransactionDT") / 86400).floor()
    return lf.select(
        account=pl.format(
            "{}_{}_{}",
            pl.col("card1"),
            pl.col("addr1"),
            (day - pl.col("D1")).cast(pl.Int64),
        ),
        ts=pl.lit(datetime(2017, 12, 1)) + pl.duration(seconds=pl.col("TransactionDT")),
        label=pl.col("isFraud").cast(pl.Int8),
    )


LOADERS = {
    "tabformer": load_tabformer,
    "amlworld": load_amlworld,
    "sparkov": load_sparkov,
    "paysim": load_paysim,
    "ieee": load_ieee,
}


def report(lf: pl.LazyFrame) -> None:
    # Rows whose account id cannot be built (missing parts) would otherwise all
    # collapse into one giant fake account.
    df = lf.collect()
    n_missing = df["account"].null_count()
    if n_missing:
        print(f"rows dropped because the account id is missing: {n_missing:,}")
        df = df.drop_nulls("account")
    df = df.sort(["account", "ts"]).with_columns(
        prior_events=pl.int_range(pl.len()).over("account")
    )

    n_rows = df.height
    n_pos = int(df["label"].sum())
    t0, t1 = df["ts"].min(), df["ts"].max()
    print(f"events (account-side rows): {n_rows:,}")
    print(f"accounts: {df['account'].n_unique():,}")
    print(f"time span: {t0} to {t1} ({(t1 - t0).days} days)")
    print(f"positive events: {n_pos:,} ({100 * n_pos / n_rows:.3f}%)")

    per_acct = df.group_by("account").agg(
        n=pl.len(),
        n_pos=pl.col("label").sum(),
        span_days=(pl.col("ts").max() - pl.col("ts").min()).dt.total_days(),
    )
    q = per_acct["n"].quantile
    print(
        f"events per account: p10={q(0.1):.0f} median={q(0.5):.0f} p90={q(0.9):.0f}; "
        f"share with >= {MIN_HISTORY} events: {(per_acct['n'] >= MIN_HISTORY).mean():.1%}; "
        f"share with exactly 1 event: {(per_acct['n'] == 1).mean():.1%}"
    )
    print(f"median active span per account: {per_acct['span_days'].median():.0f} days")

    # The key question: how much history exists BEFORE each account's first
    # positive? That is all a past-only model can use to spot the change.
    first_pos = (
        df.filter(pl.col("label") == 1).group_by("account").agg(prior=pl.col("prior_events").min())
    )
    if first_pos.height:
        fq = first_pos["prior"].quantile
        print(
            f"accounts with any positive: {first_pos.height:,} "
            f"({first_pos.height / per_acct.height:.2%} of accounts)"
        )
        print(
            f"prior events before first positive: p10={fq(0.1):.0f} median={fq(0.5):.0f} "
            f"p90={fq(0.9):.0f}; share with >= {MIN_HISTORY}: {(first_pos['prior'] >= MIN_HISTORY).mean():.1%}"
        )
        pos_per_acct = per_acct.filter(pl.col("n_pos") > 0)["n_pos"]
        print(
            f"positives per affected account: median={pos_per_acct.median():.0f} max={pos_per_acct.max()}"
        )
        # Label propagation check: once an account has a positive, how often
        # are its LATER events also positive? Near 100% means labels were
        # copied forward, which makes "had fraud before" an easy shortcut.
        after = df.join(first_pos, on="account").filter(pl.col("prior_events") > pl.col("prior"))
        if after.height:
            print(
                f"events after an account's first positive: {after.height:,}, "
                f"of which positive: {after['label'].mean():.1%}"
            )

    # Account-day "cases": what an analyst queue would hold each day.
    cases = df.group_by("account", pl.col("ts").dt.date().alias("day")).agg(
        label=pl.col("label").max()
    )
    print(
        f"account-day cases: {cases.height:,}, positive cases: {int(cases['label'].sum()):,} "
        f"({100 * cases['label'].mean():.3f}%)"
    )

    # Are positives spread across time (needed for a time split)?
    by_period = (
        df.group_by(
            pl.col("ts").dt.truncate("1y" if (t1 - t0).days > 730 else "1d").alias("period")
        )
        .agg(events=pl.len(), positives=pl.col("label").sum())
        .sort("period")
    )
    print("positives by period:")
    with pl.Config(tbl_rows=60, tbl_formatting="ASCII_FULL_CONDENSED"):
        print(by_period)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    if len(sys.argv) < 3 or sys.argv[1] not in LOADERS:
        print(__doc__)
        sys.exit(1)
    report(LOADERS[sys.argv[1]](sys.argv[2:]))


if __name__ == "__main__":
    main()
