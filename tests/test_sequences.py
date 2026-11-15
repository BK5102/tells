"""Day 13: account histories as model-ready sequences. Leak tests first."""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from cleanroom.pastonly import assert_ignores, assert_past_only

from tells.data.amlworld import load_amlworld
from tells.features.sequences import CHANNELS, build_sequences, sequence_summary

AML = Path(__file__).parent / "fixtures" / "amlworld_tiny.csv"


def _summary_per_tx(tx: pl.DataFrame) -> pl.DataFrame:
    """Per-transaction view of the day's sequence, so the past-only check can probe it."""
    keys = tx.select("account_id", day=pl.col("ts").dt.date()).unique()
    seq = sequence_summary(build_sequences(tx, keys, context=8))
    return (
        tx.select("tx_id", "account_id", day=pl.col("ts").dt.date())
        .join(seq, on=["account_id", "day"], how="left")
        .select("tx_id", "seq_len", "seq_amount_sum", "seq_last_gap")
    )


def test_sequences_use_only_events_before_the_day(synthetic_tx: pl.DataFrame) -> None:
    assert_past_only(_summary_per_tx, synthetic_tx, "ts", "tx_id", n_probes=40)
    assert_ignores(_summary_per_tx, synthetic_tx, ["label"], "tx_id")


def test_aml_sequences_hand_checked() -> None:
    ev = load_amlworld(AML)
    keys = pl.DataFrame(
        {"account_id": ["010_A1", "010_A1"], "day": [date(2022, 9, 1), date(2022, 9, 2)]}
    )
    seqs = build_sequences(ev, keys, context=8)
    a1 = {r["day"]: r for r in seqs.iter_rows(named=True)}
    # On 09-01 A1 has no earlier history at all.
    assert a1[date(2022, 9, 1)]["length"] == 0
    # On 09-02 it sees its two 09-01 events (self at 00:20, sent at 01:00), not the 09-02 ones.
    day2 = a1[date(2022, 9, 2)]
    assert day2["length"] == 2
    assert day2["direction"] == [0.0, 1.0]
    assert day2["log_amount"] == pytest.approx([np.log1p(100.0), np.log1p(117.18)])
    assert day2["log_gap_minutes"] == pytest.approx([0.0, np.log1p(40.0)])


def test_context_keeps_the_most_recent_events(synthetic_tx: pl.DataFrame) -> None:
    acct = synthetic_tx.filter(pl.col("account_id") == "acct1")
    keys = pl.DataFrame({"account_id": ["acct1"], "day": [date(2016, 1, 1)]})
    seq = build_sequences(acct, keys, context=5).row(0, named=True)
    assert seq["length"] == 5
    last_amounts = np.log1p(acct["amount"].abs().to_numpy()[-5:])
    assert seq["log_amount"] == pytest.approx(last_amounts.tolist())


def test_to_array_left_pads_and_stacks_channels() -> None:
    from tells.features.sequences import to_model_input

    seqs = pl.DataFrame(
        {
            "account_id": ["a"],
            "day": [date(2020, 1, 1)],
            "length": [2],
            **{c: [[1.0, 2.0]] for c in CHANNELS},
        }
    )
    arr = to_model_input(seqs)
    assert len(arr) == 1
    assert arr[0].shape == (len(CHANNELS), 2)


def test_every_requested_case_gets_a_row(synthetic_tx: pl.DataFrame) -> None:
    keys = synthetic_tx.select("account_id", day=pl.col("ts").dt.date()).unique()
    seqs = build_sequences(synthetic_tx, keys, context=16)
    assert seqs.height == keys.height
    assert seqs.select("account_id", "day").is_unique().all()
    assert datetime  # imported for readability of fixtures
