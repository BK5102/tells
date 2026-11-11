"""Leak tests for the checklist features. Written before the features (CLAUDE.md rule 8)."""

from pathlib import Path

import polars as pl
import pytest
from cleanroom.pastonly import assert_ignores, assert_past_only

from tells.data.tabformer import load_tabformer
from tells.features.checklist import FEATURE_COLUMNS, build_checklist_features

FIXTURE = Path(__file__).parent / "fixtures" / "tabformer_tiny.csv"
SMALL = Path(__file__).parents[1] / "data" / "interim" / "tabformer_small_n300_seed0.parquet"


def test_features_use_only_strictly_earlier_rows(synthetic_tx: pl.DataFrame) -> None:
    assert_past_only(build_checklist_features, synthetic_tx, "ts", "tx_id", n_probes=60)


def test_probes_on_tied_timestamps(synthetic_tx: pl.DataFrame) -> None:
    tied = synthetic_tx.filter(pl.len().over("account_id", "ts") > 1)["tx_id"].head(20).to_list()
    assert tied, "synthetic data should contain same-timestamp transactions"
    assert_past_only(build_checklist_features, synthetic_tx, "ts", "tx_id", probe_ids=tied)


def test_features_never_read_the_label(synthetic_tx: pl.DataFrame) -> None:
    assert_ignores(build_checklist_features, synthetic_tx, ["label"], "tx_id")


def test_output_has_one_row_per_transaction_and_declared_columns(
    synthetic_tx: pl.DataFrame,
) -> None:
    feats = build_checklist_features(synthetic_tx)
    assert feats.height == synthetic_tx.height
    assert feats["tx_id"].is_unique().all()
    assert set(feats.columns) == {"tx_id", *FEATURE_COLUMNS}
    assert "label" not in feats.columns


def test_tabformer_fixture_passes() -> None:
    tx = load_tabformer(FIXTURE)
    assert_past_only(build_checklist_features, tx, "ts", "tx_id", probe_ids=tx["tx_id"].to_list())


@pytest.mark.data
@pytest.mark.skipif(not SMALL.exists(), reason="run `uv run tells prepare tabformer` first")
def test_real_cards_pass() -> None:
    tx = pl.read_parquet(SMALL)
    cards = tx["account_id"].unique().sort().head(5).to_list()
    sub = tx.filter(pl.col("account_id").is_in(cards))
    assert_past_only(build_checklist_features, sub, "ts", "tx_id", n_probes=15, seed=1)
