"""Hand-computed checks on the tiny TabFormer fixture."""

from pathlib import Path

import polars as pl
import pytest

from tells.data.tabformer import load_tabformer
from tells.features.checklist import build_checklist_features

FIXTURE = Path(__file__).parent / "fixtures" / "tabformer_tiny.csv"


@pytest.fixture(scope="module")
def feats() -> dict[int, dict]:
    df = build_checklist_features(load_tabformer(FIXTURE))
    return {r["tx_id"]: r for r in df.iter_rows(named=True)}


def test_first_transaction_has_empty_history(feats: dict[int, dict]) -> None:
    f = feats[1]  # card 0_0 at 06:21, its first transaction
    assert f["n_prior"] == 0
    assert f["n_tx_24h"] == 0
    assert f["secs_since_prev"] is None
    assert f["is_new_merchant"] == 1


def test_counts_and_sums_over_the_past(feats: dict[int, dict]) -> None:
    f = feats[0]  # card 0_0 at 06:42, after one $134.09 at 06:21
    assert f["n_prior"] == 1
    assert f["n_tx_1h"] == 1
    assert f["sum_amount_24h"] == pytest.approx(134.09)
    assert f["secs_since_prev"] == pytest.approx(21 * 60)
    assert f["amount_vs_mean_30d"] == pytest.approx(38.48 / 134.09)


def test_window_edges(feats: dict[int, dict]) -> None:
    f = feats[2]  # card 0_0 on 09-02 23:59; earlier rows were 06:21 and 06:42 on 09-01
    assert f["n_tx_24h"] == 0  # both are more than 24 hours earlier
    assert f["n_tx_7d"] == 2
    assert f["is_new_merchant"] == 1
    assert f["is_new_category"] == 0  # 5411 already seen at 06:42
    assert f["has_error"] == 1


def test_current_row_attributes(feats: dict[int, dict]) -> None:
    assert feats[2]["is_refund"] == 1
    assert feats[3]["is_online"] == 1
    assert feats[3]["hour"] == 10


def test_merchant_seen_before_is_not_new(feats: dict[int, dict]) -> None:
    assert feats[3]["is_new_merchant"] == 1
    assert feats[4]["is_new_merchant"] == 0  # same merchant, one hour later


def test_feature_frame_keeps_row_order(feats: dict[int, dict]) -> None:
    df = build_checklist_features(load_tabformer(FIXTURE))
    assert df["tx_id"].to_list() == load_tabformer(FIXTURE)["tx_id"].to_list()
    assert isinstance(df, pl.DataFrame)
