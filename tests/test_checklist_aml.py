"""Money-movement features on the AMLWorld fixture: leak test first, then hand-computed values."""

from pathlib import Path

import pytest
from cleanroom.pastonly import assert_ignores, assert_past_only

from tells.data.amlworld import load_amlworld
from tells.features.checklist import build_checklist_features

FIXTURE = Path(__file__).parent / "fixtures" / "amlworld_tiny.csv"


@pytest.fixture(scope="module")
def ev():
    return load_amlworld(FIXTURE)


def test_aml_features_are_past_only(ev) -> None:
    assert_past_only(build_checklist_features, ev, "ts", "tx_id", probe_ids=ev["tx_id"].to_list())
    assert_ignores(build_checklist_features, ev, ["label"], "tx_id")


def test_money_in_and_out_hand_computed(ev) -> None:
    f = {r["tx_id"]: r for r in build_checklist_features(ev).iter_rows(named=True)}
    # tx_id 8 = raw row 4 (A1 sends $10 on 09-02 03:00). A1's earlier events:
    # a self-transfer at 09-01 00:20 and $117.18 sent to B1 at 09-01 01:00.
    # A1 also receives $1000 at the same minute (raw row 3), which must not count.
    a1 = f[8]
    assert a1["direction"] == 1
    assert a1["n_prior"] == 2
    assert a1["out_amount_7d"] == pytest.approx(117.18)
    assert a1["in_amount_7d"] == 0
    assert a1["out_amount_24h"] == 0  # 01:00 the day before is more than 24h earlier
    assert a1["n_new_counterparties_7d"] == 2  # itself, then B1
    # tx_id 7 = raw row 3 seen by A1 as incoming, same minute, same history.
    assert f[7]["direction"] == -1
    assert f[7]["n_prior"] == 2


def test_card_data_without_direction_is_all_outgoing(synthetic_tx) -> None:
    feats = build_checklist_features(synthetic_tx.drop("direction", strict=False))
    assert set(feats["direction"].unique()) == {1}
    assert feats["in_amount_7d"].sum() == 0
