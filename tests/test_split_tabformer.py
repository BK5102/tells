from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from cleanroom.split import Split, Window

from tells.cases import build_cases
from tells.config import load_config
from tells.data.tabformer import load_tabformer

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "tabformer.yaml"
FIXTURE = Path(__file__).parent / "fixtures" / "tabformer_tiny.csv"


def test_config_split_is_ordered_and_gapped() -> None:
    split = load_config(CONFIG).split
    assert split.min_gap == timedelta(days=90)
    assert split.test.end <= datetime(2020, 1, 1)  # 2020 has no fraud in this file


def test_cases_built_per_part_never_straddle_windows() -> None:
    # Windows start at midnight, so splitting transactions and then building
    # cases cannot put one account-day in two parts.
    tx = load_tabformer(FIXTURE)

    split = Split(
        train=Window(None, datetime(2002, 9, 2)),
        test=Window(datetime(2002, 9, 3), datetime(2004, 1, 1)),
    )
    parts = split.apply(tx, "ts")
    split.check(parts, "ts")
    days = [set(build_cases(p)["day"]) for p in parts.values()]
    assert not days[0] & days[1]


@pytest.mark.data
@pytest.mark.skipif(
    not (ROOT / "data/interim/tabformer.parquet").exists(),
    reason="run `uv run tells prepare tabformer` first",
)
def test_real_split_passes_leak_check_and_every_part_has_fraud() -> None:
    cfg = load_config(CONFIG)
    tx = pl.read_parquet(ROOT / cfg.data, columns=["tx_id", "ts", "label"])
    parts = cfg.split.apply(tx, "ts")
    cfg.split.check(parts, "ts")
    counts = {name: int(p["label"].sum()) for name, p in parts.items()}
    assert all(n > 1000 for n in counts.values()), counts
