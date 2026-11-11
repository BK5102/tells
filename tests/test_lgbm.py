from datetime import datetime

import numpy as np
import polars as pl
import pytest
from cleanroom.metrics import average_precision

from tells.features.checklist import build_checklist_features
from tells.models.contestants import ChecklistLightGBM, LGBMParams, make_contestant


@pytest.fixture(scope="module")
def featured(synthetic_tx: pl.DataFrame) -> pl.DataFrame:
    return synthetic_tx.join(build_checklist_features(synthetic_tx), on="tx_id")


def _halves(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    cut = datetime(2015, 1, 1)
    return df.filter(pl.col("ts") < cut), df.filter(pl.col("ts") >= cut)


FAST = {"n_estimators": 60, "min_child_samples": 5}


def test_learns_an_obvious_signal(featured: pl.DataFrame) -> None:
    train, later = _halves(featured)
    model = ChecklistLightGBM(LGBMParams.from_dict(FAST))
    model.fit(train)
    scores = model.score(later)
    assert scores.min() >= 0 and scores.max() <= 1
    assert average_precision(scores.to_numpy(), later["label"].to_numpy()) > 0.5


def test_same_seed_same_scores(featured: pl.DataFrame) -> None:
    train, later = _halves(featured)
    runs = []
    for _ in range(2):
        m = ChecklistLightGBM(LGBMParams.from_dict(FAST))
        m.fit(train)
        runs.append(m.score(later).to_numpy())
    np.testing.assert_array_equal(runs[0], runs[1])


def test_unseen_category_does_not_break_scoring(featured: pl.DataFrame) -> None:
    train, later = _halves(featured)
    m = ChecklistLightGBM(LGBMParams.from_dict(FAST))
    m.fit(train)
    odd = later.head(3).with_columns(category=pl.lit("9999"))
    assert m.score(odd).len() == 3


def test_model_never_sees_label_column_as_feature(featured: pl.DataFrame) -> None:
    train, _ = _halves(featured)
    m = ChecklistLightGBM(LGBMParams.from_dict(FAST))
    m.fit(train)
    assert "label" not in m.feature_names


def test_typed_params_reject_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown"):
        LGBMParams.from_dict({"num_leafs": 10})


def test_factory_reads_params_from_model_config() -> None:
    c = make_contestant("lgbm_checklist", {"lgbm_checklist": {"n_estimators": 7}})
    assert isinstance(c, ChecklistLightGBM)
    assert c.params.n_estimators == 7
