"""The contestant interface, contestant A (lazy baseline), and B (checklist + LightGBM).

Every contestant learns from training rows and then gives each row it is
asked about a score where higher means more suspicious. Contestants never
see labels of the rows they score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import polars as pl

from tells.features.checklist import FEATURE_COLUMNS


class Contestant(Protocol):
    name: str
    # True when scores are probabilities, so calibration metrics apply.
    outputs_probability: bool

    def fit(self, train: pl.DataFrame) -> None: ...

    def score(self, rows: pl.DataFrame) -> pl.Series: ...

    def describe(self) -> dict[str, Any]: ...


class LazyAmount:
    """Contestant A: rank by transaction amount, biggest first. No learning at all.

    The walker in the race: any model worth its cost must beat this.
    """

    name = "A_lazy_amount"
    outputs_probability = False

    def fit(self, train: pl.DataFrame) -> None:
        return None

    def score(self, rows: pl.DataFrame) -> pl.Series:
        return rows["amount"].alias("score")

    def describe(self) -> dict[str, Any]:
        return {"rule": "score = amount"}


@dataclass(frozen=True)
class LGBMParams:
    """LightGBM settings. A fixed number of trees, no early stopping, so the
    validation period is never used to pick the model it is scored on."""

    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    min_category_count: int = 50  # rarer categories are treated as missing
    seed: int = 0
    num_threads: int = 0  # 0 = LightGBM default (all cores)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> LGBMParams:
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown LightGBM params: {sorted(unknown)}")
        return cls(**d)


class ChecklistLightGBM:
    """Contestant B: checklist features plus merchant category into a LightGBM model.

    Roughly what banks use today: hand-built red flags and a tree model.
    """

    name = "B_checklist_lgbm"
    outputs_probability = True

    def __init__(self, params: LGBMParams | None = None) -> None:
        self.params = params or LGBMParams()
        self.feature_names = [*FEATURE_COLUMNS, "category_code"]
        self._categories: dict[str, int] = {}
        self._booster: lgb.Booster | None = None

    def _matrix(self, rows: pl.DataFrame) -> np.ndarray:
        # Unknown or rare categories become -1, which LightGBM treats as missing.
        codes = rows["category"].replace_strict(self._categories, default=-1, return_dtype=pl.Int32)
        return (
            rows.select(FEATURE_COLUMNS)
            .with_columns(category_code=codes)
            .cast(pl.Float64)
            .to_numpy()
        )

    def fit(self, train: pl.DataFrame) -> None:
        p = self.params
        counts = train["category"].value_counts()
        common = counts.filter(pl.col("count") >= p.min_category_count)["category"].sort()
        self._categories = {c: i for i, c in enumerate(common.to_list())}
        dataset = lgb.Dataset(
            self._matrix(train),
            label=train["label"].to_numpy(),
            feature_name=self.feature_names,
            categorical_feature=["category_code"],
            free_raw_data=True,
            params={"verbose": -1},
        )
        lgb_params = {
            "objective": "binary",
            "learning_rate": p.learning_rate,
            "num_leaves": p.num_leaves,
            "min_child_samples": p.min_child_samples,
            "feature_fraction": p.feature_fraction,
            "bagging_fraction": p.bagging_fraction,
            "bagging_freq": p.bagging_freq,
            "lambda_l2": p.lambda_l2,
            "seed": p.seed,
            "num_threads": p.num_threads,
            "deterministic": True,
            "force_row_wise": True,
            "verbose": -1,
        }
        self._booster = lgb.train(lgb_params, dataset, num_boost_round=p.n_estimators)

    def score(self, rows: pl.DataFrame) -> pl.Series:
        if self._booster is None:
            raise RuntimeError("fit() must be called before score()")
        return pl.Series("score", self._booster.predict(self._matrix(rows)))

    def describe(self) -> dict[str, Any]:
        return {
            "library": "lightgbm",
            "library_version": lgb.__version__,
            "params": asdict(self.params),
            "features": self.feature_names,
            "n_categories": len(self._categories),
        }


def make_contestant(key: str, model_config: dict[str, Any] | None = None) -> Contestant:
    """Build a contestant from its config key; settings come from model_config[key]."""
    settings = (model_config or {}).get(key)
    if key == "lazy_amount":
        return LazyAmount()
    if key == "lgbm_checklist":
        return ChecklistLightGBM(LGBMParams.from_dict(settings))
    raise ValueError(f"unknown contestant {key!r}")
