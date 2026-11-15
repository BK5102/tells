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


# Text columns given to the tree model as categories: merchant category or
# payment format, and channel (swipe/chip/online, or sent/received/self).
CATEGORICAL = ("category", "channel")


class LightGBMContestant:
    """A LightGBM model on a named set of numeric columns plus optional categories.

    Contestants B, C, and D are all instances of this class with the same
    settings; only their input columns differ, so any difference in results
    comes from the inputs (methodology M22).
    """

    outputs_probability = True

    def __init__(
        self,
        name: str,
        numeric: list[str],
        categorical: tuple[str, ...],
        params: LGBMParams | None = None,
    ) -> None:
        self.name = name
        self.numeric = list(numeric)
        self.categorical = tuple(categorical)
        self.params = params or LGBMParams()
        self.feature_names = [*self.numeric, *(f"{c}_code" for c in self.categorical)]
        self._categories: dict[str, dict[str, int]] = {}
        self._booster: lgb.Booster | None = None

    def _matrix(self, rows: pl.DataFrame) -> np.ndarray:
        # Unknown or rare categories become -1, which LightGBM treats as missing.
        codes = {
            f"{c}_code": rows[c].replace_strict(
                self._categories[c], default=-1, return_dtype=pl.Int32
            )
            for c in self.categorical
        }
        return rows.select(self.numeric).with_columns(**codes).cast(pl.Float64).to_numpy()

    def fit(self, train: pl.DataFrame) -> None:
        p = self.params
        for c in self.categorical:
            counts = train[c].value_counts()
            common = counts.filter(pl.col("count") >= p.min_category_count)[c].sort()
            self._categories[c] = {v: i for i, v in enumerate(common.to_list())}
        dataset = lgb.Dataset(
            self._matrix(train),
            label=train["label"].to_numpy(),
            feature_name=self.feature_names,
            categorical_feature=[f"{c}_code" for c in self.categorical] or "auto",
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

    def feature_importance(self) -> dict[str, float]:
        """Total gain per input column (how much each column improved the trees)."""
        if self._booster is None:
            return {}
        gains = self._booster.feature_importance(importance_type="gain")
        return dict(zip(self.feature_names, (float(g) for g in gains), strict=True))

    def describe(self) -> dict[str, Any]:
        return {
            "library": "lightgbm",
            "library_version": lgb.__version__,
            "params": asdict(self.params),
            "features": self.feature_names,
            "n_categories": {c: len(m) for c, m in self._categories.items()},
            "feature_gain": self.feature_importance(),
        }


class ChecklistLightGBM(LightGBMContestant):
    """Contestant B: checklist features plus category and channel into LightGBM.

    Roughly what banks use today: hand-built red flags and a tree model.
    """

    def __init__(self, params: LGBMParams | None = None) -> None:
        super().__init__("B_checklist_lgbm", FEATURE_COLUMNS, CATEGORICAL, params)


def make_contestant(key: str, model_config: dict[str, Any] | None = None) -> Contestant:
    """Build a contestant from its config key.

    Keys: "lazy_amount" (A), "lgbm_checklist" (B), "tsfm_only:<model>" (C),
    "checklist_tsfm:<model>" (D). B, C, and D all use the LightGBM settings
    under model_config["lgbm_checklist"], so they differ only in their inputs.
    PCA size comes from model_config["tsfm"]["pca_components"].
    """
    from tells.features.tsfm_features import feature_names as tsfm_names

    cfg = model_config or {}
    params = LGBMParams.from_dict(cfg.get("lgbm_checklist"))
    k = int(cfg.get("tsfm", {}).get("pca_components", 16))
    if key == "lazy_amount":
        return LazyAmount()
    if key == "lgbm_checklist":
        return ChecklistLightGBM(params)
    kind, _, model = key.partition(":")
    if kind == "tsfm_only" and model:
        return LightGBMContestant(f"C_tsfm_only_{model}", tsfm_names(model, k), (), params)
    if kind == "checklist_tsfm" and model:
        return LightGBMContestant(
            f"D_checklist_tsfm_{model}",
            [*FEATURE_COLUMNS, *tsfm_names(model, k)],
            CATEGORICAL,
            params,
        )
    raise ValueError(f"unknown contestant {key!r}")


def tsfm_models_needed(keys: tuple[str, ...]) -> list[str]:
    """TSFM names that the listed contestants need, in order."""
    out: list[str] = []
    for key in keys:
        kind, _, model = key.partition(":")
        if kind in ("tsfm_only", "checklist_tsfm") and model and model not in out:
            out.append(model)
    return out
