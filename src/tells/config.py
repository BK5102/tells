"""Read experiment configs. Every split comes from here, through `cleanroom.split`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from cleanroom.split import Split

DEFAULT_BUDGETS = (0.005, 0.01, 0.02, 0.05)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: str
    data: Path
    split: Split
    contestants: tuple[str, ...]
    budgets: tuple[float, ...] = DEFAULT_BUDGETS
    model: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)  # the parsed file, hashed by the lock


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    budgets = tuple(float(b) for b in raw.get("budgets", DEFAULT_BUDGETS))
    if not all(0 < b < 1 for b in budgets):
        raise ValueError(f"budgets must be shares between 0 and 1, got {budgets}")
    return ExperimentConfig(
        name=raw.get("name", path.stem),
        dataset=raw["dataset"],
        data=Path(raw["data"]),
        split=Split.from_dict(raw["split"]),
        contestants=tuple(raw.get("contestants", ["lazy_amount"])),
        budgets=budgets,
        model=raw.get("model", {}),
        raw=raw,
    )
