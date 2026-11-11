"""Read experiment configs. Every split comes from here, through `cleanroom.split`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from cleanroom.split import Split


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    data: Path
    split: Split
    raw: dict[str, Any]  # the parsed file, kept for hashing by the lock (Day 9)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig(
        dataset=raw["dataset"],
        data=Path(raw["data"]),
        split=Split.from_dict(raw["split"]),
        raw=raw,
    )
