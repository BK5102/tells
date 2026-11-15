"""The sealed envelope: write down the test plan before any test score exists.

`seal` stores a config and its hash in a lock file. Commit that file to git:
the commit history then shows the plan existed before the results did.
`require_lock` refuses to proceed unless a lock exists for exactly this
config, so any change to the plan (a different split, budget, or headline
metric) needs a new, visible lock. Every use of a lock is appended to a
record, so the number of test runs can always be reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cleanroom.provenance import canonical_json, config_hash, git_state


class LockMissingError(PermissionError):
    """No valid lock exists for this config, so test scoring must not happen."""


@dataclass(frozen=True)
class Lock:
    config_hash: str
    path: Path

    @property
    def uses_path(self) -> Path:
        return self.path.with_name(f"{self.config_hash}.uses.jsonl")

    def record_use(self, run_id: str, **details: Any) -> None:
        entry = {"run_id": run_id, "used_utc": datetime.now(UTC).isoformat(), **details}
        with self.uses_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def uses(self) -> list[dict[str, Any]]:
        if not self.uses_path.exists():
            return []
        lines = self.uses_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


def _lock_path(config: dict[str, Any], lock_dir: Path) -> tuple[str, Path]:
    h = config_hash(config)
    return h, Path(lock_dir) / f"{h}.lock.json"


def seal(config: dict[str, Any], lock_dir: str | Path, note: str = "") -> Lock:
    """Create the lock for `config`, or return the existing one unchanged."""
    h, path = _lock_path(config, Path(lock_dir))
    if path.exists():
        return Lock(h, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "config_hash": h,
        "sealed_utc": datetime.now(UTC).isoformat(),
        "git": git_state(path.parent),
        "note": note,
        # Stored as plain JSON so the envelope can be read and re-hashed by anyone.
        "config": json.loads(canonical_json(config)),
    }
    with path.open("x", encoding="utf-8") as f:  # "x" refuses to overwrite
        json.dump(record, f, indent=2)
    return Lock(h, path)


def require_lock(config: dict[str, Any], lock_dir: str | Path) -> Lock:
    """Return the lock for exactly this config, or raise LockMissingError."""
    h, path = _lock_path(config, Path(lock_dir))
    if not path.exists():
        raise LockMissingError(
            f"no lock for config hash {h[:12]}; seal the config before scoring the test period"
        )
    stored = json.loads(path.read_text(encoding="utf-8"))
    if config_hash(stored["config"]) != h or stored.get("config_hash") != h:
        raise LockMissingError(f"lock file {path.name} does not match its own config hash")
    return Lock(h, path)
