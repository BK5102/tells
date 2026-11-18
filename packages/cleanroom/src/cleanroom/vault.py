"""The answer-key vault: grade predictions only after they are final.

When the truth for an evaluation is known in advance (for example the events
planted in a simulation, or held-back outcomes), the file that holds it is
the answer key. `seal_answer_key` records its sha256 in a vault record; commit
that record so git dates it. `release_answer_key` hands the key back only
for a sealed run that has already finished, so predictions are written
before anyone can look at the answers, and it logs every release. Like an
exam board that keeps the marking scheme in a sealed envelope and opens it
only after the papers are handed in.

Refusals raise VaultError with the reason. Nothing here stops a person from
opening the file directly; the vault makes the honest path the easy one and
leaves a record (the release log and git history) that shows when it was
followed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cleanroom.provenance import config_hash, file_sha256, git_state, is_tracked

VAULT_SUFFIX = ".vault.json"


class VaultError(PermissionError):
    """The answer key cannot be released (or sealed) under the vault's rules."""


@dataclass(frozen=True)
class Vault:
    record_path: Path
    key_path: Path
    sha256: str

    @property
    def releases_path(self) -> Path:
        name = self.record_path.name.removesuffix(VAULT_SUFFIX)
        return self.record_path.with_name(f"{name}.releases.jsonl")

    def releases(self) -> list[dict[str, Any]]:
        if not self.releases_path.exists():
            return []
        lines = self.releases_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]


def seal_answer_key(
    path: str | Path, vault_path: str | Path | None = None, note: str = ""
) -> Vault:
    """Record the answer key's sha256 in a vault record, `<key file>.vault.json` by default.

    Sealing the same unchanged file again returns the existing record. If a
    record exists and the file has changed, sealing refuses: a new key needs
    a new record, made visibly.
    """
    key = Path(path)
    if not key.is_file():
        raise VaultError(f"answer key {key} does not exist")
    record_path = (
        Path(vault_path) if vault_path is not None else key.with_name(key.name + VAULT_SUFFIX)
    )
    if not record_path.name.endswith(VAULT_SUFFIX):
        raise ValueError(f"vault record name must end with {VAULT_SUFFIX!r}")
    digest = file_sha256(key)
    if record_path.exists():
        existing = load_vault(record_path)
        if existing.sha256 != digest or existing.key_path.resolve() != key.resolve():
            raise VaultError(
                f"vault record {record_path} already seals a different answer key; "
                "make a new record instead of editing a sealed one"
            )
        return existing
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "file": Path(os.path.relpath(key, record_path.parent)).as_posix(),
        "sha256": digest,
        "size_bytes": key.stat().st_size,
        "sealed_utc": datetime.now(UTC).isoformat(),
        "git": git_state(record_path.parent),
        "note": note,
    }
    with record_path.open("x", encoding="utf-8") as f:  # "x" refuses to overwrite
        json.dump(record, f, indent=2)
    return Vault(record_path, key, digest)


def load_vault(record_path: str | Path) -> Vault:
    p = Path(record_path)
    if not p.is_file():
        raise VaultError(f"vault record {p} does not exist")
    record = json.loads(p.read_text(encoding="utf-8"))
    return Vault(p, p.parent / record["file"], record["sha256"])


def _finished(uses_path: Path, run_id: str) -> tuple[bool, bool]:
    """(started, finished) for run_id in a uses file."""
    if not uses_path.is_file():
        return False, False
    started = finished = False
    for line in uses_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        u = json.loads(line)
        if u.get("run_id") != run_id:
            continue
        started = True
        finished = finished or u.get("status") == "finished"
    return started, finished


def release_answer_key(
    vault: Vault | str | Path,
    lock_path: str | Path,
    uses_path: str | Path,
    run_id: str,
    allow_untracked_for_tests: bool = False,
) -> bytes:
    """Return the answer key's bytes, if and only if all of these hold:

    1. `lock_path` is a valid lock file (its stored config matches its hash),
    2. the lock file and the vault record are tracked by git,
    3. `uses_path` records that `run_id` finished (see `cleanroom.guard`),
    4. the key file still has the sha256 it was sealed with.

    Each release is appended to the vault's release log. The bytes returned
    are the ones that were checked, so read tables from them, for example
    `polars.read_parquet(io.BytesIO(data))`. `allow_untracked_for_tests`
    skips only the git checks, for unit tests outside a repository.
    """
    v = vault if isinstance(vault, Vault) else load_vault(vault)
    lock = Path(lock_path)
    uses = Path(uses_path)
    if not lock.is_file():
        raise VaultError(f"no lock at {lock}; a sealed run needs a lock before grading")
    try:
        stored = json.loads(lock.read_text(encoding="utf-8"))
        valid = config_hash(stored["config"]) == stored["config_hash"]
    except (ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise VaultError(f"lock file {lock.name} does not match its own config hash")
    if not allow_untracked_for_tests:
        if not is_tracked(lock):
            raise VaultError(f"lock {lock} is not committed to git; commit it before grading")
        if not is_tracked(v.record_path):
            raise VaultError(
                f"vault record {v.record_path} is not committed to git; commit it before grading"
            )
    started, finished = _finished(uses, run_id)
    if not finished:
        state = "started but has not finished" if started else "is not recorded"
        raise VaultError(
            f"run {run_id!r} {state} in {uses.name}; the answer key opens only after "
            "a sealed run has written its results"
        )
    if not v.key_path.is_file():
        raise VaultError(f"answer key {v.key_path} is missing")
    data = v.key_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != v.sha256:
        raise VaultError(f"answer key {v.key_path.name} has changed since it was sealed")
    entry = {
        "run_id": run_id,
        "released_utc": datetime.now(UTC).isoformat(),
        "sha256": v.sha256,
        "lock": lock.name,
        "lock_config_hash": stored["config_hash"],
        "uses_file": uses.name,
        "git_commit": git_state(v.record_path.parent)["commit"],
    }
    with v.releases_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return data
