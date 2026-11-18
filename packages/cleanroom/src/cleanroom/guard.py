"""Sealed runs: score the held-back period only under a plan anyone can verify.

A lock (`cleanroom.lock`) proves the plan came before the results only if
git dates it, and a run's recorded commit names the code that ran only if
nothing was uncommitted. So a sealed run refuses to start unless:

1. a lock exists for exactly this config (and data checksum, if pinned),
2. the git working tree has no uncommitted changes to tracked files, and
3. the lock file itself is tracked by git.

Each run appends a "started" record to the lock's uses file before anything
held back is revealed, and a "finished" record after its results are
written. A run that crashes leaves only "started", and still counts, so the
number of attempts can always be reported. Records without a status (from
older code) count as one run each.

Because the uses file should itself be committed (it is the public record of
attempts), committing it after one run is needed before the next can start
on a clean tree.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cleanroom.lock import Lock, LockMissingError, require_lock
from cleanroom.provenance import git_state, is_tracked


class SealedRunError(PermissionError):
    """The run cannot be verified as sealed, so it must not score the held-back period."""


def count_runs(lock: Lock) -> int:
    """Sealed runs recorded against a lock: every record except "finished" ones."""
    return sum(1 for u in lock.uses() if u.get("status") != "finished")


def open_sealed_lock(
    config: dict[str, Any],
    lock_dir: str | Path,
    data_sha256: str | None = None,
    allow_untracked_lock_for_tests: bool = False,
) -> Lock:
    """The lock for `config`, after checking the three conditions in the module docstring.

    `allow_untracked_lock_for_tests` exists only for unit tests that seal
    locks in temporary folders outside git. It skips the git checks, never
    the lock itself.
    """
    try:
        lock = require_lock(config, lock_dir, data_sha256=data_sha256)
    except LockMissingError as err:
        raise SealedRunError(f"{err}. Seal the config and commit the lock first.") from err
    if allow_untracked_lock_for_tests:
        return lock
    state = git_state(lock.path.parent)
    if state["commit"] == "unknown" or state["dirty"] is not False:
        raise SealedRunError(
            "a sealed run needs a clean checkout: commit or discard uncommitted changes "
            f"first (commit {state['commit']}, dirty {state['dirty']})"
        )
    if not is_tracked(lock.path):
        raise SealedRunError(
            f"lock {lock.path} is not committed to git; commit it before a sealed run"
        )
    return lock


@dataclass
class SealedRun:
    lock: Lock
    run_id: str
    run_number: int  # 1 for the first sealed run under this lock
    git_commit: str
    finished: bool = field(default=False)

    @property
    def uses_path(self) -> Path:
        return self.lock.uses_path

    def finish(self, **details: Any) -> None:
        """Record that the run wrote its results. Call once, after writing them."""
        if self.finished:
            raise SealedRunError(f"run {self.run_id} is already recorded as finished")
        self.lock.record_use(self.run_id, status="finished", git_commit=self.git_commit, **details)
        self.finished = True


def start_sealed_run(
    config: dict[str, Any],
    lock_dir: str | Path,
    run_id: str,
    data_sha256: str | None = None,
    allow_untracked_lock_for_tests: bool = False,
    **details: Any,
) -> SealedRun:
    """Check the lock and git, then append the "started" record. Returns the open run.

    Call this before revealing anything held back. Extra `details` are stored
    in the started record.
    """
    if not run_id:
        raise ValueError("run_id must not be empty")
    lock = open_sealed_lock(config, lock_dir, data_sha256, allow_untracked_lock_for_tests)
    if any(u.get("run_id") == run_id for u in lock.uses()):
        raise SealedRunError(f"run id {run_id!r} is already recorded under this lock")
    number = count_runs(lock) + 1
    commit = git_state(lock.path.parent)["commit"]
    lock.record_use(run_id, status="started", git_commit=commit, run_number=number, **details)
    return SealedRun(lock=lock, run_id=run_id, run_number=number, git_commit=commit)


@contextmanager
def sealed_run(
    config: dict[str, Any],
    lock_dir: str | Path,
    run_id: str,
    data_sha256: str | None = None,
    allow_untracked_lock_for_tests: bool = False,
    **details: Any,
) -> Iterator[SealedRun]:
    """`start_sealed_run` as a context manager: "finished" is recorded only on a clean exit.

    If the body raises, the run stays recorded as started and counts as an
    attempt. The body may call `run.finish(...)` itself to add details.
    """
    run = start_sealed_run(
        config, lock_dir, run_id, data_sha256, allow_untracked_lock_for_tests, **details
    )
    yield run
    if not run.finished:
        run.finish()
