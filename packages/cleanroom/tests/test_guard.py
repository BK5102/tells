from pathlib import Path

import pytest
from cleanroom.guard import (
    SealedRunError,
    count_runs,
    open_sealed_lock,
    sealed_run,
    start_sealed_run,
)
from cleanroom.lock import seal
from cleanroom.provenance import is_tracked

CFG = {"name": "exp", "test": {"start": "2021-01-01"}, "headline": "wql"}


def _sealed_and_committed(repo) -> Path:
    lock = seal(CFG, repo.root / "locks")
    repo.commit(lock.path)
    return repo.root / "locks"


def test_no_lock_refused(git_repo) -> None:
    with pytest.raises(SealedRunError, match="no lock"):
        open_sealed_lock(CFG, git_repo.root / "locks")


def test_untracked_lock_refused(git_repo) -> None:
    lock = seal(CFG, git_repo.root / "locks")
    assert not is_tracked(lock.path)
    with pytest.raises(SealedRunError, match="not committed"):
        open_sealed_lock(CFG, git_repo.root / "locks")
    git_repo.commit(lock.path)
    assert is_tracked(lock.path)
    assert open_sealed_lock(CFG, git_repo.root / "locks").config_hash == lock.config_hash


def test_dirty_tree_refused(git_repo) -> None:
    locks = _sealed_and_committed(git_repo)
    (git_repo.root / "README").write_text("edited after sealing\n", encoding="utf-8")
    with pytest.raises(SealedRunError, match="clean checkout"):
        open_sealed_lock(CFG, locks)


def test_outside_git_refused(tmp_path: Path) -> None:
    seal(CFG, tmp_path)
    with pytest.raises(SealedRunError, match="clean checkout"):
        open_sealed_lock(CFG, tmp_path)
    # The test-only switch skips git, never the lock.
    assert open_sealed_lock(CFG, tmp_path, allow_untracked_lock_for_tests=True)
    with pytest.raises(SealedRunError, match="no lock"):
        open_sealed_lock({**CFG, "seed": 1}, tmp_path, allow_untracked_lock_for_tests=True)


def test_pinned_data_checksum_is_enforced(git_repo) -> None:
    lock = seal(CFG, git_repo.root / "locks", data_sha256="a" * 64)
    git_repo.commit(lock.path)
    with pytest.raises(SealedRunError, match="data checksum"):
        open_sealed_lock(CFG, git_repo.root / "locks", data_sha256="b" * 64)


def test_started_and_finished_are_recorded(git_repo) -> None:
    locks = _sealed_and_committed(git_repo)
    head = git_repo.git("rev-parse", "HEAD").strip()
    with sealed_run(CFG, locks, "run-1", note="first") as run:
        assert run.run_number == 1 and run.git_commit == head
        assert [u["status"] for u in run.lock.uses()] == ["started"]
    uses = run.lock.uses()
    assert [(u["run_id"], u["status"]) for u in uses] == [
        ("run-1", "started"),
        ("run-1", "finished"),
    ]
    assert uses[0]["note"] == "first" and uses[0]["git_commit"] == head
    assert count_runs(run.lock) == 1


def test_crashed_run_still_counts(git_repo) -> None:
    locks = _sealed_and_committed(git_repo)
    with pytest.raises(RuntimeError), sealed_run(CFG, locks, "run-1"):
        raise RuntimeError("scoring failed")
    lock = open_sealed_lock(CFG, locks)
    assert [u["status"] for u in lock.uses()] == ["started"]
    git_repo.commit(lock.uses_path)
    run = start_sealed_run(CFG, locks, "run-2")
    assert run.run_number == 2
    run.finish(rows=10)
    assert lock.uses()[-1] == {**lock.uses()[-1], "status": "finished", "rows": 10}
    with pytest.raises(SealedRunError, match="already recorded as finished"):
        run.finish()


def test_committed_uses_file_must_be_committed_again_before_the_next_run(git_repo) -> None:
    locks = _sealed_and_committed(git_repo)
    with sealed_run(CFG, locks, "run-1") as run:
        pass
    git_repo.commit(run.uses_path)
    with sealed_run(CFG, locks, "run-2"):
        pass
    # The tracked uses file now has uncommitted lines, so the tree is dirty.
    with pytest.raises(SealedRunError, match="clean checkout"):
        start_sealed_run(CFG, locks, "run-3")


def test_reused_run_id_refused(tmp_path: Path) -> None:
    seal(CFG, tmp_path)
    with sealed_run(CFG, tmp_path, "run-1", allow_untracked_lock_for_tests=True):
        pass
    with pytest.raises(SealedRunError, match="already recorded"):
        start_sealed_run(CFG, tmp_path, "run-1", allow_untracked_lock_for_tests=True)
    with pytest.raises(ValueError):
        start_sealed_run(CFG, tmp_path, "", allow_untracked_lock_for_tests=True)


def test_records_without_status_count_as_runs(tmp_path: Path) -> None:
    lock = seal(CFG, tmp_path)
    lock.record_use("old-run")  # written by code that predates statuses
    run = start_sealed_run(CFG, tmp_path, "new-run", allow_untracked_lock_for_tests=True)
    assert run.run_number == 2
