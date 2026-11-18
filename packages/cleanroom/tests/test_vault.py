import hashlib
import json
from pathlib import Path

import pytest
from cleanroom.guard import sealed_run, start_sealed_run
from cleanroom.lock import seal
from cleanroom.vault import VaultError, load_vault, release_answer_key, seal_answer_key

CFG = {"name": "exp", "test": {"start": "2021-01-01"}}
KEY = b"event_id,start,end\n1,10,20\n2,30,35\n"


def _setup(repo):
    """Seal and commit an answer key and a lock in a throwaway repo."""
    key = repo.root / "answers" / "key.csv"
    key.parent.mkdir()
    key.write_bytes(KEY)
    vault = seal_answer_key(key, note="planted events")
    lock = seal(CFG, repo.root / "locks")
    repo.commit(vault.record_path, lock.path)
    return vault, lock


def test_seal_records_the_sha256(tmp_path: Path) -> None:
    key = tmp_path / "key.csv"
    key.write_bytes(KEY)
    vault = seal_answer_key(key, note="n")
    assert vault.record_path == tmp_path / "key.csv.vault.json"
    record = json.loads(vault.record_path.read_text(encoding="utf-8"))
    assert record["sha256"] == hashlib.sha256(KEY).hexdigest() == vault.sha256
    assert record["file"] == "key.csv" and record["size_bytes"] == len(KEY)
    assert {"sealed_utc", "git", "note"} <= set(record)
    assert load_vault(vault.record_path) == vault


def test_resealing_unchanged_is_a_no_op_and_changed_is_refused(tmp_path: Path) -> None:
    key = tmp_path / "key.csv"
    key.write_bytes(KEY)
    first = seal_answer_key(key)
    before = first.record_path.read_text(encoding="utf-8")
    assert seal_answer_key(key) == first
    assert first.record_path.read_text(encoding="utf-8") == before
    key.write_bytes(KEY + b"3,50,60\n")
    with pytest.raises(VaultError, match="different answer key"):
        seal_answer_key(key)


def test_seal_to_a_separate_record_path(tmp_path: Path) -> None:
    key = tmp_path / "data" / "key.csv"
    key.parent.mkdir()
    key.write_bytes(KEY)
    vault = seal_answer_key(key, tmp_path / "vault" / "sim.vault.json")
    assert load_vault(vault.record_path).key_path.resolve() == key.resolve()
    with pytest.raises(ValueError):
        seal_answer_key(key, tmp_path / "vault" / "wrong.json")
    with pytest.raises(VaultError, match="does not exist"):
        seal_answer_key(tmp_path / "missing.csv")


def test_release_after_a_finished_sealed_run(git_repo) -> None:
    vault, lock = _setup(git_repo)
    # Inside the run the key stays shut: the run has not finished.
    with (
        sealed_run(CFG, lock.path.parent, "run-1"),
        pytest.raises(VaultError, match="started but has not finished"),
    ):
        release_answer_key(vault, lock.path, lock.uses_path, "run-1")
    data = release_answer_key(vault.record_path, lock.path, lock.uses_path, "run-1")
    assert data == KEY
    releases = vault.releases()
    assert [r["run_id"] for r in releases] == ["run-1"]
    assert releases[0]["sha256"] == vault.sha256
    assert releases[0]["lock_config_hash"] == lock.config_hash
    # Every release is logged, including a second one.
    release_answer_key(vault, lock.path, lock.uses_path, "run-1")
    assert len(vault.releases()) == 2


def test_release_refused_without_a_lock(git_repo) -> None:
    vault, lock = _setup(git_repo)
    with pytest.raises(VaultError, match="no lock"):
        release_answer_key(vault, lock.path.with_name("other.lock.json"), lock.uses_path, "r")


def test_release_refused_for_an_untracked_lock_or_record(git_repo) -> None:
    key = git_repo.root / "key.csv"
    key.write_bytes(KEY)
    vault = seal_answer_key(key)
    lock = seal(CFG, git_repo.root / "locks")
    lock.record_use("run-1", status="finished")
    with pytest.raises(VaultError, match="lock .* is not committed"):
        release_answer_key(vault, lock.path, lock.uses_path, "run-1")
    git_repo.commit(lock.path)
    with pytest.raises(VaultError, match="vault record .* is not committed"):
        release_answer_key(vault, lock.path, lock.uses_path, "run-1")
    git_repo.commit(vault.record_path)
    assert release_answer_key(vault, lock.path, lock.uses_path, "run-1") == KEY


def test_release_refused_for_an_unknown_or_unfinished_run(git_repo) -> None:
    vault, lock = _setup(git_repo)
    with pytest.raises(VaultError, match="is not recorded"):
        release_answer_key(vault, lock.path, lock.uses_path, "run-1")
    start_sealed_run(CFG, lock.path.parent, "run-1")  # never finished, as after a crash
    with pytest.raises(VaultError, match="started but has not finished"):
        release_answer_key(vault, lock.path, lock.uses_path, "run-1")
    assert vault.releases() == []


def test_release_refused_for_a_changed_key_or_edited_lock(tmp_path: Path) -> None:
    key = tmp_path / "key.csv"
    key.write_bytes(KEY)
    vault = seal_answer_key(key)
    lock = seal(CFG, tmp_path / "locks")
    lock.record_use("run-1", status="finished")
    args = (vault, lock.path, lock.uses_path, "run-1")
    assert release_answer_key(*args, allow_untracked_for_tests=True) == KEY

    key.write_bytes(KEY.replace(b"30", b"31"))
    with pytest.raises(VaultError, match="changed since it was sealed"):
        release_answer_key(*args, allow_untracked_for_tests=True)
    key.write_bytes(KEY)

    stored = json.loads(lock.path.read_text(encoding="utf-8"))
    stored["config"]["name"] = "edited"
    lock.path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(VaultError, match="does not match"):
        release_answer_key(*args, allow_untracked_for_tests=True)
