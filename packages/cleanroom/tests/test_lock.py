import json
from pathlib import Path

import pytest
from cleanroom.lock import LockMissingError, require_lock, seal
from cleanroom.provenance import config_hash

CFG = {"name": "exp", "split": {"train": {"end": "2020-01-01"}}, "headline": {"budget": 0.005}}


def test_scoring_refused_without_a_lock(tmp_path: Path) -> None:
    with pytest.raises(LockMissingError):
        require_lock(CFG, tmp_path)


def test_seal_then_require(tmp_path: Path) -> None:
    lock = seal(CFG, tmp_path, note="before test scoring")
    assert lock.config_hash == config_hash(CFG)
    assert require_lock(CFG, tmp_path).config_hash == lock.config_hash
    saved = json.loads(lock.path.read_text(encoding="utf-8"))
    assert saved["config"] == CFG
    assert saved["note"] == "before test scoring"
    assert "git" in saved and "sealed_utc" in saved


def test_any_change_to_the_config_needs_a_new_lock(tmp_path: Path) -> None:
    seal(CFG, tmp_path)
    changed = {**CFG, "headline": {"budget": 0.01}}
    with pytest.raises(LockMissingError):
        require_lock(changed, tmp_path)


def test_sealing_twice_never_overwrites(tmp_path: Path) -> None:
    first = seal(CFG, tmp_path, note="first")
    before = first.path.read_text(encoding="utf-8")
    second = seal(CFG, tmp_path, note="second")
    assert second.path == first.path
    assert first.path.read_text(encoding="utf-8") == before


def test_tampered_lock_file_is_rejected(tmp_path: Path) -> None:
    lock = seal(CFG, tmp_path)
    data = json.loads(lock.path.read_text(encoding="utf-8"))
    data["config"]["headline"]["budget"] = 0.05  # edit the envelope after sealing
    lock.path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(LockMissingError, match="does not match"):
        require_lock(CFG, tmp_path)


def test_every_use_is_recorded(tmp_path: Path) -> None:
    lock = seal(CFG, tmp_path)
    lock.record_use("run-1", git_commit="abc")
    lock.record_use("run-2", git_commit="abc")
    assert [u["run_id"] for u in lock.uses()] == ["run-1", "run-2"]
