import hashlib
import re
from pathlib import Path

from cleanroom.provenance import config_hash, file_sha256, git_state, stamp


def test_config_hash_ignores_key_order_but_not_values() -> None:
    a = {"split": {"train": 1, "test": 2}, "seed": 0}
    b = {"seed": 0, "split": {"test": 2, "train": 1}}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({**a, "seed": 1})


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc" * 100_000)
    assert file_sha256(f, chunk_size=1000) == hashlib.sha256(b"abc" * 100_000).hexdigest()


def test_git_state_in_this_repo() -> None:
    state = git_state(Path(__file__).parent)
    assert state["commit"] == "unknown" or re.fullmatch(r"[0-9a-f]{40}", state["commit"])


def test_git_state_outside_a_repo(tmp_path: Path) -> None:
    assert git_state(tmp_path)["commit"] == "unknown"


def test_stamp_has_required_fields(tmp_path: Path) -> None:
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello")
    s = stamp({"a": 1}, {"main": f}, packages=["numpy"])
    assert s["config_hash"] == config_hash({"a": 1})
    assert s["data"]["main"]["sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert s["packages"]["numpy"] != "not installed"
    assert {"created_utc", "git", "python", "platform"} <= set(s)


def test_stamp_reuses_precomputed_checksum(tmp_path: Path) -> None:
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello")
    s = stamp({}, {"main": f}, data_checksums={"main": "precomputed"})
    assert s["data"]["main"]["sha256"] == "precomputed"
