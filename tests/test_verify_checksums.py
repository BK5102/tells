"""scripts/verify_checksums.py: local check, a second copy, and a pasted value."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_checksums.py"
spec = importlib.util.spec_from_file_location("verify_checksums", SCRIPT)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def _setup(tmp_path: Path) -> tuple[Path, str]:
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    content = b"a,b\n1,2\n"
    (data / "raw" / "file.csv").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    sums = data / "checksums.sha256"
    sums.write_text(f"{digest}  raw/file.csv\n{'0' * 64}  raw/absent.csv\n", encoding="utf-8")
    return sums, digest


def test_local_check_reports_ok_and_missing(tmp_path):
    sums, _ = _setup(tmp_path)
    rows = dict(verify.check_local(verify.read_checksums(sums), sums.parent))
    assert rows == {"raw/file.csv": "OK", "raw/absent.csv": "MISSING"}
    assert verify.main(["--checksums", str(sums)]) == 1


def test_second_copy_matches_only_when_bytes_match(tmp_path):
    sums, _ = _setup(tmp_path)
    same = tmp_path / "same.csv"
    same.write_bytes(b"a,b\n1,2\n")
    crlf = tmp_path / "crlf.csv"
    crlf.write_bytes(b"a,b\r\n1,2\r\n")
    base = ["--checksums", str(sums), "--entry", "raw/file.csv"]
    assert verify.main([*base, "--file", str(same)]) == 0
    assert verify.main([*base, "--file", str(crlf)]) == 1


def test_pasted_value_is_validated_and_compared(tmp_path):
    sums, digest = _setup(tmp_path)
    base = ["--checksums", str(sums), "--entry", "raw/file.csv"]
    assert verify.main([*base, "--sha256", digest.upper()]) == 0
    assert verify.main([*base, "--sha256", "f" * 64]) == 1
    assert verify.main([*base, "--sha256", "not-a-hash"]) == 1
    unknown = ["--checksums", str(sums), "--entry", "raw/other.csv", "--sha256", digest]
    assert verify.main(unknown) == 1
