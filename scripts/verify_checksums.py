"""Compare data files with the checksums recorded in data/checksums.sha256.

Three uses (paths in the checksum file are relative to data/):

    # 1. Check every local file listed in data/checksums.sha256.
    uv run python scripts/verify_checksums.py

    # 2. Compare another copy, such as the official Kaggle download, with a recorded entry.
    uv run python scripts/verify_checksums.py --entry raw/amlworld_hf_mirror/HI-Small_Trans.csv \
        --file data/raw/amlworld_kaggle/HI-Small_Trans.csv

    # 3. Compare a checksum pasted from elsewhere (for example computed on another machine).
    uv run python scripts/verify_checksums.py --entry raw/amlworld_hf_mirror/HI-Small_Trans.csv \
        --sha256 <64 hex characters>

Exit code 0 when everything compared matches, 1 when anything differs or is missing.
A mismatch with the Kaggle file does not by itself mean the data differ: line endings
or column order can change the bytes. Then compare counts with scripts/check_data_fit.py.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

DATA_DIR = Path("data")
CHECKSUMS = DATA_DIR / "checksums.sha256"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks so multi-gigabyte files fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checksums(path: Path) -> dict[str, str]:
    """Parse `sha256sum` output: one `<hash>  <relative path>` per line."""
    recorded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        recorded[name.strip().lstrip("*")] = digest.lower()
    return recorded


def check_local(recorded: dict[str, str], data_dir: Path) -> list[tuple[str, str]]:
    """Status of every recorded file: OK, MISMATCH, or MISSING."""
    rows = []
    for name, digest in recorded.items():
        path = data_dir / name
        if not path.exists():
            rows.append((name, "MISSING"))
        else:
            rows.append((name, "OK" if sha256_of(path) == digest else "MISMATCH"))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checksums", type=Path, default=CHECKSUMS)
    parser.add_argument("--entry", help="recorded path to compare against, as listed in the file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", type=Path, help="another copy of the file to hash and compare")
    group.add_argument("--sha256", help="a checksum pasted from elsewhere")
    args = parser.parse_args(argv)

    recorded = read_checksums(args.checksums)
    data_dir = args.checksums.parent

    if args.file is None and args.sha256 is None:
        rows = check_local(recorded, data_dir)
        for name, status in rows:
            print(f"{status:8}  {name}")
        return 0 if all(status == "OK" for _, status in rows) else 1

    if args.entry not in recorded:
        known = ", ".join(sorted(recorded))
        print(f"--entry must be one of the recorded paths: {known}", file=sys.stderr)
        return 1
    if args.file is not None:
        if not args.file.exists():
            print(f"{args.file} not found", file=sys.stderr)
            return 1
        other = sha256_of(args.file)
        source = str(args.file)
    else:
        other = args.sha256.strip().lower()
        if not HEX64.match(other):
            print("--sha256 must be 64 hexadecimal characters", file=sys.stderr)
            return 1
        source = "pasted value"
    same = other == recorded[args.entry]
    print(f"recorded  {recorded[args.entry]}  {args.entry}")
    print(f"compared  {other}  {source}")
    print("MATCH: byte-identical" if same else "DIFFERENT: compare counts with check_data_fit.py")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
