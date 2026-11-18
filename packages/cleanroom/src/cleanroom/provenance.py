"""Provenance: record exactly what produced a result, so anyone can trace it.

Every result should carry the config hash, the code version (git commit and
whether there were uncommitted changes), checksums of the input data, and the
versions of the libraries and models involved.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    """JSON with sorted keys and no whitespace, so equal configs hash equally."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def git_state(cwd: str | Path = ".") -> dict[str, Any]:
    """Current commit and whether tracked files have uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": dirty}


def is_tracked(path: str | Path) -> bool:
    """True if git tracks this file (committed, or at least staged).

    A file only proves it existed at a point in time if git dates it, so
    sealed records must be tracked before they are relied on.
    """
    p = Path(path)
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", p.name],
            cwd=p.parent,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def package_versions(names: Iterable[str]) -> dict[str, str]:
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def stamp(
    config: Mapping[str, Any],
    data_files: Mapping[str, str | Path],
    packages: Iterable[str] = (),
    cwd: str | Path = ".",
    data_checksums: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the provenance record for one run.

    `data_checksums` may pass precomputed checksums (by the same keys as
    `data_files`) to avoid hashing large files twice.
    """
    checksums = dict(data_checksums or {})
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "config_hash": config_hash(config),
        "git": git_state(cwd),
        "data": {
            key: {"path": str(path), "sha256": checksums.get(key) or file_sha256(path)}
            for key, path in data_files.items()
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(packages),
    }
