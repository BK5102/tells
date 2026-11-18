import shutil
import subprocess
from pathlib import Path

import pytest


class Repo:
    """A throwaway git repository for tests of the git-backed checks."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.git("init", "-q")
        (root / "README").write_text("test repo\n", encoding="utf-8")
        self.commit("README")

    def git(self, *args: str) -> str:
        identity = ["-c", "user.name=cleanroom-test", "-c", "user.email=test@example.invalid"]
        return subprocess.run(
            ["git", *identity, *args], cwd=self.root, capture_output=True, text=True, check=True
        ).stdout

    def commit(self, *paths: str | Path) -> None:
        self.git("add", *[str(p) for p in paths])
        self.git("commit", "-q", "-m", "test commit")


@pytest.fixture
def git_repo(tmp_path: Path) -> Repo:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    root = tmp_path / "repo"
    root.mkdir()
    return Repo(root)
