"""The Cleanroom is shared across projects, so it must not know about any of them."""

import re
from pathlib import Path

import cleanroom

SRC = Path(cleanroom.__file__).parent

# Words that would mean domain logic leaked into the shared package.
DOMAIN_WORDS = ["fraud", "launder", "loan", "credit", "payment", "merchant", "card", "tells"]


def _source_files() -> list[Path]:
    files = sorted(SRC.rglob("*.py"))
    assert files, "no cleanroom source files found"
    return files


def test_has_version() -> None:
    assert cleanroom.__version__


def test_no_project_imports() -> None:
    pattern = re.compile(r"^\s*(from|import)\s+(tells|reserve|culprit)\b", re.MULTILINE)
    for path in _source_files():
        assert not pattern.search(path.read_text(encoding="utf-8")), path


def test_no_domain_vocabulary() -> None:
    for path in _source_files():
        text = path.read_text(encoding="utf-8").lower()
        # The package docstring names the banned domains on purpose.
        if path.name == "__init__.py":
            text = text.split('"""', 2)[-1]
        for word in DOMAIN_WORDS:
            assert not re.search(rf"\b{word}", text), f"{path.name} mentions '{word}'"
