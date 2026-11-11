import cleanroom

import tells
from tells.cli import main


def test_packages_import() -> None:
    assert tells.__version__
    assert cleanroom.__version__


def test_cli_version(capsys) -> None:
    assert main(["--version"]) == 0
    out = capsys.readouterr().out
    assert "tells" in out and "cleanroom" in out
