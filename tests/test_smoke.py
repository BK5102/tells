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


def test_cli_run_prints_a_summary_with_dates(monkeypatch, capsys, tmp_path) -> None:
    """YAML split dates load as datetime.date; printing the summary must not crash."""
    import datetime

    import tells.config
    import tells.harness

    summary = {"config": {"split": {"train": {"end": datetime.date(2015, 10, 1)}}}}
    monkeypatch.setattr(tells.config, "load_config", lambda path: object())
    monkeypatch.setattr(
        tells.harness,
        "run_experiment",
        lambda cfg, part: tells.harness.RunResult(tmp_path, summary),
    )
    assert main(["run", "configs/small.yaml"]) == 0
    assert "2015-10-01" in capsys.readouterr().out
