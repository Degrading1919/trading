from __future__ import annotations

from swingrank.cli import main
from tests.conftest import AS_OF, make_earnings, make_prices, make_universe


def _write_csv_fixture(directory) -> list[str]:
    symbols = [f"CLI{index}" for index in range(6)]
    make_universe(symbols).to_csv(directory / "universe.csv", index=False)
    make_prices(symbols).to_csv(directory / "prices.csv", index=False)
    make_earnings(symbols).to_csv(directory / "earnings.csv", index=False)
    return symbols


def test_cli_runs_screen_and_writes_artifacts(tmp_path, capsys) -> None:
    data_directory = tmp_path / "data"
    output_directory = tmp_path / "output"
    data_directory.mkdir()
    _write_csv_fixture(data_directory)

    exit_code = main(
        [
            "--data-dir",
            str(data_directory),
            "--as-of",
            AS_OF.isoformat(),
            "--output-dir",
            str(output_directory),
            "--prefix",
            "daily",
            "--top",
            "3",
        ]
    )

    assert exit_code == 0
    assert (output_directory / "daily.candidates.csv").exists()
    assert (output_directory / "daily.exclusions.csv").exists()
    assert (output_directory / "daily.manifest.json").exists()
    output = capsys.readouterr().out
    assert "Run fingerprint:" in output


def test_cli_rejects_naive_as_of(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--data-dir",
            str(tmp_path),
            "--as-of",
            "2025-01-01",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert exit_code == 2
    assert "must include a timezone" in capsys.readouterr().err
