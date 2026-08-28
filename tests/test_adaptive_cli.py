from __future__ import annotations

from swingrank.adaptive.cli import main
from swingrank.engine import ScreeningEngine
from swingrank.providers.memory import InMemoryDataProvider
from tests.adaptive_helpers import make_training_panel
from tests.conftest import AS_OF, make_earnings, make_prices, make_universe


def test_adaptive_cli_writes_side_by_side_artifacts(tmp_path, symbols) -> None:
    data_directory = tmp_path / "data"
    output_directory = tmp_path / "output"
    data_directory.mkdir()
    universe = make_universe(symbols)
    prices = make_prices(symbols)
    earnings = make_earnings(symbols)
    universe.to_csv(data_directory / "universe.csv", index=False)
    prices.to_csv(data_directory / "prices.csv", index=False)
    earnings.to_csv(data_directory / "earnings.csv", index=False)

    baseline = ScreeningEngine(
        InMemoryDataProvider(universe, prices, earnings)
    ).run(AS_OF)
    training_path = tmp_path / "training.csv"
    make_training_panel(
        baseline, cohort_count=52, start="2020-01-01"
    ).to_csv(training_path, index=False)

    exit_code = main(
        [
            "--data-dir",
            str(data_directory),
            "--training-csv",
            str(training_path),
            "--as-of",
            AS_OF.isoformat(),
            "--output-dir",
            str(output_directory),
            "--top",
            "3",
        ]
    )

    assert exit_code == 0
    assert (output_directory / "screen.adaptive.candidates.csv").exists()
    assert (output_directory / "screen.adaptive.weights.csv").exists()
    assert (output_directory / "screen.baseline.candidates.csv").exists()


def test_adaptive_cli_rejects_naive_as_of(tmp_path) -> None:
    assert (
        main(
            [
                "--data-dir",
                str(tmp_path),
                "--training-csv",
                str(tmp_path / "training.csv"),
                "--as-of",
                "2025-01-01",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 2
    )
