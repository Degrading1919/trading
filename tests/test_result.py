from __future__ import annotations

import json

from swingrank.engine import ScreeningEngine
from swingrank.providers.csv import CsvDataProvider
from swingrank.providers.memory import InMemoryDataProvider
from tests.conftest import AS_OF, make_earnings, make_prices, make_universe


def test_result_writes_all_audit_artifacts(
    complete_provider: InMemoryDataProvider, tmp_path
) -> None:
    result = ScreeningEngine(complete_provider).run(AS_OF)
    paths = result.write(tmp_path, "replay")
    assert all(path.exists() for path in paths.values())
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["run_fingerprint"] == result.manifest["run_fingerprint"]
    assert manifest["row_counts"]["candidates"] == len(result.candidates)


def test_csv_adapter_runs_the_same_point_in_time_pipeline(tmp_path) -> None:
    symbols = [f"CSV{index}" for index in range(6)]
    make_universe(symbols).to_csv(tmp_path / "universe.csv", index=False)
    make_prices(symbols).to_csv(tmp_path / "prices.csv", index=False)
    make_earnings(symbols).to_csv(tmp_path / "earnings.csv", index=False)

    result = ScreeningEngine(CsvDataProvider(tmp_path)).run(AS_OF)

    assert set(result.candidates["symbol"]) == set(symbols)
    assert result.manifest["provider"] == "csv"
    assert result.manifest["discarded_future_rows"] == {
        "universe": 0,
        "prices": 0,
        "earnings": 0,
    }
