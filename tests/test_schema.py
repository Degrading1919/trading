from __future__ import annotations

import pandas as pd
import pytest

from swingrank.exceptions import DataValidationError
from swingrank.schema import prepare_prices, prepare_universe
from tests.conftest import AS_OF, make_universe


def test_price_revision_available_after_as_of_is_discarded() -> None:
    rows = pd.DataFrame(
        [
            {
                "symbol": "ABC",
                "session_date": "2024-12-31",
                "close": 10.0,
                "split_adjusted_close": 10.0,
                "total_return_index": 100.0,
                "volume": 1_000.0,
                "available_at": "2024-12-31T22:00:00Z",
            },
            {
                "symbol": "ABC",
                "session_date": "2024-12-31",
                "close": 999.0,
                "split_adjusted_close": 999.0,
                "total_return_index": 9_999.0,
                "volume": 1_000.0,
                "available_at": "2025-01-02T00:00:00Z",
            },
        ]
    )
    prepared = prepare_prices(rows, AS_OF)
    assert prepared.discarded_future_rows == 1
    assert prepared.frame.iloc[0]["close"] == 10.0


def test_latest_known_universe_version_is_selected_before_activity_test() -> None:
    universe = make_universe(["ABC"])
    inactive = universe.iloc[0].copy()
    inactive["available_at"] = AS_OF - pd.Timedelta(hours=1)
    inactive["effective_to"] = "2024-12-31"
    universe = pd.concat([universe, inactive.to_frame().T], ignore_index=True)
    prepared = prepare_universe(universe, AS_OF)
    assert len(prepared.frame) == 1
    assert prepared.frame.iloc[0]["effective_to"] == pd.Timestamp("2024-12-31")


def test_naive_as_of_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="timezone-aware"):
        prepare_universe(make_universe(["ABC"]), pd.Timestamp("2025-01-01"))


def test_naive_observation_timestamp_is_rejected() -> None:
    universe = make_universe(["ABC"])
    universe["available_at"] = "2024-12-31 20:00:00"
    with pytest.raises(DataValidationError, match="explicit UTC offset"):
        prepare_universe(universe, AS_OF)


def test_session_date_uses_market_timezone_not_utc_rollover() -> None:
    row = pd.DataFrame(
        [
            {
                "symbol": "ABC",
                "session_date": "2025-01-01",
                "close": 10.0,
                "split_adjusted_close": 10.0,
                "total_return_index": 100.0,
                "volume": 1_000.0,
                "available_at": "2025-01-01T00:00:00Z",
            }
        ]
    )
    prepared = prepare_prices(row, pd.Timestamp("2025-01-01T00:30:00Z"))
    assert prepared.frame.empty
    assert prepared.discarded_future_rows == 1


def test_duplicate_price_versions_are_rejected_as_nondeterministic() -> None:
    row = {
        "symbol": "ABC",
        "session_date": "2024-12-31",
        "close": 10.0,
        "split_adjusted_close": 10.0,
        "total_return_index": 100.0,
        "volume": 1_000.0,
        "available_at": "2024-12-31T22:00:00Z",
    }
    with pytest.raises(DataValidationError, match="duplicate"):
        prepare_prices(pd.DataFrame([row, row]), AS_OF)
