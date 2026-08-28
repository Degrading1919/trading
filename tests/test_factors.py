from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingrank.config import ScreeningConfig
from swingrank.factors import (
    add_leave_one_out_sector_momentum,
    calculate_price_snapshot,
    calculate_standardized_unexpected_earnings,
)


def test_price_factor_formulas_use_exact_shared_session_lags() -> None:
    config = ScreeningConfig()
    sessions = pd.bdate_range("2024-01-01", periods=253)
    time = np.arange(253, dtype=float)
    tri = pd.Series(
        100.0 * np.exp(time * 0.001 + 0.005 * np.sin(time / 10.0)),
        index=sessions,
    )
    split_close = pd.Series(np.linspace(10.0, 20.0, 253), index=sessions)
    split_close.iloc[-50] = 30.0
    frame = pd.DataFrame(
        {
            "session_date": sessions,
            "close": split_close.to_numpy(),
            "split_adjusted_close": split_close.to_numpy(),
            "total_return_index": tri.to_numpy(),
            "volume": 1_000_000.0,
        }
    )

    result = calculate_price_snapshot(frame, sessions, config)

    assert result["momentum_12_7"] == pytest.approx(
        tri.iloc[-127] / tri.iloc[-253] - 1.0
    )
    assert result["momentum_6_1"] == pytest.approx(tri.iloc[-22] / tri.iloc[-127] - 1.0)
    assert result["high_52w_proximity"] == pytest.approx(20.0 / 30.0)
    expected_low_volatility = -(
        np.log(tri).diff().iloc[-63:].std(ddof=1) * np.sqrt(252.0)
    )
    assert result["low_volatility"] == pytest.approx(expected_low_volatility)
    assert result["median_dollar_volume_20d"] == pytest.approx(
        split_close.iloc[-20:].median() * 1_000_000.0
    )
    assert result["history_sessions"] == 253
    assert result["history_coverage"] == pytest.approx(1.0)


def test_sector_factor_is_market_cap_weighted_and_leave_one_out() -> None:
    config = ScreeningConfig(minimum_sector_members=5)
    frame = pd.DataFrame(
        {
            "sector": ["A"] * 5,
            "market_cap": [1.0] * 5,
            "momentum_12_7": [1.0, 2.0, 3.0, 4.0, 5.0],
            "momentum_6_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = add_leave_one_out_sector_momentum(frame, config)
    assert result.iloc[0] == pytest.approx(3.5)
    assert result.iloc[-1] == pytest.approx(2.5)


def test_sue_matches_seasonal_random_walk_standardization() -> None:
    config = ScreeningConfig()
    periods = pd.period_range(end="2024Q3", periods=13, freq="Q")
    fiscal_ends = periods.to_timestamp(how="end").normalize()
    seasonal_changes = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0])
    eps = [1.0, 1.1, 1.2, 1.3]
    for change in seasonal_changes:
        eps.append(eps[-4] + float(change))
    events = pd.DataFrame(
        {
            "fiscal_period_end": fiscal_ends,
            "reported_eps": eps,
            "announced_at": [
                end.tz_localize("UTC") + pd.Timedelta(days=35) for end in fiscal_ends
            ],
        }
    )
    as_of = pd.Timestamp("2025-01-01T00:00:00Z")
    expected = (10.0 - np.mean(np.arange(8.0))) / np.std(np.arange(8.0), ddof=1)
    assert calculate_standardized_unexpected_earnings(
        events, as_of, config
    ) == pytest.approx(expected)


def test_sue_rejects_a_missing_fiscal_quarter() -> None:
    config = ScreeningConfig()
    periods = pd.period_range(end="2024Q3", periods=14, freq="Q").delete(5)
    fiscal_ends = periods.to_timestamp(how="end").normalize()
    events = pd.DataFrame(
        {
            "fiscal_period_end": fiscal_ends,
            "reported_eps": np.linspace(1.0, 3.0, len(fiscal_ends)),
            "announced_at": [
                end.tz_localize("UTC") + pd.Timedelta(days=35) for end in fiscal_ends
            ],
        }
    )
    result = calculate_standardized_unexpected_earnings(
        events, pd.Timestamp("2025-01-01T00:00:00Z"), config
    )
    assert np.isnan(result)
