from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingrank.providers.memory import InMemoryDataProvider

AS_OF = pd.Timestamp("2025-01-01T00:00:00Z")


def make_universe(
    symbols: list[str],
    *,
    as_of: pd.Timestamp = AS_OF,
    sector: str = "Technology",
) -> pd.DataFrame:
    rows = []
    for index, symbol in enumerate(symbols):
        rows.append(
            {
                "symbol": symbol,
                "security_type": "COMMON_STOCK",
                "country": "US",
                "exchange": "NASDAQ",
                "currency": "USD",
                "sector": sector,
                "industry": "Software",
                "market_cap": 1_000_000_000.0 + index * 100_000_000.0,
                "is_primary_listing": True,
                "effective_from": "2000-01-01",
                "effective_to": None,
                "available_at": as_of - pd.Timedelta(days=1),
            }
        )
    return pd.DataFrame(rows)


def make_prices(
    symbols: list[str],
    *,
    periods: int = 260,
    end: str = "2024-12-31",
) -> pd.DataFrame:
    sessions = pd.bdate_range(end=end, periods=periods)
    rows = []
    time = np.arange(periods, dtype=float)
    for index, symbol in enumerate(symbols):
        daily_log_return = (
            0.00015
            + index * 0.000035
            + 0.0007 * np.sin(time / (7.0 + index * 0.2) + index)
        )
        tri = 100.0 * np.exp(np.cumsum(daily_log_return))
        close = 20.0 * tri / tri[0]
        for position, session in enumerate(sessions):
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": session,
                    "close": close[position],
                    "split_adjusted_close": close[position],
                    "total_return_index": tri[position],
                    "volume": 1_000_000.0 + index * 10_000.0,
                    "available_at": session.tz_localize("UTC") + pd.Timedelta(hours=22),
                }
            )
    return pd.DataFrame(rows)


def make_earnings(
    symbols: list[str],
    *,
    current_delta_offset: float = 0.0,
) -> pd.DataFrame:
    periods = pd.period_range(end="2024Q3", periods=13, freq="Q")
    fiscal_ends = periods.to_timestamp(how="end").normalize()
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        seasonal_changes = np.array(
            [0.0, 0.1, 0.3, 0.2, 0.6, 0.4, 0.7, 0.8, 1.2], dtype=float
        )
        seasonal_changes[-1] += current_delta_offset + symbol_index * 0.05
        eps = [1.0, 1.1, 1.2, 1.3]
        for change in seasonal_changes:
            eps.append(eps[-4] + float(change))
        for fiscal_end, reported_eps in zip(fiscal_ends, eps, strict=True):
            announced_at = fiscal_end.tz_localize("UTC") + pd.Timedelta(
                days=35, hours=13
            )
            rows.append(
                {
                    "symbol": symbol,
                    "fiscal_period_end": fiscal_end,
                    "reported_eps": reported_eps,
                    "announced_at": announced_at,
                    "available_at": announced_at + pd.Timedelta(minutes=2),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def symbols() -> list[str]:
    return [f"S{index:02d}" for index in range(10)]


@pytest.fixture
def complete_provider(symbols: list[str]) -> InMemoryDataProvider:
    return InMemoryDataProvider(
        make_universe(symbols),
        make_prices(symbols),
        make_earnings(symbols),
    )
