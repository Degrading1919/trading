"""Pure factor calculations; no provider-specific logic lives here."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from swingrank.config import ScreeningConfig

RAW_FACTOR_NAMES = (
    "momentum_12_7",
    "momentum_6_1",
    "high_52w_proximity",
    "sector_momentum",
    "earnings_sue",
    "low_volatility",
)


def _at_lag(series: pd.Series, lag: int) -> float:
    return float(series.iloc[-(lag + 1)])


def calculate_price_snapshot(
    symbol_prices: pd.DataFrame,
    reference_sessions: pd.DatetimeIndex,
    config: ScreeningConfig,
) -> dict[str, float | int | pd.Timestamp]:
    """Calculate price factors and eligibility diagnostics for one security.

    All lags refer to the shared market-session index, never to a security's own
    possibly sparse observations.
    """

    indexed = symbol_prices.set_index("session_date").sort_index()
    aligned = indexed.reindex(reference_sessions)
    required = aligned.iloc[-config.required_price_sessions :]
    recent_required = aligned.iloc[
        -(max(config.volatility_sessions + 1, config.liquidity_sessions)) :
    ]

    coverage = float(
        required[["total_return_index", "split_adjusted_close"]]
        .notna()
        .all(axis=1)
        .mean()
    )
    latest_session = indexed.index.max() if not indexed.empty else pd.NaT
    anchor_lags = (
        0,
        config.momentum_6_1_end_lag,
        config.momentum_6_1_start_lag,
        config.momentum_12_7_start_lag,
    )
    anchors_complete = all(
        len(required) > lag
        and pd.notna(required["total_return_index"].iloc[-(lag + 1)])
        for lag in anchor_lags
    )
    recent_complete = bool(
        recent_required[
            ["close", "split_adjusted_close", "total_return_index", "volume"]
        ]
        .notna()
        .all()
        .all()
    )

    snapshot: dict[str, float | int | pd.Timestamp] = {
        "history_sessions": int(required["total_return_index"].notna().sum()),
        "history_coverage": coverage,
        "latest_session": latest_session,
        "anchors_complete": float(anchors_complete),
        "recent_history_complete": float(recent_complete),
        "latest_price": np.nan,
        "median_dollar_volume_20d": np.nan,
        "nonzero_volume_fraction_20d": np.nan,
        "momentum_12_7": np.nan,
        "momentum_6_1": np.nan,
        "high_52w_proximity": np.nan,
        "low_volatility": np.nan,
    }
    if required.empty or not anchors_complete or not recent_complete:
        return snapshot

    tri = required["total_return_index"].astype(float)
    split_close = required["split_adjusted_close"].astype(float)
    current_close = float(required["close"].iloc[-1])
    liquidity = required.iloc[-config.liquidity_sessions :]
    dollar_volume = liquidity["close"].astype(float) * liquidity["volume"].astype(float)
    daily_log_returns = np.log(tri).diff().iloc[-config.volatility_sessions :]

    start_12_7 = _at_lag(tri, config.momentum_12_7_start_lag)
    end_12_7 = _at_lag(tri, config.momentum_12_7_end_lag)
    start_6_1 = _at_lag(tri, config.momentum_6_1_start_lag)
    end_6_1 = _at_lag(tri, config.momentum_6_1_end_lag)
    high_window = split_close.iloc[-config.high_52w_sessions :]
    realized_volatility = float(daily_log_returns.std(ddof=1) * np.sqrt(252.0))

    snapshot.update(
        {
            "latest_price": current_close,
            "median_dollar_volume_20d": float(dollar_volume.median()),
            "nonzero_volume_fraction_20d": float((liquidity["volume"] > 0).mean()),
            "momentum_12_7": end_12_7 / start_12_7 - 1.0,
            "momentum_6_1": end_6_1 / start_6_1 - 1.0,
            "high_52w_proximity": float(split_close.iloc[-1] / high_window.max()),
            # Higher values must always be better before normalization.
            "low_volatility": -realized_volatility,
        }
    )
    return snapshot


def add_leave_one_out_sector_momentum(
    factors: pd.DataFrame,
    config: ScreeningConfig,
) -> pd.Series:
    """Current-market-cap-weighted peer momentum, excluding the stock itself."""

    result = pd.Series(np.nan, index=factors.index, dtype=float)
    composite = 0.5 * factors["momentum_12_7"] + 0.5 * factors["momentum_6_1"]
    for sector, positions in factors.groupby("sector", sort=True).groups.items():
        if not isinstance(sector, str) or not sector.strip():
            continue
        positions = list(positions)
        if len(positions) < config.minimum_sector_members:
            continue
        group = factors.loc[positions]
        group_returns = composite.loc[positions]
        weights = group["market_cap"].astype(float)
        valid = group_returns.notna() & weights.notna() & (weights > 0)
        if int(valid.sum()) < config.minimum_sector_members:
            continue
        weighted_sum = float((group_returns.loc[valid] * weights.loc[valid]).sum())
        total_weight = float(weights.loc[valid].sum())
        for position in group.index[valid]:
            peer_weight = total_weight - float(weights.loc[position])
            if peer_weight <= 0:
                continue
            result.loc[position] = (
                weighted_sum
                - float(group_returns.loc[position]) * float(weights.loc[position])
            ) / peer_weight
    return result


def calculate_standardized_unexpected_earnings(
    events: pd.DataFrame,
    as_of: datetime,
    config: ScreeningConfig,
) -> float:
    """Calculate seasonal-random-walk SUE from originally reported EPS.

    The latest seasonal EPS change is centered and scaled using the preceding
    eight seasonal changes by default.  Fiscal-period observations must be
    consecutive calendar quarters to prevent a missing quarter from being used
    as the seasonal lag.
    """

    if events.empty:
        return np.nan
    ordered = events.sort_values("fiscal_period_end", kind="mergesort").copy()
    latest = ordered.iloc[-1]
    age = pd.Timestamp(as_of).tz_convert("UTC") - pd.Timestamp(latest["announced_at"])
    if age < pd.Timedelta(0) or age > pd.Timedelta(days=config.earnings_max_age_days):
        return np.nan

    required = config.required_earnings_quarters
    if len(ordered) < required:
        return np.nan
    sample = ordered.iloc[-required:].copy()
    periods = pd.PeriodIndex(sample["fiscal_period_end"], freq="Q")
    if not np.all(np.diff(periods.asi8) == 1):
        return np.nan

    eps = sample["reported_eps"].astype(float).to_numpy()
    lag = config.sue_seasonal_lag_quarters
    seasonal_changes = eps[lag:] - eps[:-lag]
    current_change = float(seasonal_changes[-1])
    baseline = seasonal_changes[-(config.sue_baseline_changes + 1) : -1]
    if len(baseline) != config.sue_baseline_changes:
        return np.nan
    baseline_std = float(np.std(baseline, ddof=1))
    if not np.isfinite(baseline_std) or baseline_std <= config.sue_minimum_std:
        return np.nan
    return (current_change - float(np.mean(baseline))) / baseline_std


def calculate_earnings_factors(
    earnings: pd.DataFrame,
    symbols: pd.Series,
    as_of: datetime,
    config: ScreeningConfig,
) -> pd.Series:
    """Calculate SUE for each symbol without changing input ordering."""

    result = pd.Series(np.nan, index=symbols.index, dtype=float)
    if earnings.empty:
        return result
    groups = {symbol: group for symbol, group in earnings.groupby("symbol", sort=True)}
    for position, symbol in symbols.items():
        group = groups.get(symbol)
        if group is not None:
            result.loc[position] = calculate_standardized_unexpected_earnings(
                group, as_of, config
            )
    return result
