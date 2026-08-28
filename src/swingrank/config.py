"""Versioned methodology configuration.

The defaults are research priors, not the result of a parameter search.  Every
field is serializable so a historical run can be reconstructed exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from swingrank.exceptions import ConfigurationError

METHODOLOGY_VERSION = "1.0.0"

DEFAULT_FACTOR_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("momentum_12_7", 0.20),
    ("momentum_6_1", 0.15),
    ("high_52w_proximity", 0.15),
    ("sector_momentum", 0.15),
    ("earnings_sue", 0.20),
    ("low_volatility", 0.15),
)


@dataclass(frozen=True, slots=True)
class ScreeningConfig:
    """All assumptions that can affect eligibility or rank."""

    methodology_version: str = METHODOLOGY_VERSION

    allowed_security_types: tuple[str, ...] = ("COMMON_STOCK",)
    allowed_countries: tuple[str, ...] = ("US", "USA")
    allowed_exchanges: tuple[str, ...] = (
        "NYSE",
        "NASDAQ",
        "NYSE_AMERICAN",
        "NYSE_ARCA",
    )
    required_currency: str = "USD"
    require_primary_listing: bool = True
    market_timezone: str = "America/New_York"

    minimum_price: float = 5.0
    minimum_market_cap: float = 300_000_000.0
    minimum_median_dollar_volume_20d: float = 5_000_000.0
    minimum_nonzero_volume_fraction_20d: float = 0.95
    minimum_history_coverage: float = 0.98

    momentum_12_7_start_lag: int = 252
    momentum_12_7_end_lag: int = 126
    momentum_6_1_start_lag: int = 126
    momentum_6_1_end_lag: int = 21
    high_52w_sessions: int = 252
    volatility_sessions: int = 63
    liquidity_sessions: int = 20

    minimum_sector_members: int = 5
    earnings_max_age_days: int = 120
    sue_seasonal_lag_quarters: int = 4
    sue_baseline_changes: int = 8
    sue_minimum_std: float = 1e-12

    minimum_factor_population: int = 3
    minimum_score_coverage: float = 0.65
    price_lookback_calendar_days: int = 550
    earnings_lookback_calendar_days: int = 2_200

    factor_weights: tuple[tuple[str, float], ...] = DEFAULT_FACTOR_WEIGHTS

    def __post_init__(self) -> None:
        weights = dict(self.factor_weights)
        if len(weights) != len(self.factor_weights):
            raise ConfigurationError("factor_weights contains duplicate factor names")
        if set(weights) != {name for name, _ in DEFAULT_FACTOR_WEIGHTS}:
            raise ConfigurationError(
                "factor_weights must contain exactly: "
                + ", ".join(name for name, _ in DEFAULT_FACTOR_WEIGHTS)
            )
        if any(weight < 0 for weight in weights.values()):
            raise ConfigurationError("factor weights cannot be negative")
        if not isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
            raise ConfigurationError("factor weights must sum to 1.0")

        if not (
            self.momentum_12_7_start_lag
            > self.momentum_12_7_end_lag
            == self.momentum_6_1_start_lag
            > self.momentum_6_1_end_lag
            >= 1
        ):
            raise ConfigurationError("momentum lags must define 12-7 and 6-1 windows")
        if self.high_52w_sessions < self.momentum_12_7_start_lag:
            raise ConfigurationError(
                "52-week history must cover the longest momentum lag"
            )
        if not 0 < self.minimum_history_coverage <= 1:
            raise ConfigurationError("minimum_history_coverage must be in (0, 1]")
        if not 0 <= self.minimum_nonzero_volume_fraction_20d <= 1:
            raise ConfigurationError(
                "minimum_nonzero_volume_fraction_20d must be in [0, 1]"
            )
        if not 0 <= self.minimum_score_coverage <= 1:
            raise ConfigurationError("minimum_score_coverage must be in [0, 1]")
        if self.minimum_sector_members < 2:
            raise ConfigurationError("minimum_sector_members must be at least 2")
        if self.minimum_factor_population < 2:
            raise ConfigurationError("minimum_factor_population must be at least 2")
        if self.volatility_sessions < 2:
            raise ConfigurationError("volatility_sessions must be at least 2")
        if self.liquidity_sessions < 1:
            raise ConfigurationError("liquidity_sessions must be positive")
        if self.sue_seasonal_lag_quarters < 1 or self.sue_baseline_changes < 2:
            raise ConfigurationError(
                "SUE requires a positive lag and at least two baselines"
            )
        if self.earnings_max_age_days < 0:
            raise ConfigurationError("earnings_max_age_days cannot be negative")
        if (
            self.price_lookback_calendar_days < 1
            or self.earnings_lookback_calendar_days < 1
        ):
            raise ConfigurationError("provider lookback windows must be positive")
        if (
            min(
                self.minimum_price,
                self.minimum_market_cap,
                self.minimum_median_dollar_volume_20d,
            )
            < 0
        ):
            raise ConfigurationError("eligibility thresholds cannot be negative")
        try:
            ZoneInfo(self.market_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ConfigurationError(
                "market_timezone must be a valid IANA zone"
            ) from exc

    @property
    def weights(self) -> dict[str, float]:
        """Return factor weights in their canonical order."""

        return dict(self.factor_weights)

    @property
    def required_price_sessions(self) -> int:
        """Number of market sessions needed, including the formation session."""

        return max(self.high_52w_sessions, self.momentum_12_7_start_lag) + 1

    @property
    def required_earnings_quarters(self) -> int:
        """Quarter count needed for current SUE plus its historical baseline."""

        return self.sue_seasonal_lag_quarters + self.sue_baseline_changes + 1

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation used in run manifests."""

        result = asdict(self)
        result["factor_weights"] = dict(self.factor_weights)
        for key in (
            "allowed_security_types",
            "allowed_countries",
            "allowed_exchanges",
        ):
            result[key] = list(result[key])
        return result
