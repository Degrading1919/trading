from __future__ import annotations

import pytest

from swingrank.config import ScreeningConfig
from swingrank.exceptions import ConfigurationError


def test_default_weights_sum_to_one_and_required_histories_are_explicit() -> None:
    config = ScreeningConfig()
    assert sum(config.weights.values()) == pytest.approx(1.0)
    assert config.required_price_sessions == 253
    assert config.required_earnings_quarters == 13


def test_invalid_weights_are_rejected() -> None:
    with pytest.raises(ConfigurationError, match=r"sum to 1\.0"):
        ScreeningConfig(
            factor_weights=(
                ("momentum_12_7", 0.10),
                ("momentum_6_1", 0.10),
                ("high_52w_proximity", 0.10),
                ("sector_momentum", 0.10),
                ("earnings_sue", 0.10),
                ("low_volatility", 0.10),
            )
        )


def test_invalid_market_timezone_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="valid IANA zone"):
        ScreeningConfig(market_timezone="Not/A_Real_Zone")
