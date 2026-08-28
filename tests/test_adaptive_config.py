from __future__ import annotations

import pytest

from swingrank.adaptive.config import AdaptiveConfig
from swingrank.exceptions import ConfigurationError


def test_adaptive_defaults_are_explicit_regularized_research_priors() -> None:
    config = AdaptiveConfig()
    assert config.horizon_sessions == 42
    assert config.training_window_calendar_days == 1_826
    assert config.l2_penalty > 0
    assert config.as_dict()["methodology_version"] == "1.0.0"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizon_sessions", 0),
        ("training_window_calendar_days", 0),
        ("minimum_training_cohorts", 1),
        ("l2_penalty", 0.0),
        ("probability_clip", 0.5),
    ],
)
def test_invalid_adaptive_configuration_fails_closed(field: str, value) -> None:
    with pytest.raises(ConfigurationError):
        AdaptiveConfig(**{field: value})
