"""Versioned, deliberately small configuration for adaptive ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from swingrank.exceptions import ConfigurationError

ADAPTIVE_METHODOLOGY_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class AdaptiveConfig:
    """Ex ante choices for the first adaptive model.

    Defaults are research/engineering priors, not values selected by maximizing
    historical trading returns.  The window is rolling, so observations outside
    the most recent five calendar years never affect a fit.
    """

    methodology_version: str = ADAPTIVE_METHODOLOGY_VERSION
    horizon_sessions: int = 42
    training_window_calendar_days: int = 1_826
    minimum_training_cohorts: int = 24
    minimum_training_samples: int = 500
    minimum_class_samples: int = 25
    l2_penalty: float = 0.10
    maximum_iterations: int = 100
    convergence_tolerance: float = 1e-9
    probability_clip: float = 1e-9

    def __post_init__(self) -> None:
        if not self.methodology_version.strip():
            raise ConfigurationError("adaptive methodology_version cannot be blank")
        if self.horizon_sessions < 1:
            raise ConfigurationError("horizon_sessions must be positive")
        if self.training_window_calendar_days < 1:
            raise ConfigurationError(
                "training_window_calendar_days must be positive"
            )
        if self.minimum_training_cohorts < 2:
            raise ConfigurationError("minimum_training_cohorts must be at least 2")
        if self.minimum_training_samples < 2:
            raise ConfigurationError("minimum_training_samples must be at least 2")
        if self.minimum_class_samples < 1:
            raise ConfigurationError("minimum_class_samples must be positive")
        if self.l2_penalty <= 0:
            raise ConfigurationError("l2_penalty must be strictly positive")
        if self.maximum_iterations < 1:
            raise ConfigurationError("maximum_iterations must be positive")
        if not 0 < self.convergence_tolerance < 1:
            raise ConfigurationError("convergence_tolerance must be in (0, 1)")
        if not 0 < self.probability_clip < 0.5:
            raise ConfigurationError("probability_clip must be in (0, 0.5)")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for audit manifests."""

        return asdict(self)
