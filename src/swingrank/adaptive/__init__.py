"""Adaptive, point-in-time probabilistic ranking layer.

The fixed :class:`swingrank.engine.ScreeningEngine` remains the source of the
eligible universe and predictive features.  This package only learns how those
unchanged features map to a forward relative-outperformance probability.
"""

from swingrank.adaptive.config import AdaptiveConfig
from swingrank.adaptive.engine import AdaptiveRankingEngine
from swingrank.adaptive.providers import (
    CsvTrainingDataProvider,
    InMemoryTrainingDataProvider,
    TrainingDataProvider,
)
from swingrank.adaptive.result import AdaptiveScreeningResult
from swingrank.adaptive.training import build_training_cohort

__all__ = [
    "AdaptiveConfig",
    "AdaptiveRankingEngine",
    "AdaptiveScreeningResult",
    "CsvTrainingDataProvider",
    "InMemoryTrainingDataProvider",
    "TrainingDataProvider",
    "build_training_cohort",
]
