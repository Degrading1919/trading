"""Interfaces that let adaptive model families be exchanged safely."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ModelExplanation:
    """Per-row additive explanation in log-odds space."""

    probabilities: np.ndarray
    log_odds: np.ndarray
    base_log_odds: np.ndarray
    contributions: pd.DataFrame


@runtime_checkable
class FittedProbabilisticModel(Protocol):
    """Immutable fitted model contract consumed by the ranking engine."""

    @property
    def name(self) -> str:
        """Stable fitted-model identifier."""

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Features in the exact order expected by the model."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return probability of the positive label for every row."""

    def explain(self, features: pd.DataFrame) -> ModelExplanation:
        """Return probabilities and additive log-odds contributions."""

    def state(self) -> dict[str, Any]:
        """Return complete JSON-safe fitted state for auditing."""

    def learned_weights(self) -> pd.DataFrame:
        """Return a transparent feature-weight table."""


@runtime_checkable
class ProbabilisticModel(Protocol):
    """Stateless model specification that creates an immutable fitted model."""

    @property
    def name(self) -> str:
        """Stable model-family identifier."""

    def fit(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        sample_weight: np.ndarray,
    ) -> FittedProbabilisticModel:
        """Fit only the rows supplied by the point-in-time engine."""
