"""Provider boundary for materialized historical baseline training cohorts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class TrainingDataProvider(Protocol):
    """Supply versioned training rows; the engine enforces time boundaries."""

    @property
    def name(self) -> str:
        """Stable provider identifier included in adaptive manifests."""

    def training_samples(self, as_of: datetime) -> pd.DataFrame:
        """Return a possibly unfiltered superset of historical training rows."""


@dataclass(slots=True)
class InMemoryTrainingDataProvider:
    """In-memory training provider for tests, notebooks, and services."""

    frame: pd.DataFrame
    provider_name: str = "memory-training"

    @property
    def name(self) -> str:
        return self.provider_name

    def training_samples(self, as_of: datetime) -> pd.DataFrame:
        return self.frame.copy(deep=True)


@dataclass(slots=True)
class CsvTrainingDataProvider:
    """Read a materialized training panel from one CSV file."""

    path: str | Path
    provider_name: str = "csv-training"

    @property
    def name(self) -> str:
        return self.provider_name

    def training_samples(self, as_of: datetime) -> pd.DataFrame:
        return pd.read_csv(Path(self.path))
