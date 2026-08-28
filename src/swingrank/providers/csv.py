"""Reference adapter for provider exports stored as CSV files."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class CsvDataProvider:
    """Load ``universe.csv``, ``prices.csv``, and optional ``earnings.csv``.

    This adapter favors auditability over scale.  A production feed can replace
    it without changing factor or ranking code.
    """

    data_directory: Path | str
    provider_name: str = "csv"

    def __post_init__(self) -> None:
        self.data_directory = Path(self.data_directory)

    @property
    def name(self) -> str:
        return self.provider_name

    def _read(self, name: str, *, optional: bool = False) -> pd.DataFrame:
        path = self.data_directory / name
        if optional and not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)

    def universe(self, as_of: datetime) -> pd.DataFrame:
        return self._read("universe.csv")

    def prices(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        as_of: datetime,
    ) -> pd.DataFrame:
        # Filtering is intentionally centralized in the engine's schema layer.
        return self._read("prices.csv")

    def earnings(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        as_of: datetime,
    ) -> pd.DataFrame:
        return self._read("earnings.csv", optional=True)
