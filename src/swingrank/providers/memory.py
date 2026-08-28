"""In-memory provider useful for tests, notebooks, and service integration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd


@dataclass(slots=True)
class InMemoryDataProvider:
    """Thin provider that deliberately performs no point-in-time filtering."""

    universe_frame: pd.DataFrame
    price_frame: pd.DataFrame
    earnings_frame: pd.DataFrame | None = None
    provider_name: str = "memory"

    @property
    def name(self) -> str:
        return self.provider_name

    def universe(self, as_of: datetime) -> pd.DataFrame:
        return self.universe_frame.copy(deep=True)

    def prices(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        as_of: datetime,
    ) -> pd.DataFrame:
        return self.price_frame.copy(deep=True)

    def earnings(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        as_of: datetime,
    ) -> pd.DataFrame:
        if self.earnings_frame is None:
            return pd.DataFrame()
        return self.earnings_frame.copy(deep=True)
