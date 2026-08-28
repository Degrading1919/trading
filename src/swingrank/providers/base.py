"""Provider boundary.

Providers return raw, versioned observations.  They must not calculate factors;
all methodology lives in :mod:`swingrank.factors` and :mod:`swingrank.engine`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataProvider(Protocol):
    """Point-in-time data source contract."""

    @property
    def name(self) -> str:
        """Stable provider identifier included in audit manifests."""

    def universe(self, as_of: datetime) -> pd.DataFrame:
        """Return versioned security-master observations known by ``as_of``.

        The engine independently enforces timestamps, so an adapter may return a
        superset for efficient bulk loading.
        """

    def prices(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        as_of: datetime,
    ) -> pd.DataFrame:
        """Return versioned daily price/volume observations."""

    def earnings(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        as_of: datetime,
    ) -> pd.DataFrame:
        """Return versioned, originally reported quarterly diluted EPS."""
