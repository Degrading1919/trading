"""Pluggable definitions of the probabilistic event being predicted."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class BinaryOutcomeLabeler(Protocol):
    """Convert complete cohort outcomes to deterministic binary labels."""

    @property
    def name(self) -> str:
        """Stable label definition identifier."""

    def labels(self, samples: pd.DataFrame) -> pd.Series:
        """Return a zero/one label aligned to ``samples.index``."""


class CrossSectionalMedianLabeler:
    """One when a stock beats its formation cohort's median total return."""

    @property
    def name(self) -> str:
        return "forward_return_above_formation_cohort_median"

    def labels(self, samples: pd.DataFrame) -> pd.Series:
        medians = samples.groupby("formation_at", sort=True)[
            "forward_return"
        ].transform("median")
        return samples["forward_return"].gt(medians).astype(int)
