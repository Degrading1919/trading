"""Deterministic cross-sectional normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def centered_percentile_rank(
    values: pd.Series,
    *,
    minimum_population: int,
) -> pd.Series:
    """Map available observations to ``[-1, 1]`` using average ranks.

    Rank normalization is monotonic, robust to outliers, and explicit about ties.
    Missing observations remain missing.  A completely tied cross-section maps to
    zero, which is the neutral score.
    """

    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    available = numeric.dropna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    count = len(available)
    if count < minimum_population:
        return result
    if available.nunique(dropna=True) == 1:
        result.loc[available.index] = 0.0
        return result
    ranks = available.rank(method="average", ascending=True)
    result.loc[available.index] = 2.0 * (ranks - 1.0) / (count - 1.0) - 1.0
    return result
