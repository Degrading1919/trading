from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingrank.normalization import centered_percentile_rank


def test_centered_rank_has_documented_endpoints_ties_and_missing_values() -> None:
    values = pd.Series([10.0, 20.0, 20.0, 40.0, np.nan], index=list("abcde"))
    result = centered_percentile_rank(values, minimum_population=3)
    assert result["a"] == pytest.approx(-1.0)
    assert result["b"] == pytest.approx(0.0)
    assert result["c"] == pytest.approx(0.0)
    assert result["d"] == pytest.approx(1.0)
    assert np.isnan(result["e"])


def test_all_ties_are_neutral() -> None:
    result = centered_percentile_rank(pd.Series([5.0, 5.0, 5.0]), minimum_population=3)
    assert result.tolist() == [0.0, 0.0, 0.0]


def test_too_small_a_cross_section_is_unavailable() -> None:
    result = centered_percentile_rank(pd.Series([1.0, 2.0]), minimum_population=3)
    assert result.isna().all()
