from __future__ import annotations

import pandas as pd
import pytest

from swingrank.adaptive.training import build_training_cohort
from swingrank.engine import ScreeningEngine
from swingrank.exceptions import DataValidationError
from swingrank.providers.memory import InMemoryDataProvider
from tests.conftest import AS_OF


def test_training_cohort_is_built_from_exact_baseline_features(
    complete_provider: InMemoryDataProvider,
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    outcomes = pd.DataFrame(
        {
            "symbol": baseline.candidates["symbol"],
            "forward_return": pd.Series(range(len(baseline.candidates))) / 100.0,
        }
    )
    cohort = build_training_cohort(
        baseline,
        outcomes,
        label_end_session="2025-03-03",
        label_end_at=pd.Timestamp("2025-03-03T22:00:00Z"),
        available_at=pd.Timestamp("2025-03-03T22:05:00Z"),
    )

    assert len(cohort) == len(baseline.candidates)
    assert (cohort["formation_universe_size"] == len(cohort)).all()
    assert cohort["baseline_run_fingerprint"].nunique() == 1
    assert cohort["horizon_sessions"].eq(42).all()


def test_training_cohort_refuses_missing_delisted_outcome(
    complete_provider: InMemoryDataProvider,
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    outcomes = pd.DataFrame(
        {
            "symbol": baseline.candidates["symbol"].iloc[:-1],
            "forward_return": 0.01,
        }
    )

    with pytest.raises(DataValidationError, match="exact formation universe"):
        build_training_cohort(
            baseline,
            outcomes,
            label_end_session="2025-03-03",
            label_end_at=pd.Timestamp("2025-03-03T22:00:00Z"),
            available_at=pd.Timestamp("2025-03-03T22:05:00Z"),
        )
