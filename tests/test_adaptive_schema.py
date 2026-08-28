from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from swingrank.adaptive.schema import prepare_training_samples
from swingrank.engine import ScreeningEngine
from swingrank.exceptions import DataValidationError
from swingrank.providers.memory import InMemoryDataProvider
from tests.adaptive_helpers import adaptive_test_config, make_training_panel
from tests.conftest import AS_OF


def test_future_training_revision_cannot_change_earlier_panel(
    complete_provider: InMemoryDataProvider,
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    panel = make_training_panel(baseline)
    prepared = prepare_training_samples(
        panel, AS_OF.to_pydatetime(), adaptive_test_config()
    )

    future_revision = panel.iloc[0].copy()
    future_revision["available_at"] = AS_OF + pd.Timedelta(days=1)
    future_revision["forward_return"] = -0.99
    augmented = pd.concat(
        [panel, future_revision.to_frame().T], ignore_index=True
    )
    prepared_augmented = prepare_training_samples(
        augmented, AS_OF.to_pydatetime(), adaptive_test_config()
    )

    pdt.assert_frame_equal(prepared.frame, prepared_augmented.frame)
    assert prepared_augmented.discarded_future_rows == 1


def test_known_revision_cannot_rewrite_frozen_historical_features(
    complete_provider: InMemoryDataProvider,
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    panel = make_training_panel(baseline)
    revision = panel.iloc[0].copy()
    revision["available_at"] = AS_OF - pd.Timedelta(minutes=1)
    revision["normalized_momentum_12_7"] = -revision[
        "normalized_momentum_12_7"
    ]

    with pytest.raises(DataValidationError, match="may change only forward_return"):
        prepare_training_samples(
            pd.concat([panel, revision.to_frame().T], ignore_index=True),
            AS_OF.to_pydatetime(),
            adaptive_test_config(),
        )


def test_incomplete_cohort_is_rejected_as_survivorship_risk(
    complete_provider: InMemoryDataProvider,
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    panel = make_training_panel(baseline)
    incomplete = panel.drop(panel.index[-1]).reset_index(drop=True)

    with pytest.raises(DataValidationError, match="survivorship bias"):
        prepare_training_samples(
            incomplete, AS_OF.to_pydatetime(), adaptive_test_config()
        )


def test_training_timestamps_must_be_timezone_aware(
    complete_provider: InMemoryDataProvider,
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    panel = make_training_panel(baseline)
    panel["formation_at"] = panel["formation_at"].dt.tz_localize(None)

    with pytest.raises(DataValidationError, match="explicit UTC offset"):
        prepare_training_samples(
            panel, AS_OF.to_pydatetime(), adaptive_test_config()
        )
