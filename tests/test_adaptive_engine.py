from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from swingrank.adaptive.engine import AdaptiveRankingEngine
from swingrank.adaptive.models.base import ModelExplanation
from swingrank.adaptive.providers import InMemoryTrainingDataProvider
from swingrank.adaptive.schema import ADAPTIVE_FEATURE_COLUMNS
from swingrank.engine import ScreeningEngine
from swingrank.exceptions import DataValidationError
from swingrank.providers.memory import InMemoryDataProvider
from tests.adaptive_helpers import adaptive_test_config, make_training_panel
from tests.conftest import AS_OF


class _ConstantFittedModel:
    name = "constant_probability_test_model"
    feature_names = ADAPTIVE_FEATURE_COLUMNS

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), 0.5)

    def explain(self, features: pd.DataFrame) -> ModelExplanation:
        return ModelExplanation(
            probabilities=np.full(len(features), 0.5),
            log_odds=np.zeros(len(features)),
            base_log_odds=np.zeros(len(features)),
            contributions=pd.DataFrame(
                np.zeros((len(features), len(self.feature_names))),
                columns=self.feature_names,
                index=features.index,
            ),
        )

    def state(self) -> dict:
        return {"name": self.name, "probability": 0.5}

    def learned_weights(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient_log_odds": np.zeros(len(self.feature_names)),
            }
        )


class _ConstantModel:
    name = "constant_probability_test_model"

    def fit(self, features, labels, sample_weight) -> _ConstantFittedModel:
        return _ConstantFittedModel()


def _engine(
    market_provider: InMemoryDataProvider,
    panel: pd.DataFrame,
) -> AdaptiveRankingEngine:
    return AdaptiveRankingEngine(
        market_provider,
        InMemoryTrainingDataProvider(panel),
        adaptive_config=adaptive_test_config(),
    )


def test_adaptive_engine_keeps_fixed_baseline_and_explains_probability(
    complete_provider: InMemoryDataProvider,
) -> None:
    fixed_before = ScreeningEngine(complete_provider).run(AS_OF)
    panel = make_training_panel(fixed_before)
    result = _engine(complete_provider, panel).run(AS_OF)

    pdt.assert_frame_equal(result.baseline.candidates, fixed_before.candidates)
    assert result.baseline.manifest == fixed_before.manifest
    assert result.candidates["adaptive_rank"].tolist() == list(range(1, 11))
    assert result.candidates["probability_outperformance"].between(0, 1).all()
    contribution_columns = [
        f"adaptive_contribution_{feature.removeprefix('normalized_')}"
        for feature in ADAPTIVE_FEATURE_COLUMNS
    ]
    assert np.allclose(
        result.candidates["adaptive_base_log_odds"]
        + result.candidates[contribution_columns].sum(axis=1),
        result.candidates["adaptive_log_odds"],
        rtol=0.0,
        atol=1e-12,
    )
    assert set(result.learned_weights["factor"]) == {
        feature.removeprefix("normalized_")
        for feature in ADAPTIVE_FEATURE_COLUMNS
    }
    assert result.manifest["llm_in_prediction_path"] is False
    assert result.manifest["baseline"]["run_fingerprint"] == fixed_before.manifest[
        "run_fingerprint"
    ]


def test_future_training_data_cannot_change_weights_or_rankings(
    complete_provider: InMemoryDataProvider,
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    panel = make_training_panel(baseline)
    original = _engine(complete_provider, panel).run(AS_OF)

    future_revision = panel.iloc[0].copy()
    future_revision["available_at"] = AS_OF + pd.Timedelta(days=7)
    future_revision["forward_return"] = -0.99
    augmented_panel = pd.concat(
        [panel, future_revision.to_frame().T], ignore_index=True
    )

    future_universe = complete_provider.universe_frame.iloc[0].copy()
    future_universe["available_at"] = AS_OF + pd.Timedelta(days=1)
    future_universe["market_cap"] = 999_000_000_000.0
    future_price = complete_provider.price_frame.iloc[0].copy()
    future_price["available_at"] = AS_OF + pd.Timedelta(days=1)
    future_price["total_return_index"] = 1_000_000.0
    future_earnings = complete_provider.earnings_frame.iloc[-1].copy()
    future_earnings["available_at"] = AS_OF + pd.Timedelta(days=1)
    future_earnings["announced_at"] = AS_OF + pd.Timedelta(hours=12)
    future_earnings["reported_eps"] = 1_000.0
    augmented_market = InMemoryDataProvider(
        pd.concat(
            [
                complete_provider.universe_frame,
                future_universe.to_frame().T,
            ],
            ignore_index=True,
        ),
        pd.concat(
            [complete_provider.price_frame, future_price.to_frame().T],
            ignore_index=True,
        ),
        pd.concat(
            [complete_provider.earnings_frame, future_earnings.to_frame().T],
            ignore_index=True,
        ),
    )
    augmented = _engine(augmented_market, augmented_panel).run(AS_OF)

    pdt.assert_frame_equal(original.baseline.candidates, augmented.baseline.candidates)
    pdt.assert_frame_equal(original.candidates, augmented.candidates)
    pdt.assert_frame_equal(original.learned_weights, augmented.learned_weights)
    assert original.manifest["training_input_hash"] == augmented.manifest[
        "training_input_hash"
    ]
    assert original.manifest["run_fingerprint"] == augmented.manifest[
        "run_fingerprint"
    ]
    assert augmented.manifest["row_counts"]["training_discarded_future"] == 1


def test_rows_outside_rolling_window_cannot_change_fit(
    complete_provider: InMemoryDataProvider,
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    panel = make_training_panel(baseline)
    original = _engine(complete_provider, panel).run(AS_OF)
    old_cohort = make_training_panel(
        baseline, cohort_count=1, start="2010-01-01"
    )
    old_cohort["forward_return"] = -0.99

    augmented = _engine(
        complete_provider,
        pd.concat([panel, old_cohort], ignore_index=True),
    ).run(AS_OF)

    pdt.assert_frame_equal(original.candidates, augmented.candidates)
    pdt.assert_frame_equal(original.learned_weights, augmented.learned_weights)
    assert original.manifest["run_fingerprint"] == augmented.manifest[
        "run_fingerprint"
    ]
    assert augmented.manifest["row_counts"][
        "training_discarded_outside_window"
    ] == len(old_cohort)


def test_training_input_order_cannot_change_adaptive_result(
    complete_provider: InMemoryDataProvider,
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    panel = make_training_panel(baseline)
    ordered = _engine(complete_provider, panel).run(AS_OF)
    shuffled = _engine(
        complete_provider,
        panel.sample(frac=1.0, random_state=1729).reset_index(drop=True),
    ).run(AS_OF)

    pdt.assert_frame_equal(ordered.candidates, shuffled.candidates)
    pdt.assert_frame_equal(ordered.learned_weights, shuffled.learned_weights)
    assert ordered.manifest["run_fingerprint"] == shuffled.manifest[
        "run_fingerprint"
    ]


def test_walk_forward_refits_using_only_labels_available_at_each_date(
    complete_provider: InMemoryDataProvider,
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    panel = make_training_panel(baseline)
    future_cohort = make_training_panel(
        baseline, cohort_count=1, start="2024-11-01"
    )
    future_cohort["label_end_session"] = pd.Timestamp("2025-01-15")
    future_cohort["label_end_at"] = pd.Timestamp("2025-01-15T22:00:00Z")
    future_cohort["available_at"] = pd.Timestamp("2025-01-15T22:05:00Z")
    combined = pd.concat([panel, future_cohort], ignore_index=True)
    engine = _engine(complete_provider, combined)

    early, later = engine.run_walk_forward(
        [AS_OF, pd.Timestamp("2025-02-01T00:00:00Z")]
    )

    assert early.manifest["training_window"]["cohorts"] == len(
        panel["formation_at"].unique()
    )
    assert later.manifest["training_window"]["cohorts"] == (
        early.manifest["training_window"]["cohorts"] + 1
    )
    assert early.manifest["training_window"]["latest_formation_at"] < later.manifest[
        "training_window"
    ]["latest_formation_at"]


def test_training_must_match_current_baseline_configuration(
    complete_provider: InMemoryDataProvider,
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    panel = make_training_panel(baseline)
    panel["baseline_config_hash"] = "0" * 64

    with pytest.raises(DataValidationError, match="baseline configuration"):
        _engine(complete_provider, panel).run(AS_OF)


def test_model_interface_allows_alternative_methods_without_baseline_changes(
    complete_provider: InMemoryDataProvider,
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    result = AdaptiveRankingEngine(
        complete_provider,
        InMemoryTrainingDataProvider(make_training_panel(baseline)),
        adaptive_config=adaptive_test_config(),
        model=_ConstantModel(),
    ).run(AS_OF)

    assert result.candidates["probability_outperformance"].eq(0.5).all()
    assert result.candidates["symbol"].tolist() == sorted(
        baseline.candidates["symbol"]
    )
    pdt.assert_frame_equal(result.baseline.candidates, baseline.candidates)


def test_adaptive_result_writes_baseline_and_adaptive_artifacts(
    complete_provider: InMemoryDataProvider, tmp_path
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    result = _engine(complete_provider, make_training_panel(baseline)).run(AS_OF)
    paths = result.write(tmp_path, "comparison")

    assert set(paths) == {
        "adaptive_candidates",
        "adaptive_weights",
        "adaptive_manifest",
        "baseline_candidates",
        "baseline_exclusions",
        "baseline_manifest",
    }
    assert all(path.exists() for path in paths.values())


def test_walk_forward_rejects_non_increasing_decision_dates(
    complete_provider: InMemoryDataProvider,
) -> None:
    baseline = ScreeningEngine(complete_provider).run(AS_OF)
    engine = _engine(complete_provider, make_training_panel(baseline))

    with pytest.raises(DataValidationError, match="strictly increasing"):
        engine.run_walk_forward([AS_OF, AS_OF])
