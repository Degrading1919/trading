"""Point-in-time orchestration for adaptive probabilistic ranking."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from swingrank.adaptive.config import AdaptiveConfig
from swingrank.adaptive.labels import (
    BinaryOutcomeLabeler,
    CrossSectionalMedianLabeler,
)
from swingrank.adaptive.models.base import ProbabilisticModel
from swingrank.adaptive.models.logistic import RidgeLogisticModel
from swingrank.adaptive.providers import TrainingDataProvider
from swingrank.adaptive.result import AdaptiveScreeningResult
from swingrank.adaptive.schema import (
    ADAPTIVE_FEATURE_COLUMNS,
    PreparedTrainingFrame,
    prepare_training_samples,
)
from swingrank.audit import hash_frame, sha256_json
from swingrank.config import ScreeningConfig
from swingrank.engine import ScreeningEngine
from swingrank.exceptions import DataValidationError
from swingrank.factors import RAW_FACTOR_NAMES
from swingrank.providers.base import MarketDataProvider
from swingrank.schema import validate_as_of


class AdaptiveRankingEngine:
    """Learn recent feature weights while retaining the fixed baseline intact."""

    def __init__(
        self,
        market_provider: MarketDataProvider,
        training_provider: TrainingDataProvider,
        baseline_config: ScreeningConfig | None = None,
        adaptive_config: AdaptiveConfig | None = None,
        model: ProbabilisticModel | None = None,
        labeler: BinaryOutcomeLabeler | None = None,
    ) -> None:
        self.market_provider = market_provider
        self.training_provider = training_provider
        self.baseline_config = baseline_config or ScreeningConfig()
        self.adaptive_config = adaptive_config or AdaptiveConfig()
        self.model = (
            model
            if model is not None
            else RidgeLogisticModel(
                l2_penalty=self.adaptive_config.l2_penalty,
                maximum_iterations=self.adaptive_config.maximum_iterations,
                convergence_tolerance=self.adaptive_config.convergence_tolerance,
            )
        )
        self.labeler = (
            labeler if labeler is not None else CrossSectionalMedianLabeler()
        )

    def run(self, as_of: pd.Timestamp | datetime | Any) -> AdaptiveScreeningResult:
        as_of_utc = validate_as_of(pd.Timestamp(as_of).to_pydatetime())
        as_of_dt = as_of_utc.to_pydatetime()

        # Calling the unchanged baseline engine is the only path to the current
        # universe and features; the adaptive layer never reimplements factors.
        baseline = ScreeningEngine(
            self.market_provider, self.baseline_config
        ).run(as_of_dt)
        if baseline.candidates.empty:
            raise DataValidationError(
                "adaptive ranking requires a non-empty fixed-baseline universe"
            )

        raw_training = self.training_provider.training_samples(as_of_dt)
        prepared = prepare_training_samples(
            raw_training, as_of_dt, self.adaptive_config
        )
        training = prepared.frame
        self._validate_training_panel(training, baseline.manifest)

        labels = self.labeler.labels(training).to_numpy(dtype=int)
        self._validate_labels(labels)
        training_features = training.loc[:, ADAPTIVE_FEATURE_COLUMNS].fillna(0.0)
        sample_weight = self._equal_cohort_weights(training)
        fitted = self.model.fit(training_features, labels, sample_weight)
        if tuple(fitted.feature_names) != ADAPTIVE_FEATURE_COLUMNS:
            raise DataValidationError(
                "fitted model changed the canonical adaptive feature order"
            )

        current_features = baseline.candidates.loc[
            :, ADAPTIVE_FEATURE_COLUMNS
        ].fillna(0.0)
        explanation = fitted.explain(current_features)
        candidates = self._candidate_table(baseline.candidates, explanation)
        learned_weights = fitted.learned_weights().copy()
        learned_weights["factor"] = learned_weights["feature"].str.removeprefix(
            "normalized_"
        )
        learned_weights["baseline_fixed_weight"] = learned_weights["factor"].map(
            self.baseline_config.weights
        )
        learned_weights = learned_weights[
            [
                "factor",
                "feature",
                "baseline_fixed_weight",
                *(
                    column
                    for column in learned_weights.columns
                    if column
                    not in {"factor", "feature", "baseline_fixed_weight"}
                ),
            ]
        ]

        manifest = self._manifest(
            as_of_utc,
            baseline.manifest,
            prepared,
            training,
            training_features,
            labels,
            sample_weight,
            fitted,
            candidates,
        )
        return AdaptiveScreeningResult(
            baseline=baseline,
            candidates=candidates,
            learned_weights=learned_weights,
            manifest=manifest,
        )

    def run_walk_forward(
        self, decision_dates: Iterable[pd.Timestamp | datetime | Any]
    ) -> tuple[AdaptiveScreeningResult, ...]:
        """Fit a fresh model at each strictly increasing decision timestamp."""

        dates = [
            validate_as_of(pd.Timestamp(value).to_pydatetime())
            for value in decision_dates
        ]
        if any(later <= earlier for earlier, later in pairwise(dates)):
            raise DataValidationError(
                "walk-forward decision dates must be strictly increasing"
            )
        return tuple(self.run(value) for value in dates)

    def _validate_training_panel(
        self, training: pd.DataFrame, baseline_manifest: dict[str, Any]
    ) -> None:
        config = self.adaptive_config
        cohorts = training["formation_at"].nunique() if not training.empty else 0
        if len(training) < config.minimum_training_samples:
            raise DataValidationError(
                "insufficient adaptive training samples: "
                f"{len(training)} < {config.minimum_training_samples}"
            )
        if cohorts < config.minimum_training_cohorts:
            raise DataValidationError(
                "insufficient adaptive training cohorts: "
                f"{cohorts} < {config.minimum_training_cohorts}"
            )

        versions = set(training["baseline_methodology_version"])
        if versions != {str(baseline_manifest["methodology_version"])}:
            raise DataValidationError(
                "training features were not generated by the current fixed-baseline "
                "methodology version"
            )
        config_hashes = set(training["baseline_config_hash"])
        if config_hashes != {str(baseline_manifest["config_hash"])}:
            raise DataValidationError(
                "training features were not generated with the current fixed-baseline "
                "configuration"
            )

    def _validate_labels(self, labels: np.ndarray) -> None:
        if labels.shape[0] == 0 or not np.isin(labels, [0, 1]).all():
            raise DataValidationError("adaptive labeler must return binary labels")
        counts = np.bincount(labels, minlength=2)
        if (counts < self.adaptive_config.minimum_class_samples).any():
            raise DataValidationError(
                "adaptive training needs at least "
                f"{self.adaptive_config.minimum_class_samples} samples in each class"
            )

    @staticmethod
    def _equal_cohort_weights(training: pd.DataFrame) -> np.ndarray:
        sizes = training.groupby("formation_at", sort=True)["symbol"].transform(
            "size"
        )
        raw = 1.0 / sizes.to_numpy(dtype=float)
        return raw * (len(raw) / raw.sum())

    @staticmethod
    def _candidate_table(
        baseline_candidates: pd.DataFrame,
        explanation: Any,
    ) -> pd.DataFrame:
        expected_rows = len(baseline_candidates)
        if (
            explanation.probabilities.shape != (expected_rows,)
            or explanation.log_odds.shape != (expected_rows,)
            or explanation.base_log_odds.shape != (expected_rows,)
        ):
            raise DataValidationError("model explanation arrays do not align to candidates")
        if list(explanation.contributions.columns) != list(ADAPTIVE_FEATURE_COLUMNS):
            raise DataValidationError(
                "model explanation must use the canonical adaptive features"
            )

        probabilities = np.asarray(explanation.probabilities, dtype=float)
        log_odds = np.asarray(explanation.log_odds, dtype=float)
        base_log_odds = np.asarray(explanation.base_log_odds, dtype=float)
        contributions = explanation.contributions.to_numpy(dtype=float)
        if (
            not np.isfinite(probabilities).all()
            or not np.isfinite(log_odds).all()
            or not np.isfinite(base_log_odds).all()
            or not np.isfinite(contributions).all()
        ):
            raise DataValidationError("adaptive model output must be finite")
        if ((probabilities <= 0.0) | (probabilities >= 1.0)).any():
            raise DataValidationError("adaptive probabilities must be strictly in (0, 1)")
        if not np.allclose(
            base_log_odds + contributions.sum(axis=1),
            log_odds,
            rtol=0.0,
            atol=1e-12,
        ):
            raise DataValidationError(
                "adaptive contributions plus base value must sum to log odds"
            )
        reconstructed = np.empty_like(log_odds)
        positive = log_odds >= 0
        reconstructed[positive] = 1.0 / (1.0 + np.exp(-log_odds[positive]))
        exponent = np.exp(log_odds[~positive])
        reconstructed[~positive] = exponent / (1.0 + exponent)
        if not np.allclose(
            reconstructed, probabilities, rtol=0.0, atol=1e-12
        ):
            raise DataValidationError(
                "adaptive probability does not match explained log odds"
            )

        renamed = {
            "rank": "baseline_rank",
            "score": "baseline_score",
            "score_coverage": "baseline_score_coverage",
            **{
                f"contribution_{factor}": f"baseline_contribution_{factor}"
                for factor in RAW_FACTOR_NAMES
            },
        }
        candidates = baseline_candidates.rename(columns=renamed).copy()
        candidates["probability_outperformance"] = probabilities
        candidates["confidence_from_50pct"] = 2.0 * np.abs(probabilities - 0.5)
        candidates["adaptive_log_odds"] = log_odds
        candidates["adaptive_base_log_odds"] = base_log_odds
        for index, feature in enumerate(ADAPTIVE_FEATURE_COLUMNS):
            factor = feature.removeprefix("normalized_")
            candidates[f"adaptive_contribution_{factor}"] = contributions[:, index]

        candidates = candidates.sort_values(
            ["probability_outperformance", "symbol"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        candidates.insert(
            0, "adaptive_rank", np.arange(1, len(candidates) + 1, dtype=int)
        )
        leading = [
            "adaptive_rank",
            "symbol",
            "probability_outperformance",
            "confidence_from_50pct",
            "adaptive_log_odds",
            "adaptive_base_log_odds",
            "baseline_rank",
            "baseline_score",
            "baseline_score_coverage",
        ]
        remaining = [column for column in candidates if column not in leading]
        return candidates.loc[:, leading + remaining]

    def _manifest(
        self,
        as_of_utc: pd.Timestamp,
        baseline_manifest: dict[str, Any],
        prepared: PreparedTrainingFrame,
        training: pd.DataFrame,
        training_features: pd.DataFrame,
        labels: np.ndarray,
        sample_weight: np.ndarray,
        fitted: Any,
        candidates: pd.DataFrame,
    ) -> dict[str, Any]:
        config_dict = self.adaptive_config.as_dict()
        config_hash = sha256_json(config_dict)
        training_hash = hash_frame(training)
        model_state = fitted.state()
        train_probability = np.asarray(
            fitted.predict_proba(training_features), dtype=float
        )
        if (
            train_probability.shape != labels.shape
            or not np.isfinite(train_probability).all()
            or ((train_probability <= 0.0) | (train_probability >= 1.0)).any()
        ):
            raise DataValidationError(
                "fitted model training probabilities must be finite and in (0, 1)"
            )
        clipped = np.clip(
            train_probability,
            self.adaptive_config.probability_clip,
            1.0 - self.adaptive_config.probability_clip,
        )
        weight_sum = float(sample_weight.sum())
        prevalence = float(sample_weight @ labels / weight_sum)
        brier = float(sample_weight @ ((train_probability - labels) ** 2) / weight_sum)
        log_loss = float(
            -sample_weight
            @ (labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped))
            / weight_sum
        )
        naive_brier = float(
            sample_weight @ ((prevalence - labels) ** 2) / weight_sum
        )
        fingerprint_payload = {
            "as_of": as_of_utc.isoformat(),
            "adaptive_methodology_version": self.adaptive_config.methodology_version,
            "adaptive_config_hash": config_hash,
            "baseline_run_fingerprint": baseline_manifest["run_fingerprint"],
            "training_input_hash": training_hash,
            "labeler": self.labeler.name,
            "model_state": model_state,
        }
        missing_rates = {
            feature.removeprefix("normalized_"): float(training[feature].isna().mean())
            for feature in ADAPTIVE_FEATURE_COLUMNS
        }
        return {
            **fingerprint_payload,
            "run_fingerprint": sha256_json(fingerprint_payload),
            "market_provider": self.market_provider.name,
            "training_provider": self.training_provider.name,
            "adaptive_config": config_dict,
            "model": model_state,
            "probability_event": self.labeler.name,
            "probability_horizon_sessions": self.adaptive_config.horizon_sessions,
            "confidence_definition": (
                "2 * abs(probability_outperformance - 0.5); this is distance from "
                "an even-probability forecast, not a statistical confidence interval"
            ),
            "feature_missing_policy": (
                "missing baseline-normalized factors enter the adaptive model as zero, "
                "matching their neutral fixed-baseline contribution"
            ),
            "cohort_weighting": (
                "each formation cohort receives equal total likelihood weight"
            ),
            "training_window": {
                "requested_start": (
                    as_of_utc
                    - pd.Timedelta(
                        days=self.adaptive_config.training_window_calendar_days
                    )
                ).isoformat(),
                "decision_as_of": as_of_utc.isoformat(),
                "earliest_formation_at": training["formation_at"].min().isoformat(),
                "latest_formation_at": training["formation_at"].max().isoformat(),
                "latest_label_end_at": training["label_end_at"].max().isoformat(),
                "latest_sample_available_at": training["available_at"].max().isoformat(),
                "cohorts": int(training["formation_at"].nunique()),
                "samples": len(training),
                "positive_samples": int(labels.sum()),
                "weighted_positive_prevalence": prevalence,
                "missing_feature_rates": missing_rates,
            },
            "training_diagnostics": {
                "scope": "in_sample_fit_diagnostic_not_out_of_sample_performance",
                "weighted_brier_score": brier,
                "weighted_log_loss": log_loss,
                "weighted_prevalence_brier_score": naive_brier,
            },
            "row_counts": {
                "training_input": prepared.input_rows,
                "training_used": len(training),
                "training_discarded_future": prepared.discarded_future_rows,
                "training_discarded_outside_window": (
                    prepared.discarded_outside_window_rows
                ),
                "candidates": len(candidates),
            },
            "baseline": {
                "methodology_version": baseline_manifest["methodology_version"],
                "config_hash": baseline_manifest["config_hash"],
                "run_fingerprint": baseline_manifest["run_fingerprint"],
                "reference_session": baseline_manifest["reference_session"],
            },
            "llm_in_prediction_path": False,
        }
