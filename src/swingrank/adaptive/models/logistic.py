"""Deterministic L2-regularized logistic probability model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from swingrank.adaptive.exceptions import ModelFitError
from swingrank.adaptive.models.base import ModelExplanation


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _objective(
    design: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
    l2_penalty: float,
) -> float:
    logits = intercept + design @ coefficients
    data_loss = np.sum(weights * (np.logaddexp(0.0, logits) - labels * logits))
    penalty = 0.5 * l2_penalty * float(coefficients @ coefficients)
    return float(data_loss / weights.sum() + penalty)


@dataclass(frozen=True, slots=True)
class FittedRidgeLogisticModel:
    """Immutable coefficients and diagnostics from a ridge-logistic fit."""

    feature_names: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    l2_penalty: float
    iterations: int
    objective: float

    @property
    def name(self) -> str:
        return "ridge_logistic_v1"

    def _design(self, features: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.feature_names) - set(features.columns))
        if missing:
            raise ModelFitError(
                "prediction frame is missing features: " + ", ".join(missing)
            )
        design = features.loc[:, self.feature_names].to_numpy(dtype=float)
        if design.ndim != 2 or not np.isfinite(design).all():
            raise ModelFitError("prediction features must be a finite matrix")
        return design

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        design = self._design(features)
        coefficients = np.asarray(self.coefficients, dtype=float)
        return _sigmoid(self.intercept + design @ coefficients)

    def explain(self, features: pd.DataFrame) -> ModelExplanation:
        design = self._design(features)
        coefficients = np.asarray(self.coefficients, dtype=float)
        contributions_array = design * coefficients
        log_odds = self.intercept + contributions_array.sum(axis=1)
        contributions = pd.DataFrame(
            contributions_array,
            columns=self.feature_names,
            index=features.index,
        )
        return ModelExplanation(
            probabilities=_sigmoid(log_odds),
            log_odds=log_odds,
            base_log_odds=np.full(len(features), self.intercept, dtype=float),
            contributions=contributions,
        )

    def state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_names": list(self.feature_names),
            "intercept_log_odds": self.intercept,
            "coefficients_log_odds": dict(
                zip(self.feature_names, self.coefficients, strict=True)
            ),
            "l2_penalty": self.l2_penalty,
            "iterations": self.iterations,
            "objective": self.objective,
            "converged": True,
            "objective_definition": (
                "weighted_mean_log_loss_plus_half_l2_coefficient_norm"
            ),
        }

    def learned_weights(self) -> pd.DataFrame:
        coefficients = np.asarray(self.coefficients, dtype=float)
        absolute = np.abs(coefficients)
        denominator = float(absolute.sum())
        relative = absolute / denominator if denominator > 0 else np.zeros_like(absolute)
        return pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient_log_odds": coefficients,
                "odds_ratio_per_one_unit": np.exp(coefficients),
                "odds_ratio_full_feature_range": np.exp(2.0 * coefficients),
                "relative_absolute_weight": relative,
            }
        )


@dataclass(frozen=True, slots=True)
class RidgeLogisticModel:
    """Fit a transparent linear probability ranking with coefficient shrinkage."""

    l2_penalty: float = 0.10
    maximum_iterations: int = 100
    convergence_tolerance: float = 1e-9

    @property
    def name(self) -> str:
        return "ridge_logistic_v1"

    def fit(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        sample_weight: np.ndarray,
    ) -> FittedRidgeLogisticModel:
        design = features.to_numpy(dtype=float)
        target = np.asarray(labels, dtype=float)
        weights = np.asarray(sample_weight, dtype=float)
        self._validate_fit_inputs(features, design, target, weights)

        weight_sum = float(weights.sum())
        prevalence = float(weights @ target / weight_sum)
        prevalence = float(np.clip(prevalence, 1e-12, 1.0 - 1e-12))
        intercept = float(np.log(prevalence / (1.0 - prevalence)))
        coefficients = np.zeros(design.shape[1], dtype=float)
        converged = False
        final_iteration = 0

        for iteration in range(1, self.maximum_iterations + 1):
            final_iteration = iteration
            logits = intercept + design @ coefficients
            probabilities = _sigmoid(logits)
            residual = weights * (probabilities - target)
            gradient = np.concatenate(
                (
                    [residual.sum() / weight_sum],
                    design.T @ residual / weight_sum
                    + self.l2_penalty * coefficients,
                )
            )
            if float(np.max(np.abs(gradient))) <= self.convergence_tolerance:
                converged = True
                break

            curvature = weights * probabilities * (1.0 - probabilities)
            hessian = np.empty(
                (design.shape[1] + 1, design.shape[1] + 1), dtype=float
            )
            hessian[0, 0] = curvature.sum() / weight_sum
            cross = design.T @ curvature / weight_sum
            hessian[0, 1:] = cross
            hessian[1:, 0] = cross
            hessian[1:, 1:] = (
                design.T @ (design * curvature[:, None]) / weight_sum
                + self.l2_penalty * np.eye(design.shape[1])
            )
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError as exc:
                raise ModelFitError("ridge-logistic Hessian is singular") from exc

            current_objective = _objective(
                design,
                target,
                weights,
                intercept,
                coefficients,
                self.l2_penalty,
            )
            scale = 1.0
            accepted = False
            while scale >= 2.0**-30:
                candidate_intercept = float(intercept - scale * step[0])
                candidate_coefficients = coefficients - scale * step[1:]
                candidate_objective = _objective(
                    design,
                    target,
                    weights,
                    candidate_intercept,
                    candidate_coefficients,
                    self.l2_penalty,
                )
                if candidate_objective <= current_objective + 1e-15:
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                raise ModelFitError("ridge-logistic line search failed")

            intercept = candidate_intercept
            coefficients = candidate_coefficients
            scaled_step = scale * step
            if float(np.max(np.abs(scaled_step))) <= self.convergence_tolerance * (
                1.0
                + max(abs(intercept), float(np.max(np.abs(coefficients), initial=0.0)))
            ):
                converged = True
                break

        if not converged:
            raise ModelFitError(
                "ridge-logistic model did not converge within "
                f"{self.maximum_iterations} iterations"
            )
        if not np.isfinite(intercept) or not np.isfinite(coefficients).all():
            raise ModelFitError("ridge-logistic fit produced non-finite coefficients")

        final_objective = _objective(
            design,
            target,
            weights,
            intercept,
            coefficients,
            self.l2_penalty,
        )
        return FittedRidgeLogisticModel(
            feature_names=tuple(str(column) for column in features.columns),
            intercept=intercept,
            coefficients=tuple(float(value) for value in coefficients),
            l2_penalty=self.l2_penalty,
            iterations=final_iteration,
            objective=final_objective,
        )

    @staticmethod
    def _validate_fit_inputs(
        features: pd.DataFrame,
        design: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        if features.columns.duplicated().any() or design.ndim != 2:
            raise ModelFitError("training features must be a two-dimensional table")
        if len(design) < 2 or design.shape[1] < 1:
            raise ModelFitError("training requires at least two rows and one feature")
        if labels.shape != (len(design),) or weights.shape != (len(design),):
            raise ModelFitError("labels and sample weights must align to features")
        if not np.isfinite(design).all():
            raise ModelFitError("training features must be finite")
        if not np.isin(labels, [0.0, 1.0]).all() or len(np.unique(labels)) != 2:
            raise ModelFitError("training labels must contain both binary classes")
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ModelFitError("sample weights must be finite and strictly positive")
