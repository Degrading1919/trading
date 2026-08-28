"""Safe construction of complete training cohorts from baseline artifacts."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from swingrank.adaptive.schema import (
    ADAPTIVE_FEATURE_COLUMNS,
    TRAINING_SAMPLE_COLUMNS,
)
from swingrank.exceptions import DataValidationError
from swingrank.result import ScreeningResult
from swingrank.schema import validate_as_of


def build_training_cohort(
    baseline: ScreeningResult,
    forward_returns: pd.DataFrame,
    *,
    label_end_session: date | str | pd.Timestamp,
    label_end_at: datetime | pd.Timestamp,
    available_at: datetime | pd.Timestamp,
    horizon_sessions: int = 42,
) -> pd.DataFrame:
    """Join a frozen baseline snapshot to a complete, delisting-aware outcome set.

    ``forward_returns`` must contain exactly one finite total return for every
    baseline candidate.  A provider must incorporate any delisting return; the
    builder refuses missing symbols rather than creating survivor-only labels.
    """

    if horizon_sessions < 1:
        raise DataValidationError("horizon_sessions must be positive")
    if baseline.candidates.empty:
        raise DataValidationError("cannot build a cohort from an empty baseline run")
    required_manifest = {
        "as_of",
        "reference_session",
        "methodology_version",
        "config_hash",
        "run_fingerprint",
    }
    missing_manifest = sorted(required_manifest - set(baseline.manifest))
    if missing_manifest:
        raise DataValidationError(
            "baseline manifest is missing: " + ", ".join(missing_manifest)
        )
    missing_features = sorted(
        set(ADAPTIVE_FEATURE_COLUMNS) - set(baseline.candidates.columns)
    )
    if missing_features:
        raise DataValidationError(
            "baseline candidates are missing adaptive features: "
            + ", ".join(missing_features)
        )
    required_outcomes = {"symbol", "forward_return"}
    missing_outcomes = sorted(required_outcomes - set(forward_returns.columns))
    if missing_outcomes:
        raise DataValidationError(
            "forward_returns is missing: " + ", ".join(missing_outcomes)
        )

    outcomes = forward_returns.loc[:, ["symbol", "forward_return"]].copy()
    outcomes["symbol"] = outcomes["symbol"].astype(str).str.strip().str.upper()
    if outcomes["symbol"].duplicated().any():
        raise DataValidationError("forward_returns symbols must be unique")
    outcomes["forward_return"] = pd.to_numeric(
        outcomes["forward_return"], errors="coerce"
    )
    if (
        outcomes["forward_return"].isna().any()
        or not np.isfinite(outcomes["forward_return"]).all()
        or (outcomes["forward_return"] < -1.0).any()
    ):
        raise DataValidationError(
            "forward_returns must be finite total returns no lower than -100%"
        )

    candidate_symbols = set(baseline.candidates["symbol"].astype(str))
    outcome_symbols = set(outcomes["symbol"])
    if candidate_symbols != outcome_symbols:
        missing = sorted(candidate_symbols - outcome_symbols)
        unexpected = sorted(outcome_symbols - candidate_symbols)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise DataValidationError(
            "forward_returns must cover the exact formation universe ("
            + "; ".join(details)
            + ")"
        )

    formation_at = validate_as_of(
        pd.Timestamp(baseline.manifest["as_of"]).to_pydatetime()
    )
    label_end_at_utc = validate_as_of(pd.Timestamp(label_end_at).to_pydatetime())
    available_at_utc = validate_as_of(pd.Timestamp(available_at).to_pydatetime())
    if label_end_at_utc <= formation_at:
        raise DataValidationError("label_end_at must be after baseline formation")
    if available_at_utc < label_end_at_utc:
        raise DataValidationError("available_at cannot precede label_end_at")

    label_start = pd.Timestamp(baseline.manifest["reference_session"]).normalize()
    label_end = pd.Timestamp(label_end_session)
    if label_end.tzinfo is not None:
        label_end = label_end.tz_convert("UTC").tz_localize(None)
    label_end = label_end.normalize()
    if label_end <= label_start:
        raise DataValidationError("label_end_session must follow label_start_session")

    features = baseline.candidates.loc[
        :, ["symbol", *ADAPTIVE_FEATURE_COLUMNS]
    ].copy()
    features["symbol"] = features["symbol"].astype(str)
    cohort = features.merge(outcomes, on="symbol", how="inner", validate="one_to_one")
    cohort["formation_at"] = formation_at
    cohort["label_start_session"] = label_start
    cohort["label_end_session"] = label_end
    cohort["label_end_at"] = label_end_at_utc
    cohort["horizon_sessions"] = int(horizon_sessions)
    cohort["formation_universe_size"] = len(cohort)
    cohort["baseline_methodology_version"] = str(
        baseline.manifest["methodology_version"]
    )
    cohort["baseline_config_hash"] = str(baseline.manifest["config_hash"])
    cohort["baseline_run_fingerprint"] = str(
        baseline.manifest["run_fingerprint"]
    )
    cohort["available_at"] = available_at_utc
    return cohort.loc[:, TRAINING_SAMPLE_COLUMNS].sort_values(
        "symbol", kind="mergesort"
    ).reset_index(drop=True)
