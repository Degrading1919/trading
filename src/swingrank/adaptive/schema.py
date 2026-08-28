"""Point-in-time contract for historical adaptive-model training rows."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from swingrank.adaptive.config import AdaptiveConfig
from swingrank.exceptions import DataValidationError
from swingrank.factors import RAW_FACTOR_NAMES
from swingrank.schema import validate_as_of

ADAPTIVE_FEATURE_COLUMNS = tuple(
    f"normalized_{factor}" for factor in RAW_FACTOR_NAMES
)

TRAINING_SAMPLE_COLUMNS = (
    "symbol",
    "formation_at",
    "label_start_session",
    "label_end_session",
    "label_end_at",
    "horizon_sessions",
    "forward_return",
    "formation_universe_size",
    "baseline_methodology_version",
    "baseline_config_hash",
    "baseline_run_fingerprint",
    "available_at",
    *ADAPTIVE_FEATURE_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class PreparedTrainingFrame:
    frame: pd.DataFrame
    input_rows: int
    discarded_future_rows: int
    discarded_outside_window_rows: int


def _require_columns(
    frame: pd.DataFrame, required: Iterable[str], name: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataValidationError(
            f"{name} is missing required columns: {', '.join(missing)}"
        )


def _parse_timestamp(frame: pd.DataFrame, column: str) -> pd.Series:
    dtype = frame[column].dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        explicit_timezone = True
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        explicit_timezone = False
    else:
        non_null = frame[column].dropna()
        explicit_timezone = bool(
            non_null.astype(str)
            .str.strip()
            .str.contains(r"(?:[zZ]|[+-]\d{2}:?\d{2})$", regex=True)
            .all()
        )
    if not explicit_timezone:
        raise DataValidationError(
            f"training_samples.{column} must include an explicit UTC offset or Z"
        )
    try:
        parsed = pd.to_datetime(frame[column], utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"training_samples.{column} is not a valid timestamp"
        ) from exc
    if parsed.isna().any():
        raise DataValidationError(f"training_samples.{column} cannot be null")
    return parsed


def _parse_date(frame: pd.DataFrame, column: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(frame[column], errors="coerce")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"training_samples.{column} is not a valid date"
        ) from exc
    if parsed.isna().any():
        raise DataValidationError(f"training_samples.{column} cannot be null")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed.dt.normalize()


def _numeric(
    frame: pd.DataFrame, column: str, *, nullable: bool = False
) -> pd.Series:
    parsed = pd.to_numeric(frame[column], errors="coerce")
    invalid = parsed.isna() & frame[column].notna()
    if invalid.any() or (not nullable and parsed.isna().any()):
        qualifier = "finite numeric values or null" if nullable else "numeric values"
        raise DataValidationError(
            f"training_samples.{column} must contain {qualifier}"
        )
    return parsed.astype(float)


def prepare_training_samples(
    frame: pd.DataFrame,
    as_of: datetime,
    config: AdaptiveConfig,
) -> PreparedTrainingFrame:
    """Select the exact versioned rolling training panel knowable at ``as_of``."""

    _require_columns(frame, TRAINING_SAMPLE_COLUMNS, "training_samples")
    result = frame.loc[:, TRAINING_SAMPLE_COLUMNS].copy()
    input_rows = len(result)
    as_of_utc = validate_as_of(as_of)

    for column in ("formation_at", "label_end_at", "available_at"):
        result[column] = _parse_timestamp(result, column)
    for column in ("label_start_session", "label_end_session"):
        result[column] = _parse_date(result, column)

    known = result["available_at"] <= as_of_utc
    discarded_future = int((~known).sum())
    result = result.loc[known].copy()

    window_start = as_of_utc - pd.Timedelta(
        days=config.training_window_calendar_days
    )
    inside_window = (result["formation_at"] >= window_start) & (
        result["formation_at"] < as_of_utc
    )
    discarded_outside = int((~inside_window).sum())
    result = result.loc[inside_window].copy()
    if result.empty:
        return PreparedTrainingFrame(
            result.reset_index(drop=True),
            input_rows,
            discarded_future,
            discarded_outside,
        )

    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    if (result["symbol"] == "").any():
        raise DataValidationError("training_samples.symbol cannot be blank")

    result["forward_return"] = _numeric(result, "forward_return")
    result["horizon_sessions"] = _numeric(result, "horizon_sessions")
    result["formation_universe_size"] = _numeric(
        result, "formation_universe_size"
    )
    for column in ADAPTIVE_FEATURE_COLUMNS:
        result[column] = _numeric(result, column, nullable=True)

    if not np.isfinite(result["forward_return"]).all():
        raise DataValidationError("training_samples.forward_return must be finite")
    if (result["forward_return"] < -1.0).any():
        raise DataValidationError("training forward returns cannot be below -100%")
    for column in ("horizon_sessions", "formation_universe_size"):
        values = result[column]
        if (
            ~np.isfinite(values)
        ).any() or (values <= 0).any() or (values != np.floor(values)).any():
            raise DataValidationError(
                f"training_samples.{column} must contain positive integers"
            )
        result[column] = values.astype(int)
    if (result["horizon_sessions"] != config.horizon_sessions).any():
        raise DataValidationError(
            "training sample horizon does not match AdaptiveConfig.horizon_sessions"
        )

    feature_values = result.loc[:, ADAPTIVE_FEATURE_COLUMNS].to_numpy(dtype=float)
    observed_features = feature_values[~np.isnan(feature_values)]
    if (
        not np.isfinite(observed_features).all()
        or ((observed_features < -1.0) | (observed_features > 1.0)).any()
    ):
        raise DataValidationError("adaptive normalized features must be in [-1, 1]")

    if (result["formation_at"] >= result["label_end_at"]).any():
        raise DataValidationError("label_end_at must be after formation_at")
    if (result["available_at"] < result["label_end_at"]).any():
        raise DataValidationError("available_at cannot precede label_end_at")
    if (result["label_start_session"] >= result["label_end_session"]).any():
        raise DataValidationError(
            "label_end_session must be after label_start_session"
        )
    if (
        result["label_start_session"]
        > result["formation_at"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    ).any():
        raise DataValidationError(
            "label_start_session cannot be after the formation date"
        )

    for column in (
        "baseline_methodology_version",
        "baseline_config_hash",
        "baseline_run_fingerprint",
    ):
        result[column] = result[column].astype(str).str.strip()
        if (result[column] == "").any():
            raise DataValidationError(f"training_samples.{column} cannot be blank")
    for column in ("baseline_config_hash", "baseline_run_fingerprint"):
        if not result[column].str.fullmatch(r"[0-9a-f]{64}").all():
            raise DataValidationError(
                f"training_samples.{column} must be a lowercase SHA-256 hex digest"
            )

    duplicate_versions = result.duplicated(
        ["symbol", "formation_at", "available_at"], keep=False
    )
    if duplicate_versions.any():
        raise DataValidationError(
            "training samples have duplicate symbol/formation_at/available_at versions"
        )

    immutable_columns = (
        "label_start_session",
        "label_end_session",
        "label_end_at",
        "horizon_sessions",
        "formation_universe_size",
        "baseline_methodology_version",
        "baseline_config_hash",
        "baseline_run_fingerprint",
        *ADAPTIVE_FEATURE_COLUMNS,
    )
    revision_groups = result.groupby(["symbol", "formation_at"], sort=True)
    for column in immutable_columns:
        if (revision_groups[column].nunique(dropna=False) > 1).any():
            raise DataValidationError(
                "training revisions may change only forward_return and available_at; "
                f"found a change in {column}"
            )

    result = result.sort_values(
        ["symbol", "formation_at", "available_at"], kind="mergesort"
    )
    result = result.groupby(
        ["symbol", "formation_at"], sort=True, as_index=False
    ).tail(1)

    cohort_constants = (
        "label_start_session",
        "label_end_session",
        "label_end_at",
        "horizon_sessions",
        "formation_universe_size",
        "baseline_methodology_version",
        "baseline_config_hash",
        "baseline_run_fingerprint",
    )
    cohort_groups = result.groupby("formation_at", sort=True)
    for column in cohort_constants:
        if (cohort_groups[column].nunique(dropna=False) != 1).any():
            raise DataValidationError(
                f"training cohort has inconsistent {column} values"
            )

    actual_sizes = cohort_groups.size()
    declared_sizes = cohort_groups["formation_universe_size"].first()
    incomplete = actual_sizes != declared_sizes
    if incomplete.any():
        dates = [stamp.isoformat() for stamp in actual_sizes.index[incomplete]]
        raise DataValidationError(
            "training cohort is incomplete; missing outcomes can create survivorship "
            "bias: " + ", ".join(dates)
        )
    if (actual_sizes < 2).any():
        raise DataValidationError("training cohorts must contain at least two stocks")

    result = result.sort_values(
        ["formation_at", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    return PreparedTrainingFrame(
        result,
        input_rows,
        discarded_future,
        discarded_outside,
    )
