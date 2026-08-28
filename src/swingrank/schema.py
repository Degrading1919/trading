"""Data contract enforcement and point-in-time filtering."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from swingrank.exceptions import DataValidationError

UNIVERSE_COLUMNS = (
    "symbol",
    "security_type",
    "country",
    "exchange",
    "currency",
    "sector",
    "industry",
    "market_cap",
    "is_primary_listing",
    "effective_from",
    "effective_to",
    "available_at",
)

PRICE_COLUMNS = (
    "symbol",
    "session_date",
    "close",
    "split_adjusted_close",
    "total_return_index",
    "volume",
    "available_at",
)

EARNINGS_COLUMNS = (
    "symbol",
    "fiscal_period_end",
    "reported_eps",
    "announced_at",
    "available_at",
)


@dataclass(frozen=True, slots=True)
class PreparedFrame:
    frame: pd.DataFrame
    input_rows: int
    discarded_future_rows: int


def validate_as_of(as_of: datetime) -> pd.Timestamp:
    stamp = pd.Timestamp(as_of)
    if stamp.tzinfo is None:
        raise DataValidationError("as_of must be timezone-aware")
    return stamp.tz_convert("UTC")


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise DataValidationError(
            f"{name} is missing required columns: {', '.join(missing)}"
        )


def _parse_timestamp(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    dtype = frame[column].dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        has_explicit_timezone = True
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        has_explicit_timezone = False
    else:
        non_null = frame[column].dropna()
        timezone_suffix = (
            non_null.astype(str)
            .str.strip()
            .str.contains(r"(?:[zZ]|[+-]\d{2}:?\d{2})$", regex=True)
        )
        has_explicit_timezone = bool(timezone_suffix.all())
    if not has_explicit_timezone:
        raise DataValidationError(
            f"{name}.{column} must include an explicit UTC offset or Z suffix"
        )
    try:
        parsed = pd.to_datetime(frame[column], utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name}.{column} is not a valid timestamp") from exc
    if parsed.isna().any():
        raise DataValidationError(f"{name}.{column} cannot be null")
    return parsed


def _parse_date(
    frame: pd.DataFrame,
    column: str,
    name: str,
    *,
    nullable: bool = False,
) -> pd.Series:
    try:
        parsed = pd.to_datetime(frame[column], errors="coerce")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name}.{column} is not a valid date") from exc
    if not nullable and parsed.isna().any():
        raise DataValidationError(f"{name}.{column} cannot be null")
    return parsed.dt.normalize()


def _numeric(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    parsed = pd.to_numeric(frame[column], errors="coerce")
    invalid = parsed.isna() & frame[column].notna()
    if invalid.any():
        raise DataValidationError(f"{name}.{column} contains non-numeric values")
    return parsed.astype(float)


def prepare_universe(frame: pd.DataFrame, as_of: datetime) -> PreparedFrame:
    _require_columns(frame, UNIVERSE_COLUMNS, "universe")
    result = frame.loc[:, UNIVERSE_COLUMNS].copy()
    input_rows = len(result)
    as_of_utc = validate_as_of(as_of)

    result["available_at"] = _parse_timestamp(result, "available_at", "universe")
    known = result["available_at"] <= as_of_utc
    discarded = int((~known).sum())
    result = result.loc[known].copy()

    result["effective_from"] = _parse_date(result, "effective_from", "universe")
    result["effective_to"] = _parse_date(
        result, "effective_to", "universe", nullable=True
    )
    result["market_cap"] = _numeric(result, "market_cap", "universe")
    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    if (result["symbol"] == "").any():
        raise DataValidationError("universe.symbol cannot be blank")

    bool_map = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    if result["is_primary_listing"].dtype != bool:
        converted = result["is_primary_listing"].astype(str).str.lower().map(bool_map)
        if converted.isna().any():
            raise DataValidationError("universe.is_primary_listing must be boolean")
        result["is_primary_listing"] = converted.astype(bool)

    for column in ("security_type", "country", "exchange", "currency"):
        result[column] = result[column].astype(str).str.strip().str.upper()
    for column in ("sector", "industry"):
        result[column] = result[column].fillna("").astype(str).str.strip()

    duplicate_versions = result.duplicated(["symbol", "available_at"], keep=False)
    if duplicate_versions.any():
        raise DataValidationError(
            "universe has duplicate symbol/available_at versions: "
            + ", ".join(sorted(result.loc[duplicate_versions, "symbol"].unique()))
        )

    # Select the latest snapshot known at the decision time before testing active
    # dates.  Doing this in the opposite order can resurrect an obsolete record.
    result = result.sort_values(["symbol", "available_at"], kind="mergesort")
    result = result.groupby("symbol", sort=True, as_index=False).tail(1)
    result = result.sort_values("symbol", kind="mergesort").reset_index(drop=True)
    return PreparedFrame(result, input_rows, discarded)


def prepare_prices(
    frame: pd.DataFrame,
    as_of: datetime,
    market_timezone: str = "America/New_York",
) -> PreparedFrame:
    _require_columns(frame, PRICE_COLUMNS, "prices")
    result = frame.loc[:, PRICE_COLUMNS].copy()
    input_rows = len(result)
    as_of_utc = validate_as_of(as_of)

    result["available_at"] = _parse_timestamp(result, "available_at", "prices")
    result["session_date"] = _parse_date(result, "session_date", "prices")
    market_date = as_of_utc.tz_convert(market_timezone).tz_localize(None).normalize()
    known = (result["available_at"] <= as_of_utc) & (
        result["session_date"] <= market_date
    )
    discarded = int((~known).sum())
    result = result.loc[known].copy()
    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()
    for column in ("close", "split_adjusted_close", "total_return_index", "volume"):
        result[column] = _numeric(result, column, "prices")

    if (result["symbol"] == "").any():
        raise DataValidationError("prices.symbol cannot be blank")
    if (
        (
            ~np.isfinite(
                result[
                    ["close", "split_adjusted_close", "total_return_index", "volume"]
                ]
            )
        )
        .any()
        .any()
    ):
        raise DataValidationError("prices contains infinite numeric values")
    if (
        (result[["close", "split_adjusted_close", "total_return_index"]] <= 0)
        .any()
        .any()
    ):
        raise DataValidationError("prices and return indices must be strictly positive")
    if (result["volume"] < 0).any():
        raise DataValidationError("prices.volume cannot be negative")

    duplicate_versions = result.duplicated(
        ["symbol", "session_date", "available_at"], keep=False
    )
    if duplicate_versions.any():
        raise DataValidationError(
            "prices has duplicate symbol/session_date/available_at versions"
        )

    result = result.sort_values(
        ["symbol", "session_date", "available_at"], kind="mergesort"
    )
    result = result.groupby(["symbol", "session_date"], sort=True, as_index=False).tail(
        1
    )
    result = result.sort_values(
        ["symbol", "session_date"], kind="mergesort"
    ).reset_index(drop=True)
    return PreparedFrame(result, input_rows, discarded)


def prepare_earnings(frame: pd.DataFrame, as_of: datetime) -> PreparedFrame:
    if frame.empty and len(frame.columns) == 0:
        return PreparedFrame(pd.DataFrame(columns=EARNINGS_COLUMNS), 0, 0)
    _require_columns(frame, EARNINGS_COLUMNS, "earnings")
    result = frame.loc[:, EARNINGS_COLUMNS].copy()
    input_rows = len(result)
    as_of_utc = validate_as_of(as_of)

    result["available_at"] = _parse_timestamp(result, "available_at", "earnings")
    result["announced_at"] = _parse_timestamp(result, "announced_at", "earnings")
    result["fiscal_period_end"] = _parse_date(result, "fiscal_period_end", "earnings")
    known = (result["available_at"] <= as_of_utc) & (
        result["announced_at"] <= as_of_utc
    )
    discarded = int((~known).sum())
    result = result.loc[known].copy()
    result["reported_eps"] = _numeric(result, "reported_eps", "earnings")
    result["symbol"] = result["symbol"].astype(str).str.strip().str.upper()

    if (result["symbol"] == "").any():
        raise DataValidationError("earnings.symbol cannot be blank")
    if (~np.isfinite(result["reported_eps"])).any():
        raise DataValidationError("earnings.reported_eps must be finite")
    if (result["available_at"] < result["announced_at"]).any():
        raise DataValidationError("earnings.available_at cannot precede announced_at")
    fiscal_utc = result["fiscal_period_end"].dt.tz_localize("UTC")
    if (fiscal_utc > result["announced_at"]).any():
        raise DataValidationError(
            "earnings fiscal period cannot end after announcement"
        )

    duplicate_versions = result.duplicated(
        ["symbol", "fiscal_period_end", "available_at"], keep=False
    )
    if duplicate_versions.any():
        raise DataValidationError(
            "earnings has duplicate symbol/fiscal_period_end/available_at versions"
        )

    result = result.sort_values(
        ["symbol", "fiscal_period_end", "available_at"], kind="mergesort"
    )
    result = result.groupby(
        ["symbol", "fiscal_period_end"], sort=True, as_index=False
    ).tail(1)
    result = result.sort_values(
        ["symbol", "fiscal_period_end"], kind="mergesort"
    ).reset_index(drop=True)
    return PreparedFrame(result, input_rows, discarded)
