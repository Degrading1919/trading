"""Point-in-time screening, factor normalization, and ranking orchestration."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import timedelta
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any

import numpy as np
import pandas as pd

from swingrank.audit import combined_input_hash, sha256_json
from swingrank.config import ScreeningConfig
from swingrank.exceptions import DataValidationError
from swingrank.factors import (
    RAW_FACTOR_NAMES,
    add_leave_one_out_sector_momentum,
    calculate_earnings_factors,
    calculate_price_snapshot,
)
from swingrank.normalization import centered_percentile_rank
from swingrank.providers.base import MarketDataProvider
from swingrank.result import ScreeningResult
from swingrank.schema import (
    PreparedFrame,
    prepare_earnings,
    prepare_prices,
    prepare_universe,
    validate_as_of,
)
from swingrank.version import __version__


@dataclass(slots=True)
class _Exclusion:
    symbol: str
    reasons: list[str]


class ScreeningEngine:
    """Run the fixed methodology against any compliant market-data provider."""

    def __init__(
        self,
        provider: MarketDataProvider,
        config: ScreeningConfig | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or ScreeningConfig()

    def run(self, as_of: pd.Timestamp | Any) -> ScreeningResult:
        as_of_utc = validate_as_of(pd.Timestamp(as_of).to_pydatetime())
        as_of_dt = as_of_utc.to_pydatetime()
        raw_universe = self.provider.universe(as_of_dt)
        universe_prepared = prepare_universe(raw_universe, as_of_dt)
        universe = universe_prepared.frame
        symbols = universe["symbol"].drop_duplicates().sort_values().tolist()

        price_start = (
            as_of_utc - timedelta(days=self.config.price_lookback_calendar_days)
        ).date()
        earnings_start = (
            as_of_utc - timedelta(days=self.config.earnings_lookback_calendar_days)
        ).date()
        raw_prices = self.provider.prices(
            symbols, price_start, as_of_utc.date(), as_of_dt
        )
        raw_earnings = self.provider.earnings(
            symbols, earnings_start, as_of_utc.date(), as_of_dt
        )
        prices_prepared = prepare_prices(
            raw_prices, as_of_dt, market_timezone=self.config.market_timezone
        )
        earnings_prepared = prepare_earnings(raw_earnings, as_of_dt)

        prices = prices_prepared.frame
        earnings = earnings_prepared.frame
        symbol_set = set(symbols)
        prices = prices.loc[prices["symbol"].isin(symbol_set)].reset_index(drop=True)
        earnings = earnings.loc[earnings["symbol"].isin(symbol_set)].reset_index(
            drop=True
        )

        input_hash, component_hashes = combined_input_hash(
            {"universe": universe, "prices": prices, "earnings": earnings}
        )
        candidates, exclusions, reference_session = self._screen_and_score(
            universe, prices, earnings, as_of_dt
        )
        manifest = self._manifest(
            as_of_utc,
            universe_prepared,
            prices_prepared,
            earnings_prepared,
            input_hash,
            component_hashes,
            candidates,
            exclusions,
            reference_session,
        )
        return ScreeningResult(candidates, exclusions, manifest)

    def _screen_and_score(
        self,
        universe: pd.DataFrame,
        prices: pd.DataFrame,
        earnings: pd.DataFrame,
        as_of: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
        config = self.config
        exclusions: list[_Exclusion] = []
        eligible_metadata: list[pd.Series] = []
        as_of_day = (
            pd.Timestamp(as_of)
            .tz_convert(config.market_timezone)
            .tz_localize(None)
            .normalize()
        )

        for _, row in universe.sort_values("symbol", kind="mergesort").iterrows():
            reasons: list[str] = []
            if row["effective_from"] > as_of_day or (
                pd.notna(row["effective_to"]) and as_of_day >= row["effective_to"]
            ):
                reasons.append("inactive_as_of")
            if row["security_type"] not in config.allowed_security_types:
                reasons.append("security_type")
            if row["country"] not in config.allowed_countries:
                reasons.append("country")
            if row["exchange"] not in config.allowed_exchanges:
                reasons.append("exchange")
            if row["currency"] != config.required_currency:
                reasons.append("currency")
            if config.require_primary_listing and not bool(row["is_primary_listing"]):
                reasons.append("not_primary_listing")
            if (
                not np.isfinite(row["market_cap"])
                or row["market_cap"] < config.minimum_market_cap
            ):
                reasons.append("market_cap")
            if reasons:
                exclusions.append(_Exclusion(str(row["symbol"]), reasons))
            else:
                eligible_metadata.append(row)

        if not eligible_metadata:
            return self._empty_candidates(), self._exclusion_frame(exclusions), None

        metadata = pd.DataFrame(eligible_metadata).reset_index(drop=True)
        metadata_symbols = set(metadata["symbol"])
        relevant_prices = prices.loc[prices["symbol"].isin(metadata_symbols)].copy()
        sessions = pd.DatetimeIndex(
            sorted(relevant_prices["session_date"].dropna().unique())
        )
        reference_session = sessions[-1].date().isoformat() if len(sessions) else None
        if len(sessions) < config.required_price_sessions:
            for symbol in sorted(metadata_symbols):
                exclusions.append(_Exclusion(symbol, ["insufficient_market_history"]))
            return (
                self._empty_candidates(),
                self._exclusion_frame(exclusions),
                reference_session,
            )

        reference_sessions = sessions[-config.required_price_sessions :]
        price_groups = {
            symbol: group
            for symbol, group in relevant_prices.groupby("symbol", sort=True)
        }
        factor_rows: list[dict[str, Any]] = []
        for _, row in metadata.sort_values("symbol", kind="mergesort").iterrows():
            symbol = str(row["symbol"])
            group = price_groups.get(symbol)
            if group is None or group.empty:
                exclusions.append(_Exclusion(symbol, ["missing_prices"]))
                continue
            snapshot = calculate_price_snapshot(group, reference_sessions, config)
            reasons = []
            if pd.Timestamp(snapshot["latest_session"]) != reference_sessions[-1]:
                reasons.append("stale_price")
            if float(snapshot["history_coverage"]) < config.minimum_history_coverage:
                reasons.append("price_history_coverage")
            if not bool(snapshot["anchors_complete"]):
                reasons.append("missing_momentum_anchor")
            if not bool(snapshot["recent_history_complete"]):
                reasons.append("missing_recent_history")
            latest_price = float(snapshot["latest_price"])
            if not np.isfinite(latest_price) or latest_price < config.minimum_price:
                reasons.append("price")
            median_dollar_volume = float(snapshot["median_dollar_volume_20d"])
            if (
                not np.isfinite(median_dollar_volume)
                or median_dollar_volume < config.minimum_median_dollar_volume_20d
            ):
                reasons.append("dollar_volume")
            nonzero_fraction = float(snapshot["nonzero_volume_fraction_20d"])
            if (
                not np.isfinite(nonzero_fraction)
                or nonzero_fraction < config.minimum_nonzero_volume_fraction_20d
            ):
                reasons.append("zero_volume_days")
            if reasons:
                exclusions.append(_Exclusion(symbol, reasons))
                continue
            factor_rows.append(
                {
                    "symbol": symbol,
                    "sector": row["sector"],
                    "industry": row["industry"],
                    "market_cap": float(row["market_cap"]),
                    **snapshot,
                }
            )

        if not factor_rows:
            return (
                self._empty_candidates(),
                self._exclusion_frame(exclusions),
                reference_session,
            )

        scored = (
            pd.DataFrame(factor_rows)
            .sort_values("symbol", kind="mergesort")
            .reset_index(drop=True)
        )
        scored["sector_momentum"] = add_leave_one_out_sector_momentum(scored, config)
        scored["earnings_sue"] = calculate_earnings_factors(
            earnings, scored["symbol"], as_of, config
        )

        for factor in RAW_FACTOR_NAMES:
            normalized = centered_percentile_rank(
                scored[factor], minimum_population=config.minimum_factor_population
            )
            scored[f"normalized_{factor}"] = normalized
            scored[f"contribution_{factor}"] = (
                normalized.fillna(0.0) * config.weights[factor] * 100.0
            )

        normalized_columns = [f"normalized_{factor}" for factor in RAW_FACTOR_NAMES]
        contribution_columns = [f"contribution_{factor}" for factor in RAW_FACTOR_NAMES]
        coverage = pd.Series(0.0, index=scored.index)
        for factor, weight in config.factor_weights:
            coverage += scored[f"normalized_{factor}"].notna().astype(float) * weight
        scored["score_coverage"] = coverage
        scored["score"] = scored[contribution_columns].sum(axis=1)

        low_coverage = scored["score_coverage"] + 1e-12 < config.minimum_score_coverage
        for symbol in scored.loc[low_coverage, "symbol"]:
            exclusions.append(_Exclusion(str(symbol), ["score_coverage"]))
        scored = scored.loc[~low_coverage].copy()
        if scored.empty:
            return (
                self._empty_candidates(),
                self._exclusion_frame(exclusions),
                reference_session,
            )

        # Stable symbol tie-break makes ranks unique and reproducible.
        scored = scored.sort_values(
            ["score", "symbol"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)
        scored.insert(0, "rank", np.arange(1, len(scored) + 1, dtype=int))

        leading = [
            "rank",
            "symbol",
            "score",
            "score_coverage",
            "sector",
            "industry",
            "market_cap",
            "latest_price",
            "median_dollar_volume_20d",
            "nonzero_volume_fraction_20d",
            "history_sessions",
            "history_coverage",
            "latest_session",
        ]
        raw_columns = list(RAW_FACTOR_NAMES)
        scored = scored[
            leading + raw_columns + normalized_columns + contribution_columns
        ]
        self._validate_candidate_invariants(scored)
        return scored, self._exclusion_frame(exclusions), reference_session

    def _manifest(
        self,
        as_of_utc: pd.Timestamp,
        universe: PreparedFrame,
        prices: PreparedFrame,
        earnings: PreparedFrame,
        input_hash: str,
        component_hashes: dict[str, str],
        candidates: pd.DataFrame,
        exclusions: pd.DataFrame,
        reference_session: str | None,
    ) -> dict[str, Any]:
        config_dict = self.config.as_dict()
        config_hash = sha256_json(config_dict)
        fingerprint_payload = {
            "as_of": as_of_utc.isoformat(),
            "engine_version": __version__,
            "methodology_version": self.config.methodology_version,
            "config_hash": config_hash,
            "input_hash": input_hash,
        }
        try:
            pandas_version = package_version("pandas")
            numpy_version = package_version("numpy")
        except PackageNotFoundError:  # pragma: no cover - normal installs have metadata
            pandas_version = pd.__version__
            numpy_version = np.__version__
        return {
            **fingerprint_payload,
            "run_fingerprint": sha256_json(fingerprint_payload),
            "provider": self.provider.name,
            "reference_session": reference_session,
            "config": config_dict,
            "component_input_hashes": component_hashes,
            "row_counts": {
                "universe_input": universe.input_rows,
                "universe_known": len(universe.frame),
                "prices_input": prices.input_rows,
                "prices_known": len(prices.frame),
                "earnings_input": earnings.input_rows,
                "earnings_known": len(earnings.frame),
                "candidates": len(candidates),
                "exclusions": len(exclusions),
            },
            "discarded_future_rows": {
                "universe": universe.discarded_future_rows,
                "prices": prices.discarded_future_rows,
                "earnings": earnings.discarded_future_rows,
            },
            "runtime": {
                "python": platform.python_version(),
                "pandas": pandas_version,
                "numpy": numpy_version,
            },
        }

    @staticmethod
    def _exclusion_frame(exclusions: list[_Exclusion]) -> pd.DataFrame:
        if not exclusions:
            return pd.DataFrame(columns=["symbol", "reasons"])
        rows = [
            {"symbol": item.symbol, "reasons": "|".join(sorted(set(item.reasons)))}
            for item in exclusions
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("symbol", kind="mergesort")
            .reset_index(drop=True)
        )

    @staticmethod
    def _validate_candidate_invariants(candidates: pd.DataFrame) -> None:
        """Fail closed if score arithmetic or normalization is inconsistent."""

        if candidates["symbol"].duplicated().any():
            raise DataValidationError("candidate symbols must be unique")
        expected_ranks = np.arange(1, len(candidates) + 1, dtype=int)
        if not np.array_equal(candidates["rank"].to_numpy(), expected_ranks):
            raise DataValidationError("candidate ranks must be contiguous")
        if not np.isfinite(candidates["score"]).all():
            raise DataValidationError("candidate scores must be finite")
        if not candidates["score_coverage"].between(0.0, 1.0, inclusive="both").all():
            raise DataValidationError("score coverage must be in [0, 1]")

        normalized_columns = [f"normalized_{factor}" for factor in RAW_FACTOR_NAMES]
        normalized_values = candidates[normalized_columns].to_numpy(dtype=float)
        available_normalized = normalized_values[~np.isnan(normalized_values)]
        if (
            not np.isfinite(available_normalized).all()
            or (
                (available_normalized < -1.0 - 1e-12)
                | (available_normalized > 1.0 + 1e-12)
            ).any()
        ):
            raise DataValidationError("normalized factor values must be in [-1, 1]")

        contribution_columns = [f"contribution_{factor}" for factor in RAW_FACTOR_NAMES]
        contribution_sum = candidates[contribution_columns].sum(axis=1)
        if not np.allclose(
            contribution_sum,
            candidates["score"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise DataValidationError("factor contributions must sum to score")
        if not candidates["score"].between(-100.0, 100.0, inclusive="both").all():
            raise DataValidationError("candidate score must be in [-100, 100]")

    @staticmethod
    def _empty_candidates() -> pd.DataFrame:
        columns = [
            "rank",
            "symbol",
            "score",
            "score_coverage",
            "sector",
            "industry",
            "market_cap",
            "latest_price",
            "median_dollar_volume_20d",
            "nonzero_volume_fraction_20d",
            "history_sessions",
            "history_coverage",
            "latest_session",
            *RAW_FACTOR_NAMES,
            *(f"normalized_{factor}" for factor in RAW_FACTOR_NAMES),
            *(f"contribution_{factor}" for factor in RAW_FACTOR_NAMES),
        ]
        return pd.DataFrame(columns=columns)
