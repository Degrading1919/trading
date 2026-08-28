from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from swingrank.engine import ScreeningEngine
from swingrank.providers.memory import InMemoryDataProvider
from tests.conftest import AS_OF, make_earnings, make_prices, make_universe


def test_engine_score_is_sum_of_fixed_contributions_and_is_explainable(
    complete_provider: InMemoryDataProvider,
) -> None:
    result = ScreeningEngine(complete_provider).run(AS_OF)
    assert len(result.candidates) == 10
    contribution_columns = [
        column for column in result.candidates if column.startswith("contribution_")
    ]
    pdt.assert_series_equal(
        result.candidates[contribution_columns].sum(axis=1),
        result.candidates["score"],
        check_names=False,
        atol=1e-10,
        rtol=1e-10,
    )
    assert (result.candidates["score_coverage"] == 1.0).all()
    assert result.candidates["rank"].tolist() == list(range(1, 11))
    assert result.manifest["engine_version"] == "0.1.0"
    assert result.manifest["methodology_version"] == "1.0.0"
    for factor, weight in result.manifest["config"]["factor_weights"].items():
        expected = (
            result.candidates[f"normalized_{factor}"].fillna(0.0) * weight * 100.0
        )
        assert np.allclose(result.candidates[f"contribution_{factor}"], expected)


def test_missing_earnings_is_neutral_not_redistributed(symbols: list[str]) -> None:
    provider = InMemoryDataProvider(make_universe(symbols), make_prices(symbols))
    result = ScreeningEngine(provider).run(AS_OF)
    assert len(result.candidates) == len(symbols)
    assert result.candidates["normalized_earnings_sue"].isna().all()
    assert (result.candidates["contribution_earnings_sue"] == 0.0).all()
    assert np.allclose(result.candidates["score_coverage"], 0.80)


def test_eligibility_filters_report_specific_reasons() -> None:
    passing = [f"PASS{index}" for index in range(6)]
    failing = ["PENNY", "ILLIQ", "SMALL", "FUND"]
    symbols = passing + failing
    universe = make_universe(symbols)
    universe.loc[universe["symbol"] == "SMALL", "market_cap"] = 100_000_000.0
    universe.loc[universe["symbol"] == "FUND", "security_type"] = "ETF"
    prices = make_prices(symbols)
    penny_latest = prices["symbol"].eq("PENNY") & prices["session_date"].eq(
        prices["session_date"].max()
    )
    prices.loc[penny_latest, "close"] = 4.0
    illiquid_recent = prices["symbol"].eq("ILLIQ") & prices["session_date"].isin(
        sorted(prices["session_date"].unique())[-20:]
    )
    prices.loc[illiquid_recent, "volume"] = 100.0

    result = ScreeningEngine(InMemoryDataProvider(universe, prices)).run(AS_OF)
    assert set(result.candidates["symbol"]) == set(passing)
    reasons = result.exclusions.set_index("symbol")["reasons"].to_dict()
    assert "price" in reasons["PENNY"]
    assert "dollar_volume" in reasons["ILLIQ"]
    assert "market_cap" in reasons["SMALL"]
    assert "security_type" in reasons["FUND"]


def test_inactive_latest_universe_version_is_not_resurrected(
    symbols: list[str],
) -> None:
    universe = make_universe(symbols)
    prior = universe.loc[universe["symbol"] == symbols[0]].iloc[0].copy()
    inactive = prior.copy()
    inactive["available_at"] = AS_OF - pd.Timedelta(hours=1)
    inactive["effective_to"] = "2024-12-31"
    universe = pd.concat([universe, inactive.to_frame().T], ignore_index=True)

    result = ScreeningEngine(
        InMemoryDataProvider(universe, make_prices(symbols), make_earnings(symbols))
    ).run(AS_OF)
    assert symbols[0] not in set(result.candidates["symbol"])
    reason = result.exclusions.set_index("symbol").loc[symbols[0], "reasons"]
    assert reason == "inactive_as_of"


def test_future_rows_cannot_change_an_earlier_rank_or_fingerprint(
    symbols: list[str],
) -> None:
    universe = make_universe(symbols)
    prices = make_prices(symbols)
    earnings = make_earnings(symbols)
    baseline = ScreeningEngine(InMemoryDataProvider(universe, prices, earnings)).run(
        AS_OF
    )

    future_universe = universe.iloc[0].copy()
    future_universe["available_at"] = AS_OF + pd.Timedelta(days=1)
    future_universe["market_cap"] = 99_000_000_000.0
    augmented_universe = pd.concat(
        [universe, future_universe.to_frame().T], ignore_index=True
    )

    future_price = prices.iloc[0].copy()
    future_price["available_at"] = AS_OF + pd.Timedelta(days=1)
    future_price["total_return_index"] = 1_000_000.0
    augmented_prices = pd.concat([prices, future_price.to_frame().T], ignore_index=True)

    future_earnings = earnings.iloc[-1].copy()
    future_earnings["available_at"] = AS_OF + pd.Timedelta(days=1)
    future_earnings["announced_at"] = AS_OF + pd.Timedelta(hours=12)
    future_earnings["reported_eps"] = 1_000.0
    augmented_earnings = pd.concat(
        [earnings, future_earnings.to_frame().T], ignore_index=True
    )

    augmented = ScreeningEngine(
        InMemoryDataProvider(augmented_universe, augmented_prices, augmented_earnings)
    ).run(AS_OF)
    pdt.assert_frame_equal(baseline.candidates, augmented.candidates)
    assert baseline.manifest["input_hash"] == augmented.manifest["input_hash"]
    assert baseline.manifest["run_fingerprint"] == augmented.manifest["run_fingerprint"]
    assert sum(augmented.manifest["discarded_future_rows"].values()) == 3


def test_input_order_does_not_change_outputs_or_hashes(symbols: list[str]) -> None:
    universe = make_universe(symbols)
    prices = make_prices(symbols)
    earnings = make_earnings(symbols)
    ordered = ScreeningEngine(InMemoryDataProvider(universe, prices, earnings)).run(
        AS_OF
    )
    shuffled = ScreeningEngine(
        InMemoryDataProvider(
            universe.sample(frac=1.0, random_state=11).reset_index(drop=True),
            prices.sample(frac=1.0, random_state=12).reset_index(drop=True),
            earnings.sample(frac=1.0, random_state=13).reset_index(drop=True),
        )
    ).run(AS_OF)
    pdt.assert_frame_equal(ordered.candidates, shuffled.candidates)
    pdt.assert_frame_equal(ordered.exclusions, shuffled.exclusions)
    assert ordered.manifest["run_fingerprint"] == shuffled.manifest["run_fingerprint"]
