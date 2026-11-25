import csv
import gzip
from dataclasses import replace
from datetime import date, timedelta

import pytest

from dayrank import cli
from dayrank.anomalies import (
    METRICS,
    AnomalyConfig,
    Observation,
    SessionSample,
    detect,
    rank_alerts,
    read_sample,
)
from dayrank.data import at, write_json


def observation(
    day,
    symbol,
    *,
    clock="09:25",
    cohort="0:0",
    status="fresh",
    bars=5,
    short=0.0,
    gap=50.0,
    dollars=10_000.0,
    trades=20.0,
):
    return Observation(
        symbol,
        at(day.isoformat(), clock).isoformat(),
        clock,
        cohort,
        status,
        "unsampled",
        None,
        bars,
        dict(short_return=short, gap=gap, dollars=dollars, trades=trades),
    )


def sample(day, rows, *, feed="sip"):
    return SessionSample(
        dict(
            schema=1,
            date=day.isoformat(),
            feed=feed,
            bar_lag_seconds=35,
            observation="historical_completion_plus_lag",
            price_edges=[1, 5, 20, 100],
            prior_dollar_volume_edges=[100_000],
            feature_windows=[5, 15, 30],
            snapshot_rows=len(rows),
            eligible_listings=len(rows),
            coverage=[
                dict(decision_at=when)
                for when in sorted({row.decision for row in rows})
            ],
        ),
        rows,
    )


@pytest.fixture
def market_samples():
    current = date(2026, 11, 23)
    cohort = [f"P{i:02}" for i in range(11)]
    shocked = [f"M{i:02}" for i in range(11)]
    regular = ["MOVE", "DOLLAR", "THIN", "GAP", "SPLT", "LATER", "STALE"]
    historical = []
    for age in range(20, 0, -1):
        day = current - timedelta(days=age)
        typical = (age % 3 - 1) * 10
        historic = [
            observation(
                day, name, clock="09:35" if name == "LATER" else "09:25", short=typical
            )
            for name in regular + cohort
        ] + [observation(day, name, cohort="4:2", short=typical) for name in shocked]
        historical.append(sample(day, historic))
    rows = [
        observation(current, "MOVE", short=550),
        observation(current, "DOLLAR", dollars=400_000, trades=300),
        observation(current, "THIN", short=700, bars=1, trades=2),
        observation(current, "GAP", gap=2_500),
        observation(current, "SPLT", short=-600, gap=-5_000),
        observation(current, "LATER", short=550),
        observation(current, "STALE", short=550, status="stale"),
    ]
    rows += [observation(current, name) for name in cohort]
    rows += [observation(current, name, cohort="4:2", short=400) for name in shocked]
    return sample(current, rows), historical


def test_learns_same_clock_and_separates_behaviors(market_samples):
    target, history = market_samples
    result = detect(target, history, AnomalyConfig())
    rank_alerts(result, 15, set(), "not_requested")
    alerts = {(row["entity"], row["kind"]) for row in result["alerts"]}
    assert ("MOVE", "price_dislocation") in alerts
    assert ("DOLLAR", "participation_surge") in alerts
    assert ("4:2", "breadth_shift") in alerts
    assert all(not name.startswith("M0") for name, _ in alerts)
    assert ("GAP", "gap_dislocation") not in alerts
    audit = {row["symbol"]: row for row in result["audit"]}
    assert audit["MOVE"]["short_return_available_prior_sessions"] == 20
    assert audit["THIN"]["data_quality"] == "thin_activity"
    assert audit["STALE"]["data_quality"] == "stale"
    assert audit["LATER"]["data_quality"] == "insufficient_history"
    by_name = {(r["entity"], r["kind"]): r for r in result["candidates"]}
    assert by_name["GAP", "gap_dislocation"]["alert_status"] == "requires_split_check"
    assert by_name["M00", "price_dislocation"]["alert_status"] == "peer_aligned"


def test_split_exclusion_rank_budget_and_no_future_inputs(market_samples):
    target, history = market_samples
    config = AnomalyConfig()
    result = detect(target, history, config)
    rank_alerts(result, 2, {"SPLT"}, "available")
    assert len(result["alerts"]) == 2
    assert result["alerts"][0]["rank"] == 1
    assert all(
        row["alert_status"] == "split_effective"
        for row in result["candidates"]
        if row["entity"] == "SPLT"
    )
    assert ("GAP", "gap_dislocation") in {
        (r["entity"], r["kind"]) for r in result["alerts"]
    }
    with pytest.raises(ValueError, match="precede"):
        detect(target, [*history, target], config)
    different_feed = replace(
        history[-1], manifest={**history[-1].manifest, "feed": "iex"}
    )
    with pytest.raises(ValueError, match="same feed"):
        detect(target, [*history[:-1], different_feed], config)


def _save_sample(path, value):
    write_json(path / "manifest.json", value.manifest)
    with gzip.open(
        path / "snapshots.csv.gz", "wt", encoding="utf-8", newline=""
    ) as file:
        columns = [
            "date",
            "feed",
            "symbol",
            "decision_at",
            "peer_cohort",
            "bar_status",
            "quote_status",
            "spread_bps",
            "trailing_5m_bars",
            *METRICS.values(),
        ]
        writer = csv.DictWriter(file, columns)
        writer.writeheader()
        for row in value.rows:
            writer.writerow(
                dict(
                    date=value.manifest["date"],
                    feed=value.manifest["feed"],
                    symbol=row.symbol,
                    decision_at=row.decision,
                    peer_cohort=row.cohort,
                    bar_status=row.bar_status,
                    quote_status=row.quote_status,
                    spread_bps=row.spread,
                    trailing_5m_bars=row.bars,
                    **{col: row.values[name] for name, col in METRICS.items()},
                )
            )


def test_cli_reads_snapshot_files_and_runs_without_credentials(tmp_path, monkeypatch):
    current = date(2026, 11, 23)
    paths = []
    for age in range(12, -1, -1):
        day = current - timedelta(days=age)
        path = tmp_path / day.isoformat()
        _save_sample(
            path, sample(day, [observation(day, "TEST", short=0 if age else 450)])
        )
        paths.append(str(path))
    monkeypatch.setattr(
        cli, "_client", lambda _: pytest.fail("Offline scoring requested credentials")
    )
    output = tmp_path / "out"
    assert (
        cli.main(
            [
                "anomalies",
                "--target",
                paths[-1],
                "--history",
                *paths[:-1],
                "--minimum-history",
                "10",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    with (output / "alerts.csv").open(encoding="utf-8", newline="") as file:
        alerts = list(csv.DictReader(file))
    assert [(row["entity"], row["kind"]) for row in alerts] == [
        ("TEST", "price_dislocation")
    ]
    bad = tmp_path / "bad"
    last = read_sample(tmp_path / current.isoformat())
    _save_sample(bad, replace(last, rows=last.rows * 2))
    with pytest.raises(ValueError, match="duplicate"):
        read_sample(bad)
