from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from dayrank.alpaca import DATA, PAPER, Alpaca, capture
from dayrank.cli import main
from dayrank.data import Session, at, write_json
from dayrank.replay import ReplayConfig, replay
from dayrank.scan import scan


def make_session(day="2026-11-20", settlement="2026-11-23"):
    def bar(clock, high, close, volume=100):
        return dict(t=at(day, clock).isoformat(), h=high, l=4.8, c=close, v=volume)

    def quote(clock, bid, ask, size=1000):
        return {
            "t": at(day, clock).isoformat(),
            "bp": bid,
            "ap": ask,
            "bs": size,
            "as": size,
        }

    def headline(head, created, changed):
        return {
            "headline": head,
            "created_at": at(day, created).isoformat(),
            "updated_at": at(day, changed).isoformat(),
        }

    return dict(
        schema=1,
        date=day,
        settlement_date=settlement,
        quote_size_multiplier=1,
        previous_sessions=pd.bdate_range(
            end=pd.Timestamp(day) - pd.offsets.BDay(), periods=10
        )
        .strftime("%Y-%m-%d")
        .tolist(),
        premarket=at(day, "04:00").isoformat(),
        scan_at=at(day, "09:25").isoformat(),
        market_open=at(day, "09:30").isoformat(),
        market_close=at(day, "16:00").isoformat(),
        observed_through=at(day, "10:45").isoformat(),
        feed="sip",
        provenance={"news_status": "available"},
        symbols={
            "TEST": dict(
                previous_close=4.5,
                history_volumes=[10_000] * 10,
                split=False,
                execution_until=at(day, "10:45").isoformat(),
                bars=[
                    bar("09:24", 5.0, 5.0, 30_000),
                    bar("09:25", 6.0, 6.0, 10_000),
                    *(bar(f"09:{minute}", 5.10, 5.0) for minute in range(30, 35)),
                    bar("09:35", 5.15, 5.12),
                ],
                headlines=[
                    headline("Issuer reports FDA approval", "09:00", "09:00"),
                    headline("Issuer reports a new offering", "09:10", "09:31"),
                ],
                quotes=[
                    quote("09:24:50", 4.98, 0),
                    quote("09:36:00.250", 5.11, 5.12),
                    quote("09:36:00.251", 5.11, 5.12),
                    quote("10:30:01", 5.40, 5.41),
                ],
            )
        },
    )


def test_watchlist_uses_completed_bars_and_timestamped_headlines():
    session = Session.read(make_session())
    row = scan(session, "catalyst-rvol")[0]
    assert row["eligible"]
    assert row["relative_volume"] == 3
    assert row["premarket_high"] == 5
    assert row["catalyst"] == "material_keyword"
    assert row["quote_status"] == "one_sided_or_crossed"
    source = make_session()
    source["symbols"]["TEST"]["history_volumes"] = [None] * 10
    incomplete = Session.read(source)
    assert scan(incomplete, "gap-only")[0]["eligible"]
    assert scan(incomplete, "rvol")[0]["reasons"] == "rvol_unavailable"


def test_replay_after_latency_and_settlement_rules():
    friday = Session.read(make_session())
    monday = Session.read(make_session("2026-11-23", "2026-11-24"))
    config = ReplayConfig(
        capital=1000, risk_fraction=1, max_exposure_fraction=1, cost_bps=0
    )
    result = replay([monday, friday], "rvol", "opening-range", config)
    first, second = result["ledger"]
    assert first["status"] == second["status"] == "closed"
    assert pd.Timestamp(first["entry_at"]) == at("2026-11-20", "09:36:00.251")
    assert first["shares"] == 195
    assert first["pnl"] == pytest.approx(54.60)
    assert result["daily"][0]["settled_cash"] == pytest.approx(1.60)
    assert result["daily"][0]["unsettled_cash"] == pytest.approx(1053)
    assert second["shares"] > first["shares"]
    capped = replay(
        [friday, monday], "rvol", "opening-range", ReplayConfig(round_trips_per_five=1)
    )
    assert capped["ledger"][1]["status"] == "five_session_trade_limit"


def test_wide_quote_and_unresolved_partial_exit():
    source = make_session()
    row = source["symbols"]["TEST"]
    for quote in row["quotes"][1:3]:
        quote["bp"] = 4.9
    result = replay([Session.read(source)], "gap-only", "opening-range", ReplayConfig())
    assert result["ledger"][0]["status"] == "spread_too_large_for_stop"
    source = make_session()
    quotes = source["symbols"]["TEST"]["quotes"]
    quotes[-1]["bs"] = 20
    quotes.append(
        dict(quotes[-1], t=at(source["date"], "10:30:02").isoformat(), ap=5.42)
    )
    result = replay([Session.read(source)], "gap-only", "opening-range", ReplayConfig())
    fill = result["ledger"][0]
    assert fill["status"] == "open_unresolved"
    assert fill["remaining_shares"] == fill["shares"] - 20
    assert result["summary"]["return_fraction"] is None


def test_cli_writes_comparison_artifacts_and_rejects_scan_only_replay(tmp_path):
    source = make_session()
    source["symbols"]["TEST"]["execution_until"] = None
    snapshot = tmp_path / "session.json"
    write_json(snapshot, source)
    assert main(["scan", str(snapshot), "--out", str(tmp_path / "watch.csv")]) == 0
    result_path = tmp_path / "replay"
    assert main(["replay", str(snapshot), "--out", str(result_path)]) == 0
    manifest = json.loads((result_path / "manifest.json").read_text())
    assert len(manifest["variants"]) == 6
    assert (
        "insufficient_execution_capture"
        in (result_path / "trade-ledger.csv").read_text()
    )


def test_capture_same_feed_and_historical_quote_units(monkeypatch):
    client = Alpaca("fixture-key", "fixture-secret")
    trade_date = date(2025, 10, 31)
    days = pd.bdate_range("2025-10-16", "2025-11-04").strftime("%Y-%m-%d").tolist()
    hits = []

    def get(host, path, **query):
        hits.append((host, path, query))
        if host == PAPER:
            return [
                dict(
                    date=day, open="09:30", close="16:00", settlement_date="2025-11-03"
                )
                for day in days
            ]
        if path == "/v2/stocks/bars":
            day = (
                pd.Timestamp(query["start"])
                .tz_convert("America/New_York")
                .date()
                .isoformat()
            )
            if query["timeframe"] == "1Day":
                bars = [{"t": at(d, "00:00").isoformat(), "c": 4.5} for d in days[:-3]]
            else:
                one = {
                    "t": at(day, "09:24").isoformat(),
                    "h": 5,
                    "l": 4.8,
                    "c": 5,
                    "v": 50_000 if day == str(trade_date) else 10_000,
                }
                bars = [one]
            return {"bars": {"TEST": bars}}
        if host == DATA and path == "/v2/stocks/quotes":
            return {
                "quotes": {
                    "TEST": [
                        {
                            "t": at(str(trade_date), "09:24:50").isoformat(),
                            "bp": 4.99,
                            "ap": 5,
                            "bs": 2,
                            "as": 3,
                        }
                    ]
                }
            }
        return {}

    monkeypatch.setattr(client, "get", get)
    result = capture(client, trade_date, ["TEST"], history_sessions=10, scan_only=True)
    session = Session.read(result)
    assert session.symbols["TEST"].quotes[-1].ask_size == 300
    assert session.settlement_date == "2025-11-03"
    assert scan(session, "rvol")[0]["relative_volume"] == 5
    assert all(
        q["feed"] == "sip" for _, path, q in hits if path.startswith("/v2/stocks/")
    )
