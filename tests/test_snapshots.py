import csv
import gzip
from datetime import date, time

import pytest

from dayrank import cli
from dayrank.data import at, write_json
from dayrank.snapshots import _add_peers, collect_snapshots

DAY = date(2026, 11, 23)
PRIOR = "2026-11-20"


@pytest.fixture
def directory():
    return {
        "schema": 1,
        "source": "fixture",
        "available_at": at(PRIOR, "22:00").isoformat(),
        "published_at": at(PRIOR, "22:00").isoformat(),
        "sha256": "fixture",
        "rows": [
            {"symbol": symbol, "listing_reasons": reason}
            for symbol, reason in [
                ("ALFA", ""),
                ("BETA", ""),
                ("QUIET", ""),
                ("NEW", ""),
                ("BOND", "non_common_security"),
            ]
        ],
    }


def minute(clock, close):
    return {
        "t": at(str(DAY), clock).isoformat(),
        "o": close,
        "c": close,
        "l": close * 0.99,
        "h": close * 1.01,
        "v": 10,
        "vw": close,
        "n": 2,
    }


class Market:
    def __init__(self):
        self.calls = []

    def get(self, host, path, **query):
        assert path == "/v2/calendar"
        return [
            {"date": prior, "open": "09:30", "close": "16:00"}
            for prior in (PRIOR, str(DAY))
        ]

    def stocks(self, endpoint, names, start, end, **params):
        self.calls.append((endpoint, names, start, end, params))
        assert "BOND" not in names and params["feed"] == "sip"
        if endpoint == "bars" and params["timeframe"] == "1Day":
            return {
                name: [
                    {
                        "t": at(PRIOR, "00:00").isoformat(),
                        "c": price,
                        "v": 10,
                        "vw": price,
                    }
                ]
                for name, price in {"ALFA": 10, "BETA": 500, "QUIET": 3}.items()
            }
        if endpoint == "bars":
            return {
                "ALFA": [
                    minute("08:13", 10),
                    minute("08:14", 12),
                    minute("08:16", 100),
                ],
                "BETA": [minute("08:10", 490)],
                "NEW": [minute("08:10", 2)],
            }
        return {
            "ALFA": [
                dict(
                    t=at(str(DAY), clock).isoformat(),
                    bp=bid,
                    ap=ask,
                    bs=50,
                    **{"as": 40},
                )
                for clock, bid, ask in [
                    ("08:15", 10, 10.1),
                    ("08:16", 10.2, 10.18),
                    ("08:17", 20, 25),
                ]
            ]
        }


def test_fixed_snapshots_keep_quiet_expensive_and_unseasoned_stocks(directory):
    market = Market()
    result = collect_snapshots(
        market,
        directory,
        DAY,
        feed="sip",
        clocks=[time(8, 16), time(8, 15)],
        quote_sample_size=4,
    )
    manifest = result["manifest"]
    assert len(result["listings"]) == 5
    assert (manifest["eligible_listings"], manifest["snapshot_rows"]) == (4, 8)
    rows = result["rows"]
    before = {r["symbol"]: r for r in rows[:4]}
    after = {r["symbol"]: r for r in rows[4:]}
    assert (
        before["QUIET"]["bar_status"] == "no_bar"
        and before["QUIET"]["last_price"] is None
    )
    assert before["QUIET"]["quote_status"] == "no_quote"
    assert (
        before["BETA"]["previous_close"] == 500
        and before["BETA"]["bar_status"] == "stale"
    )
    assert before["NEW"]["last_price"] == 2 and before["NEW"]["previous_close"] is None
    assert before["ALFA"]["last_price"] == 10 and before["ALFA"]["session_bars"] == 1
    assert before["ALFA"]["last_bar_observed_at"] == "2026-11-23T13:14:35+00:00"
    assert after["ALFA"]["last_price"] == 12 and after["ALFA"]["session_bars"] == 2
    assert after["ALFA"]["trailing_5m_trade_count"] == 4
    assert after["ALFA"]["trailing_5m_close_return_bps"] == pytest.approx(2_000)
    assert before["ALFA"]["quote_status"] == "two_sided"
    assert (
        after["ALFA"]["quote_status"] == "invalid_book"
        and after["ALFA"]["spread_bps"] is None
    )
    assert [call[0] for call in market.calls] == ["bars", "bars", "quotes", "quotes"]


def test_peer_strength_excludes_own_return_and_stale_observations():
    rows = [
        {
            "peer_cohort": "1:2",
            "bar_status": "fresh",
            "return_since_previous_close_bps": value,
        }
        for value in [10, 20, 30, 40, 50, 60, 20]
    ]
    rows.append(
        {
            "peer_cohort": "1:2",
            "bar_status": "stale",
            "return_since_previous_close_bps": 1_000,
        }
    )
    _add_peers(rows)
    assert rows[1]["peer_observations"] == 6
    assert rows[1]["peer_median_return_bps"] == 35
    assert rows[1]["excess_vs_peer_median_bps"] == -15
    assert rows[-1]["peer_observations"] == 7
    assert rows[-1]["excess_vs_peer_median_bps"] is None


def test_cli_writes_gzip_and_quote_sample_stays_fixed(directory, tmp_path, monkeypatch):
    market = Market()
    monkeypatch.setattr(cli, "_client", lambda args: market)
    frozen = tmp_path / "universe.json"
    write_json(frozen, directory)
    output = tmp_path / "result"
    assert (
        cli.main(
            [
                "snapshots",
                "--date",
                str(DAY),
                "--universe",
                str(frozen),
                "--at",
                "08:15",
                "--at",
                "08:16",
                "--quote-sample-size",
                "1",
                "--out",
                str(output),
            ]
        )
        == 0
    )
    quotes = [call[1] for call in market.calls if call[0] == "quotes"]
    assert quotes[0] == quotes[1] and len(quotes[0]) == 1
    with gzip.open(
        output / "snapshots.csv.gz", "rt", encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    assert {row["symbol"] for row in rows if row["quote_status"] != "unsampled"} == set(
        quotes[0]
    )
    assert {row["quote_inclusion_probability"] for row in rows} == {"0.25"}
    assert (output / "manifest.json").exists() and (output / "listings.csv").exists()
