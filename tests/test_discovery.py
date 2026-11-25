import asyncio
import json
from contextlib import asynccontextmanager
from datetime import date, time
from types import SimpleNamespace

import pytest

from dayrank import cli, recorder
from dayrank.data import Bar, at, timestamp, write_json
from dayrank.discovery import _points, discover, premarket_window
from dayrank.recorder import read_recording
from dayrank.universe import parse_directory

DAY = date(2026, 11, 23)
DATES = (
    "2026-11-09",
    "2026-11-10",
    "2026-11-11",
    "2026-11-12",
    "2026-11-13",
    "2026-11-16",
    "2026-11-17",
    "2026-11-18",
    "2026-11-19",
    "2026-11-20",
    "2026-11-23",
)
DIRECTORY = b"""Nasdaq Traded|Symbol|Security Name|Listing Exchange|ETF|Test Issue|NextShares
Y|NEW|Unseeded Inc - Common Stock|Q|N|N|N
Y|HIGH|High Inc - Common Stock|N|N|N|N
Y|QUIET|Quiet Inc - Common Stock|Q|N|N|N
Y|FUND|Famous Fund - Common Stock|N|Y|N|N
Y|WARR|New Inc - Warrants to purchase Common Stock|Q|N|N|N
Y|TEST|Test Inc - Common Stock|Q|N|Y|N
File Creation Time: 1120202621:33|||||
"""


@pytest.fixture
def directory():
    return parse_directory(
        DIRECTORY,
        "fixture",
        timestamp("2026-11-24T00:00:00Z"),
        date(2026, 11, 22),
    )


def minute(day, clock, close, volume):
    return {
        "t": at(day, clock).isoformat(),
        "h": close,
        "l": close,
        "c": close,
        "v": volume,
    }


class Market:
    def __init__(self):
        self.minute_queries = []

    def get(self, host, path, **query):
        assert path == "/v2/calendar"
        return [{"date": day, "open": "09:30"} for day in DATES]

    def stocks(self, endpoint, names, start, end, *, feed, timeframe, **query):
        assert endpoint == "bars" and query["asof"] == "-" and feed == "sip"
        if timeframe == "1Day":
            return {
                name: [
                    minute(day, "00:00", 30 if name == "HIGH" else 4.5, 1_000)
                    for day in DATES[:-1]
                    if start <= at(day, "00:00") <= end
                ]
                for name in names
            }
        day = start.tz_convert("America/New_York").date().isoformat()
        self.minute_queries.append((day, set(names)))
        if day == DAY.isoformat():
            return {
                "NEW": [
                    minute(day, "04:10", 5, 30_000),
                    minute(day, "09:23", 4.6, 5_000),
                    minute(day, "09:24", 6, 50_000),
                ],
                "QUIET": [],
            }
        return {"NEW": [minute(day, "04:10", 4.5, 3_000)]}

    def pages(self, path, **query):
        if path == "/v1beta1/news":
            yield {
                "news": [
                    {
                        "id": 1,
                        "symbols": ["NEW"],
                        "headline": "Unseeded receives FDA approval",
                        "created_at": "2026-11-23T09:00:00Z",
                        "updated_at": "2026-11-23T09:03:00Z",
                    }
                ]
            }
        else:
            assert path == "/v1/corporate-actions"
            yield {"corporate_actions": {}}


def test_broad_discovery_keeps_early_fades_and_excluded_listings(directory):
    market = Market()
    result = discover(market, directory, DAY, lookback=10)
    audit = {row["symbol"]: row for row in result["audit"]}
    assert len(audit) == 6
    assert audit["WARR"]["cutoff_exclusion"] == "non_common_security"
    assert audit["FUND"]["cutoff_exclusion"] == "fund_flag"
    assert audit["TEST"]["cutoff_exclusion"] == "test_issue"
    assert audit["HIGH"]["cutoff_exclusion"] == "prior_close_above_possible_gap_price"
    assert audit["QUIET"]["cutoff_exclusion"] == "no_completed_premarket_bars"
    assert audit["NEW"]["cutoff_exclusion"] == "gap"
    assert audit["NEW"]["cutoff_price"] == 4.6
    assert audit["NEW"]["cutoff_rules"] == ""
    assert audit["NEW"]["first_rvol_at"] == "2026-11-23T09:11:35+00:00"
    assert {row["rule"] for row in result["alerts"]} == {
        "gap-only",
        "rvol",
        "catalyst-rvol",
    }
    assert all(
        row["symbol"] == "NEW" and row["relative_volume"] == 10
        for row in result["alerts"]
    )
    assert market.minute_queries[0] == (DAY.isoformat(), {"NEW", "QUIET"})
    assert all(names == {"NEW"} for _, names in market.minute_queries[1:])


def test_directory_rejects_future_publication_or_stale_live_roster(directory):
    with pytest.raises(ValueError, match="later"):
        parse_directory(
            DIRECTORY, "fixture", timestamp("2026-11-24T00:00:00Z"), date(2026, 11, 19)
        )
    with pytest.raises(ValueError, match="creation timestamp"):
        parse_directory(
            DIRECTORY.split(b"File Creation Time:")[0], "fixture", at(str(DAY), "00:00")
        )
    stale = {
        **directory,
        "available_at": at(str(DAY), "03:50").isoformat(),
        "published_at": at("2026-11-19", "22:00").isoformat(),
    }
    with pytest.raises(ValueError, match="fresh directory"):
        discover(Market(), stale, DAY, lookback=10)


def test_feeds_can_be_compared_over_the_same_window():
    sip = premarket_window(Market(), DAY, "sip", 10, start_at=time(8, 0))
    iex = premarket_window(Market(), DAY, "iex", 10)
    assert (sip.start, sip.cutoff) == (iex.start, iex.cutoff)
    with pytest.raises(ValueError, match="08:00"):
        premarket_window(Market(), DAY, "iex", 10, start_at=time(7, 59))


def test_receipt_tape_preserves_original_alert_and_replaces_revision(
    directory, tmp_path
):
    day = DAY.isoformat()
    file = tmp_path / "recording.jsonl"
    initial = minute(day, "04:10", 5, 30_000)
    revision = minute(day, "04:10", 4.5, 2_000)
    start = {
        "type": "start",
        "schema": 1,
        "date": day,
        "feed": "sip",
        "directory_sha256": directory["sha256"],
        "started_at": at(day, "03:59").isoformat(),
    }
    records = [
        {
            "type": "bar",
            "received_at": "2026-11-23T09:11:00.584Z",
            "event": {**initial, "T": "b", "S": "NEW"},
        },
        {
            "type": "bar",
            "received_at": "2026-11-23T09:11:31Z",
            "event": {**revision, "T": "u", "S": "NEW"},
        },
    ]

    def save(*rows):
        file.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    save(start, *records)
    with pytest.raises(ValueError, match="successful completion"):
        read_recording(file)
    finish = {
        "type": "complete",
        "records": 2,
        "finished_at": at(day, "09:25").isoformat(),
    }
    save(start, *records[::-1], finish)
    with pytest.raises(ValueError, match="out of order"):
        read_recording(file)
    save(start, *records, finish)
    tape = read_recording(file)
    result = discover(Market(), directory, DAY, lookback=10, recorded=tape)
    assert result["manifest"]["observation"] == "recorded_receipt"
    assert result["alerts"][0]["observed_at"] == "2026-11-23T09:11:00.584000+00:00"
    assert result["audit"][0]["cutoff_dollar_volume"] == 9_000
    assert result["audit"][0]["cutoff_rules"] == ""
    window = premarket_window(Market(), DAY, "sip", 10)
    with pytest.raises(ValueError, match="before its minute"):
        _points([(at(day, "04:10"), Bar.read(initial))], 4.5, window)
    save(start, *records, {**finish, "finished_at": at(day, "09:24").isoformat()})
    with pytest.raises(ValueError, match="full window"):
        discover(Market(), directory, DAY, lookback=10, recorded=read_recording(file))


def test_cli_writes_auditable_outputs(directory, tmp_path, monkeypatch):
    roster = tmp_path / "directory.json"
    out = tmp_path / "outputs"
    write_json(roster, directory)
    monkeypatch.setattr(cli, "_client", lambda args: Market())
    assert (
        cli.main(
            [
                "discover",
                "--date",
                DAY.isoformat(),
                "--universe",
                str(roster),
                "--history-sessions",
                "10",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert {file.name for file in out.iterdir()} == {
        "symbols.csv",
        "alerts.csv",
        "manifest.json",
        "eligible-symbols.txt",
    }
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["universe_json_sha256"]) == 64
    assert manifest["listing_rows"] == 6
    assert (out / "eligible-symbols.txt").read_text(encoding="utf-8") == ""


def test_recorder_writes_only_frozen_names_and_publishes_completed_file(
    directory,
    tmp_path,
    monkeypatch,
):
    day = DAY.isoformat()
    original_time = timestamp("2026-11-23T09:11:00.584Z")
    clock = iter(
        [
            at(day, "03:59:30"),
            at(day, "03:59:31"),
            at(day, "04:11"),
            original_time,
            at(day, "09:25"),
            at(day, "09:25"),
        ]
    )
    monkeypatch.setattr(
        recorder,
        "pd",
        SimpleNamespace(Timestamp=SimpleNamespace(now=lambda **kw: next(clock))),
    )
    market = Market()
    out = tmp_path / "stream.jsonl"
    partial = tmp_path / "stream.jsonl.partial"
    window = premarket_window(market, DAY, "sip", 10)
    bar = minute(day, "04:10", 5, 30_000)

    async def recv():
        assert partial.exists() and not out.exists()
        return json.dumps([{**bar, "T": "b", "S": name} for name in ("FUND", "NEW")])

    @asynccontextmanager
    async def subscribe(client, feed):
        assert client is market and feed == "sip"
        yield SimpleNamespace(recv=recv)

    monkeypatch.setattr(recorder, "subscribed_bars", subscribe)
    assert asyncio.run(recorder.record(market, directory, window, "sip", out)) == 1
    assert not partial.exists()
    header, events = read_recording(out)
    assert set(events) == {"NEW"}
    assert events["NEW"][0][0] == original_time
    assert header["finished_at"] == window.cutoff.isoformat()
