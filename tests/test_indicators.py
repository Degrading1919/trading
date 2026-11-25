import csv
import gzip
import json

import pytest

from dayrank import cli
from dayrank.data import at
from dayrank.indicators import IndicatorRules, evaluate_tape, summarize

DAY = "2026-11-23"


def candle(clock, close, high=5, low=4.7, volume=1_000, trades=12):
    return dict(
        t=at(DAY, clock).isoformat(),
        o=close,
        c=close,
        l=low,
        h=high,
        v=volume,
        n=trades,
        vw=close,
    )


@pytest.fixture
def tape():
    bars = [candle(f"08:0{i}", 4.8) for i in range(5)]
    bars.extend(
        [
            candle("08:05", 5.1, high=5.2, volume=3_000, trades=40),
            candle("08:06", 5.15, high=5.2),
            candle("08:08", 5.05, high=5.2),
            candle("08:11", 10, high=10.1),
            *[candle(f"09:3{i}", 4.8, high=5.1) for i in range(5)],
            candle("09:35", 5.25, high=5.3, volume=3_000, trades=50),
            candle("09:36", 5.3, high=5.35),
        ]
    )
    return dict(
        schema=1,
        date=DAY,
        feed="sip",
        premarket_only=False,
        bar_lag_seconds=35,
        start=at(DAY, "08:00").isoformat(),
        cutoff=at(DAY, "08:12").isoformat(),
        open=at(DAY, "09:30").isoformat(),
        through=at(DAY, "10:06").isoformat(),
        listing_rows=20,
        eligible_listings=10,
        split_status="available",
        directory_sha256="fixture",
        scan_config=dict(
            minimum_price=1, maximum_price=20, minimum_dollar_volume=1_000
        ),
        previous_closes={"NEW": 4.5},
        bars={"NEW": bars},
    )


def test_case_timestamps_freeze_levels_and_keep_outcomes_separate(tape):
    cases, forward, coverage = evaluate_tape(tape, IndicatorRules())
    rolling = {r["case"]: r for r in cases if r["pattern"] == "premarket_rolling_high"}
    assert rolling["close_break"]["observation"] == "2026-11-23T13:06:35+00:00"
    assert rolling["hold_confirmed"]["observation"] == "2026-11-23T13:07:35+00:00"
    assert rolling["close_break"]["prior_5m_share_volume"] == 5_000
    assert rolling["close_break"]["volume_confirmed"]
    assert not any("return" in key for key in rolling["close_break"])
    outcomes = {row["case_id"]: row for row in forward}
    assert outcomes[rolling["close_break"]["case_id"]]["close_return_1m_bps"] > 0
    assert outcomes[rolling["close_break"]["case_id"]]["close_return_5m_bps"] is None
    after_open = {
        row["pattern"]: row["level"] for row in cases if row["case"] == "close_break"
    }
    assert after_open == {
        "premarket_rolling_high": 5,
        "premarket_high_at_open": 5.2,
        "opening_range_high": 5.1,
    }
    assert coverage["symbols_with_premarket"] == 1
    assert coverage["listing_rows"] == 20


def test_review_reproducibility_and_offline_premarket_path(tape, tmp_path, monkeypatch):
    rules = IndicatorRules()
    result = evaluate_tape({**tape, "premarket_only": True}, rules)
    summary = summarize([result], rules, 2)
    assert len(summary["review"]) == 2
    assert all(
        row["review_label"] == "" and row["pattern"] == "premarket_rolling_high"
        for row in summary["review"]
    )
    assert summary == summarize([result], rules, 2)
    with pytest.raises(ValueError, match="each date and feed once"):
        summarize([result, result], rules, 5)
    file = tmp_path / "tape.json.gz"
    with gzip.open(file, "wt", encoding="utf-8") as handle:
        json.dump({**tape, "premarket_only": True}, handle)

    def forbidden_client(args):
        pytest.fail("Offline inputs must not initialize an Alpaca session")

    monkeypatch.setattr(cli, "_client", forbidden_client)
    out = tmp_path / "out"
    assert (
        cli.main(["indicators", str(file), "--review-size", "2", "--out", str(out)])
        == 0
    )
    with (out / "review.csv").open(encoding="utf-8") as handle:
        review = list(csv.DictReader(handle))
    assert {row["case_id"] for row in review} == {
        row["case_id"] for row in summary["review"]
    }
