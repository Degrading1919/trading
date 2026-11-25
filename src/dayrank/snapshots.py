"""Sample the exchange universe on fixed clocks without indicator admission rules."""

from __future__ import annotations

import hashlib
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import date, time, timedelta

import pandas as pd

from dayrank.alpaca import PAPER, Alpaca
from dayrank.data import ET, MINUTE, SECOND, Candle, Quote, at, number, timestamp
from dayrank.universe import directory_rows

CLOCKS = ("08:15", "08:45", "09:15", "09:25", "09:35", "09:45")
WINDOWS = (5, 15, 30)
PRICE_EDGES = (1, 5, 20, 100)
NOTIONAL_EDGES = (100_000, 1_000_000, 10_000_000)
MAX_PEER_AGE = 120 * SECOND
QUOTE_WINDOW = MINUTE


def _daily_reference(raw: list[dict], day: str) -> tuple[float | None, float | None]:
    matches = [
        row
        for row in raw
        if timestamp(row["t"]).tz_convert(ET).date().isoformat() == day
    ]
    if len(matches) != 1:
        return None, None
    row = matches[0]
    price = number(row["c"], "prior close")
    if not price:
        return None, None
    volume = number(row["v"], "prior volume")
    vwap = number(row["vw"], "prior VWAP") if row.get("vw") is not None else None
    return price, (vwap or price) * volume


def _snapshot_features(
    candles: list[Candle],
    cutoff: pd.Timestamp,
    starts: list[pd.Timestamp],
    start: pd.Timestamp,
    lag: pd.Timedelta,
    previous: float | None,
) -> dict:
    end = bisect_right(starts, cutoff - MINUTE - lag)
    known = candles[:end]
    last = known[-1].bar if known else None
    observed = last.available_at + lag if last else None
    age = cutoff - observed if observed is not None else None
    status = "no_bar" if age is None else "fresh" if age <= MAX_PEER_AGE else "stale"
    row = dict(
        bar_status=status,
        last_bar_start=last.start.isoformat() if last else None,
        last_bar_observed_at=observed.isoformat() if observed is not None else None,
        bar_age_seconds=age.total_seconds() if age is not None else None,
        last_price=last.close if last else None,
        return_since_previous_close_bps=(last.close / previous - 1) * 10_000
        if last and previous
        else None,
        elapsed_minutes=(cutoff - start).total_seconds() / 60,
        session_bars=end,
        session_share_volume=sum(c.bar.volume for c in known),
        session_dollar_volume=sum(c.notional for c in known),
    )
    for size in WINDOWS:
        # The trailing window uses simulated observation times.
        first = bisect_right(starts, cutoff - (size + 1) * MINUTE - lag, hi=end)
        window = known[first:]
        low = min((c.bar.low for c in window), default=None)
        high = max((c.bar.high for c in window), default=None)
        recent = window[-1].bar.close if window else None
        trades = [c.trades for c in window if c.trades is not None]
        volume = sum(c.bar.volume for c in window)
        notional = sum(c.notional for c in window)
        features = dict(
            bars=len(window),
            share_volume=volume,
            dollar_volume=notional,
            trade_count=sum(trades) if window and len(trades) == len(window) else None,
            high=high,
            low=low,
            close_vs_high_bps=(recent / high - 1) * 10_000 if high else None,
            close_vs_vwap_bps=(recent / (notional / volume) - 1) * 10_000
            if notional and volume
            else None,
            close_return_bps=(recent / window[0].bar.close - 1) * 10_000
            if len(window) >= 2
            else None,
        )
        row.update(
            {f"trailing_{size}m_{key}": value for key, value in features.items()}
        )
    return row


def _add_peers(rows: list[dict]) -> None:
    cohorts = defaultdict(list)
    for row in rows:
        change = row["return_since_previous_close_bps"]
        if row["peer_cohort"] and row["bar_status"] == "fresh" and change is not None:
            cohorts[row["peer_cohort"]].append(change)
    for values in cohorts.values():
        values.sort()
    for row in rows:
        values = cohorts.get(row["peer_cohort"], [])
        change = row["return_since_previous_close_bps"]
        own = (
            row["peer_cohort"] is not None
            and row["bar_status"] == "fresh"
            and change is not None
        )
        position = bisect_left(values, change) if own else len(values)
        count = len(values) - int(own)
        baseline = None
        if count >= 5:
            low, high = (count - 1) // 2, count // 2
            baseline = (
                values[low + (low >= position)] + values[high + (high >= position)]
            ) / 2
        row["peer_observations"] = count
        row["peer_median_return_bps"] = baseline
        row["excess_vs_peer_median_bps"] = (
            change - baseline if own and baseline is not None else None
        )


def _quote_features(
    raw: list[dict] | None,
    start: pd.Timestamp,
    decision: pd.Timestamp,
    size_multiplier: int,
) -> dict:
    quote = None
    if raw is not None:
        quotes = (Quote.read(item, size_multiplier) for item in raw)
        quote = max(
            (q for q in quotes if start <= q.time <= decision),
            key=lambda q: q.time,
            default=None,
        )
    two_sided = quote is not None and quote.two_sided
    status = (
        "unsampled"
        if raw is None
        else "no_quote"
        if quote is None
        else "two_sided"
        if two_sided
        else "invalid_book"
    )
    return dict(
        quote_status=status,
        quote_time=quote.time.isoformat() if quote else None,
        quote_age_seconds=(decision - quote.time).total_seconds() if quote else None,
        bid=quote.bid if quote else None,
        ask=quote.ask if quote else None,
        bid_size=quote.bid_size if quote else None,
        ask_size=quote.ask_size if quote else None,
        spread_bps=20_000 * (quote.ask - quote.bid) / (quote.ask + quote.bid)
        if two_sided
        else None,
    )


def collect_snapshots(
    client: Alpaca,
    directory: dict,
    day: date,
    *,
    feed: str,
    clocks: list[time],
    bar_lag_seconds: int = 35,
    quote_sample_size: int = 100,
) -> dict:
    if (
        feed not in {"sip", "iex"}
        or not 0 <= bar_lag_seconds <= 120
        or quote_sample_size < 0
    ):
        raise ValueError("Invalid snapshot feed, bar lag or quote sample size")
    if (
        not clocks
        or len(clocks) != len(set(clocks))
        or any(
            clock.tzinfo is not None or clock.second or clock.microsecond
            for clock in clocks
        )
    ):
        raise ValueError("Provide distinct whole-minute snapshot times in ET")
    day_key = day.isoformat()
    calendar = client.get(
        PAPER,
        "/v2/calendar",
        start=(day - timedelta(days=21)).isoformat(),
        end=day_key,
    )
    session = next((row for row in calendar if row["date"] == day_key), None)
    previous = max(
        (row["date"] for row in calendar if row["date"] < day_key), default=None
    )
    if session is None or previous is None:
        raise ValueError(f"Calendar has no complete exchange session for {day_key}")
    start = at(day_key, "04:00" if feed == "sip" else "08:00")
    opening, closing = (at(day_key, session[key]) for key in ("open", "close"))
    decisions = sorted(
        at(day_key, clock.isoformat(timespec="minutes")) for clock in clocks
    )
    if decisions[0] < start or decisions[-1] > closing:
        raise ValueError(
            "Snapshot times must be between the feed's premarket start and market close"
        )
    entitlement_delay = 16 * MINUTE if feed == "sip" else pd.Timedelta(0)
    if pd.Timestamp.now(tz="UTC") < decisions[-1] + entitlement_delay:
        raise ValueError(
            f"{feed.upper()} archive is unavailable through {decisions[-1].isoformat()}"
        )
    listings = directory_rows(directory, previous, start)
    names = sorted(row["symbol"] for row in listings if not row["listing_reasons"])
    daily = client.stocks(
        "bars",
        names,
        at(previous, "00:00"),
        at(previous, "23:59"),
        feed=feed,
        timeframe="1Day",
        adjustment="raw",
        asof="-",
        batch_size=200,
    )
    refs = {name: _daily_reference(daily.get(name, []), previous) for name in names}
    lag = bar_lag_seconds * SECOND
    last_start = decisions[-1] - MINUTE - lag
    bars = (
        client.stocks(
            "bars",
            names,
            start,
            last_start,
            feed=feed,
            timeframe="1Min",
            adjustment="raw",
            asof="-",
            batch_size=200,
        )
        if last_start >= start
        else {}
    )
    tapes = {}
    for name in names:
        tape = sorted(
            (Candle.read(row) for row in bars.get(name, [])), key=lambda c: c.bar.start
        )
        tape = [c for c in tape if start <= c.bar.start <= last_start]
        starts = [c.bar.start for c in tape]
        if len(starts) != len(set(starts)):
            raise ValueError(f"Duplicate minute bars for {name} on {day_key}")
        tapes[name] = tape, starts

    # Keep the same sample throughout the session. Sort by a hash of the
    # dated identifier so inclusion depends on neither prices nor activity.
    quoted = sorted(
        sorted(
            names,
            key=lambda name: hashlib.sha256(f"{day_key}/{name}".encode()).digest(),
        )[:quote_sample_size]
    )
    probability = len(quoted) / len(names)
    quoted_names = set(quoted)
    size_multiplier = 100 if day < date(2025, 11, 3) else 1
    output, coverage = [], []
    for decision in decisions:
        quote_start = max(start, decision - QUOTE_WINDOW)
        quotes = (
            client.stocks(
                "quotes", quoted, quote_start, decision, feed=feed, batch_size=100
            )
            if quoted
            else {}
        )
        rows = []
        for name in names:
            price, notional = refs[name]
            cohort = None
            if price is not None and notional is not None:
                cohort = f"{bisect_right(PRICE_EDGES, price)}:{bisect_right(NOTIONAL_EDGES, notional)}"
            tape, starts = tapes[name]
            row = dict(
                date=day_key,
                feed=feed,
                symbol=name,
                decision_at=decision.isoformat(),
                phase="premarket" if decision < opening else "regular",
                previous_close=price,
                previous_dollar_volume=notional,
                peer_cohort=cohort,
                quote_inclusion_probability=probability,
                **_snapshot_features(tape, decision, starts, start, lag, price),
                **_quote_features(
                    quotes.get(name, []) if name in quoted_names else None,
                    quote_start,
                    decision,
                    size_multiplier,
                ),
            )
            rows.append(row)
        _add_peers(rows)
        output.extend(rows)
        coverage.append(
            dict(
                decision_at=decision.isoformat(),
                sampled=len(rows),
                bar_status=dict(Counter(row["bar_status"] for row in rows)),
                quote_status=dict(Counter(row["quote_status"] for row in rows)),
                comparable_peers=sum(
                    row["excess_vs_peer_median_bps"] is not None for row in rows
                ),
            )
        )
    return dict(
        rows=output,
        listings=listings,
        manifest=dict(
            schema=1,
            date=day_key,
            feed=feed,
            session_start=start.isoformat(),
            market_open=opening.isoformat(),
            previous_session=previous,
            directory_source=directory["source"],
            directory_sha256=directory["sha256"],
            directory_available_at=directory["available_at"],
            listing_rows=len(listings),
            eligible_listings=len(names),
            snapshot_rows=len(output),
            coverage=coverage,
            observation="historical_completion_plus_lag",
            bar_lag_seconds=bar_lag_seconds,
            price_edges=PRICE_EDGES,
            prior_dollar_volume_edges=NOTIONAL_EDGES,
            peer_max_bar_age_seconds=MAX_PEER_AGE.total_seconds(),
            peer_min_observations=5,
            quote_lookback_seconds=QUOTE_WINDOW.total_seconds(),
            quote_sample_size=len(quoted),
            quote_selection="sha256(date/symbol)",
            quote_timing="vendor_event_time",
            prior_reference="raw_same_feed_daily_bar",
            feature_windows=WINDOWS,
        ),
    )
