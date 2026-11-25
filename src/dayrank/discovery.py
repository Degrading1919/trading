"""Find premarket opportunities from a dated security directory."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import date, time, timedelta
from statistics import median

import pandas as pd

from dayrank.alpaca import PAPER, Alpaca, _news, _splits
from dayrank.data import ET, MINUTE, SECOND, Bar, at, timestamp
from dayrank.scan import SELECTIONS, ScanConfig
from dayrank.scan import _news as classify_news
from dayrank.universe import directory_rows


@dataclass(frozen=True, slots=True)
class Window:
    date: date
    start: pd.Timestamp
    cutoff: pd.Timestamp
    previous_days: tuple[str, ...]
    clock: str


@dataclass(frozen=True, slots=True)
class Point:
    observed: pd.Timestamp
    bar: Bar
    volume: float
    dollar_volume: float
    gap: float


@dataclass(frozen=True, slots=True)
class VolumeProfile:
    minutes: tuple[int, ...]
    totals: tuple[float, ...]

    def at(self, minute: int) -> float:
        index = bisect_right(self.minutes, minute)
        return self.totals[index - 1] if index else 0.0


def premarket_window(
    client: Alpaca,
    day: date,
    feed: str,
    lookback: int,
    scan_before: int = 5,
    *,
    start_at: time | None = None,
) -> Window:
    if (
        feed not in {"sip", "iex"}
        or not 10 <= lookback <= 60
        or not 1 <= scan_before <= 60
    ):
        raise ValueError("Invalid premarket feed, lookback or cutoff")
    calendar = client.get(
        PAPER,
        "/v2/calendar",
        start=(day - timedelta(days=lookback * 2 + 14)).isoformat(),
        end=day.isoformat(),
    )
    current = next((row for row in calendar if row["date"] == day.isoformat()), None)
    if not current:
        raise ValueError(f"{day} is not an exchange session")
    prior = sorted(row["date"] for row in calendar if row["date"] < day.isoformat())[
        -lookback:
    ]
    if len(prior) != lookback:
        raise ValueError("Exchange calendar does not contain enough preceding sessions")
    earliest = "04:00" if feed == "sip" else "08:00"
    if start_at is not None and (
        start_at.tzinfo is not None
        or start_at < time.fromisoformat(earliest)
        or start_at.second
        or start_at.microsecond
    ):
        raise ValueError(
            f"Start must use a full minute at or after {earliest} ET for {feed.upper()}"
        )
    clock = earliest if start_at is None else start_at.isoformat(timespec="minutes")
    start = at(day.isoformat(), clock)
    cutoff = at(day.isoformat(), current["open"]) - scan_before * MINUTE
    if start >= cutoff:
        raise ValueError("The scan cutoff precedes the feed's premarket session")
    return Window(day, start, cutoff, tuple(prior), clock)


def _points(
    events: list[tuple[pd.Timestamp, Bar]],
    previous: float,
    window: Window,
) -> list[Point]:
    """Apply revisions at receipt time, replacing the affected minute."""
    bars: dict[pd.Timestamp, Bar] = {}
    latest = None
    volume = 0.0
    dollars = 0.0
    points = []
    for observed, bar in sorted(events, key=lambda record: record[0]):
        if observed > window.cutoff or not window.start <= bar.start < window.cutoff:
            continue
        if observed < bar.available_at:
            raise ValueError("A recorded bar arrived before its minute completed")
        old = bars.get(bar.start)
        if old is not None:
            volume -= old.volume
            dollars -= old.volume * old.close
        bars[bar.start] = bar
        volume += bar.volume
        dollars += bar.volume * bar.close
        if latest is None or bar.start >= latest.start:
            latest = bar
        points.append(
            Point(observed, latest, volume, dollars, latest.close / previous - 1)
        )
    return points


def _gate(point: Point, now: pd.Timestamp, config: ScanConfig) -> str:
    if now - point.bar.available_at > 120 * SECOND:
        return "stale_premarket_price"
    if not config.minimum_price <= point.bar.close <= config.maximum_price:
        return "price"
    if point.gap < config.minimum_gap:
        return "gap"
    if point.dollar_volume < config.minimum_dollar_volume:
        return "premarket_dollar_volume"
    return ""


def _profile(
    rows: list[dict],
    start: pd.Timestamp,
    length: int,
) -> VolumeProfile:
    minutes = []
    totals = []
    running = 0.0
    for row in sorted(rows, key=lambda x: x["t"]):
        minute = int((timestamp(row["t"]) - start) // MINUTE)
        if 0 <= minute < length:
            running += row["v"]
            minutes.append(minute)
            totals.append(running)
    return VolumeProfile(tuple(minutes), tuple(totals))


def _history(
    client: Alpaca,
    names: list[str],
    window: Window,
    feed: str,
) -> dict[str, list[VolumeProfile]]:
    if not names:
        return {}
    days = window.previous_days
    daily = client.stocks(
        "bars",
        names,
        at(days[0], "00:00"),
        at(days[-1], "23:59"),
        feed=feed,
        timeframe="1Day",
        adjustment="raw",
        asof="-",
        batch_size=200,
    )
    observed_days = {
        name: {timestamp(bar["t"]).tz_convert(ET).date().isoformat() for bar in rows}
        for name, rows in daily.items()
    }
    history: dict[str, list[VolumeProfile]] = {name: [] for name in names}
    length = int((window.cutoff - window.start) // MINUTE)
    for day in days:
        start = at(day, window.clock)
        rows = client.stocks(
            "bars",
            names,
            start,
            start + (length - 1) * MINUTE,
            feed=feed,
            timeframe="1Min",
            adjustment="raw",
            asof="-",
            batch_size=200,
        )
        for name, bars in rows.items():
            # A daily observation confirms the symbol existed in the feed that day.
            # An empty but successful premarket response is then a zero-volume interval.
            if bars or day in observed_days[name]:
                history[name].append(_profile(bars, start, length))
    return history


def _features(
    point: Point,
    history: list[VolumeProfile],
    headlines: list[dict],
    window: Window,
    now: pd.Timestamp,
    config: ScanConfig,
    news_status: str,
) -> tuple[float | None, str, str]:
    minute = int((point.bar.start - window.start) // MINUTE)
    usual = (
        median(p.at(minute) for p in history)
        if len(history) >= config.minimum_history
        else None
    )
    relative = point.volume / usual if usual else None
    kind, headline = classify_news(tuple(headlines), now, config.news_hours)
    if kind == "none_observed" and news_status != "available":
        kind = "feed_unavailable"
    return relative, kind, headline


def discover(
    client: Alpaca,
    directory: dict,
    day: date,
    *,
    feed: str = "sip",
    lookback: int = 20,
    scan_before: int = 5,
    bar_lag_seconds: int = 35,
    start_at: time | None = None,
    recorded: tuple[dict, dict[str, list[tuple[pd.Timestamp, Bar]]]] | None = None,
) -> dict:
    config = ScanConfig()
    if not 0 <= bar_lag_seconds <= 120:
        raise ValueError("Bar availability lag must be between 0 and 120 seconds")
    window = premarket_window(
        client, day, feed, lookback, scan_before, start_at=start_at
    )
    listings = directory_rows(directory, window.previous_days[-1], window.start)
    if recorded is None:
        delay = 16 * MINUTE if feed == "sip" else pd.Timedelta(0)
        if pd.Timestamp.now(tz="UTC") < window.cutoff + delay:
            raise ValueError("Requested archive data is not yet available")
    else:
        header, _ = recorded
        if (
            header["date"] != day.isoformat()
            or header["feed"] != feed
            or timestamp(header["started_at"]) > window.start
            or timestamp(header["finished_at"]) < window.cutoff
            or header["directory_sha256"] != directory["sha256"]
        ):
            raise ValueError(
                "Receipt tape must cover the full window and match the directory"
            )
    names = sorted(row["symbol"] for row in listings if not row["listing_reasons"])
    prior = window.previous_days[-1]
    raw_close = client.stocks(
        "bars",
        names,
        at(prior, "00:00"),
        at(prior, "23:59"),
        feed=feed,
        timeframe="1Day",
        adjustment="raw",
        asof="-",
        batch_size=200,
    )
    close = {}
    for name, bars in raw_close.items():
        matches = [
            bar["c"]
            for bar in bars
            if timestamp(bar["t"]).tz_convert(ET).date().isoformat() == prior
        ]
        if len(matches) == 1 and matches[0] > 0:
            close[name] = matches[0]
    bound = config.maximum_price / (1 + config.minimum_gap)
    price_possible = sorted(name for name, price in close.items() if price <= bound)
    if recorded is None:
        minutes = client.stocks(
            "bars",
            price_possible,
            window.start,
            window.cutoff - MINUTE,
            feed=feed,
            timeframe="1Min",
            adjustment="raw",
            asof="-",
            batch_size=200,
        )
        events = {}
        for name, bars in minutes.items():
            parsed = [Bar.read(bar) for bar in bars]
            events[name] = [
                (bar.available_at + bar_lag_seconds * SECOND, bar) for bar in parsed
            ]
    else:
        _, events = recorded
    paths = {
        name: _points(events.get(name, []), close[name], window)
        for name in price_possible
    }
    preliminary = sorted(
        name
        for name, points in paths.items()
        if any(not _gate(point, point.observed, config) for point in points)
    )
    split, split_status = (
        _splits(client, preliminary, day) if preliminary else (set(), "not_needed")
    )
    candidates = [name for name in preliminary if name not in split]
    history = _history(client, candidates, window, feed)
    news, news_status = (
        _news(
            client,
            candidates,
            window.start - config.news_hours * 60 * MINUTE,
            window.cutoff,
        )
        if candidates
        else ({}, "not_needed")
    )

    audit = []
    alerts = []
    for row in listings:
        name = row["symbol"]
        reason = row["listing_reasons"]
        points = paths.get(name, [])
        final = points[-1] if points else None
        profiles = history.get(name, [])
        headlines = news.get(name, [])
        first: dict[str, str] = {}
        final_modes: list[str] = []
        relative = None
        kind = ""
        headline = ""
        if not reason:
            if name not in close:
                reason = "no_previous_close"
            elif name not in paths:
                reason = "prior_close_above_possible_gap_price"
            elif final is None:
                reason = "no_completed_premarket_bars"
            elif name in split:
                reason = "split_effective"
            else:
                reason = _gate(final, window.cutoff, config)
                for point in points:
                    if _gate(point, point.observed, config):
                        continue
                    rv, catalyst, headline_ = _features(
                        point,
                        profiles,
                        headlines,
                        window,
                        point.observed,
                        config,
                        news_status,
                    )
                    modes = _eligible_modes(rv, catalyst, config)
                    for mode in modes:
                        if mode not in first:
                            first[mode] = point.observed.isoformat()
                            alerts.append(
                                dict(
                                    symbol=name,
                                    rule=mode,
                                    observed_at=first[mode],
                                    bar_start=point.bar.start.isoformat(),
                                    price=point.bar.close,
                                    gap=point.gap,
                                    relative_volume=rv,
                                    catalyst=catalyst,
                                    headline=headline_,
                                )
                            )
                if not reason:
                    relative, kind, headline = _features(
                        final,
                        profiles,
                        headlines,
                        window,
                        window.cutoff,
                        config,
                        news_status,
                    )
                    final_modes = _eligible_modes(relative, kind, config)
        audit.append(
            dict(
                **row,
                previous_close=close.get(name),
                cutoff_price=final.bar.close if final else None,
                last_bar_at=final.bar.start.isoformat() if final else None,
                cutoff_gap=final.gap if final else None,
                cutoff_dollar_volume=final.dollar_volume if final else None,
                history_sessions=len(profiles),
                cutoff_rvol=relative,
                rvol_status=(
                    "not_applicable"
                    if reason
                    else "unknown"
                    if relative is None
                    else "below_threshold"
                    if relative < config.minimum_rvol
                    else "qualified"
                ),
                catalyst=kind,
                headline=headline,
                cutoff_exclusion=reason,
                first_gap_at=first.get("gap-only"),
                first_rvol_at=first.get("rvol"),
                first_catalyst_at=first.get("catalyst-rvol"),
                cutoff_rules="|".join(final_modes),
            )
        )
    alerts.sort(key=lambda row: (row["observed_at"], row["symbol"], row["rule"]))
    return dict(
        audit=audit,
        alerts=alerts,
        manifest=dict(
            date=day.isoformat(),
            feed=feed,
            window_start=window.start.isoformat(),
            cutoff=window.cutoff.isoformat(),
            lookback_dates=list(window.previous_days),
            directory_sha256=directory["sha256"],
            directory_source=directory["source"],
            directory_available_at=directory["available_at"],
            scan_config=asdict(config),
            observation="recorded_receipt"
            if recorded
            else "historical_completion_plus_lag",
            bar_lag_seconds=None if recorded else bar_lag_seconds,
            listing_rows=len(listings),
            eligible_listings=len(names),
            with_previous_close=len(close),
            price_possible=len(price_possible),
            with_premarket_bars=sum(bool(points) for points in paths.values()),
            preliminary_gap_candidates=len(preliminary),
            news_status=news_status,
            split_status=split_status,
            first_alerts={
                mode: sum(r["rule"] == mode for r in alerts) for mode in SELECTIONS
            },
            cutoff_candidates={
                mode: sum(mode in row["cutoff_rules"].split("|") for row in audit)
                for mode in SELECTIONS
            },
        ),
    )


def _eligible_modes(
    relative: float | None, catalyst: str, config: ScanConfig
) -> list[str]:
    modes = ["gap-only"]
    if relative is not None and relative >= config.minimum_rvol:
        modes.append("rvol")
        if catalyst == "material_keyword":
            modes.append("catalyst-rvol")
    return modes
