"""Build a deterministic case census from exchange-wide discovery and minute bars."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from statistics import median

import pandas as pd

from dayrank.alpaca import PAPER, Alpaca
from dayrank.data import MINUTE, SECOND, Candle, at, timestamp

HORIZONS = (1, 3, 5)
FEATURE_WINDOWS = (5, 15, 30)


@dataclass(frozen=True, slots=True)
class IndicatorRules:
    rolling_minutes: int = 15
    minimum_prior_bars: int = 5
    opening_range_minutes: int = 5
    opening_scan_minutes: int = 30
    near_miss_bps: float = 25
    break_buffer_bps: float = 15
    minimum_bar_volume_multiple: float = 2
    minimum_signal_trades: int = 20


def collect_tape(
    client: Alpaca,
    manifest: dict,
    symbols: list[dict],
    premarket_only: bool,
    rules: IndicatorRules,
) -> dict:
    day, feed = manifest["date"], manifest["feed"]
    lag_seconds = manifest["bar_lag_seconds"]
    if (
        manifest["observation"] != "historical_completion_plus_lag"
        or not 0 <= lag_seconds <= 120
    ):
        raise ValueError(
            "Indicator capture requires a discovery run using historical bar timing"
        )
    if feed not in {"sip", "iex"} or len(symbols) != manifest["listing_rows"]:
        raise ValueError("Discovery symbols and manifest do not match")
    start, cutoff = (timestamp(manifest[k]) for k in ("window_start", "cutoff"))
    calendar = client.get(PAPER, "/v2/calendar", start=day, end=day)
    session = next((row for row in calendar if row["date"] == day), None)
    if session is None:
        raise ValueError(f"No exchange session for {day}")
    opening = at(day, session["open"])
    if not start < cutoff < opening:
        raise ValueError("Unexpected premarket discovery window")
    through = (
        cutoff
        if premarket_only
        else opening
        + (rules.opening_scan_minutes + max(HORIZONS)) * MINUTE
        + lag_seconds * SECOND
    )
    entitlement_delay = 16 * MINUTE if feed == "sip" else pd.Timedelta(0)
    if pd.Timestamp.now(tz="UTC") < through + entitlement_delay:
        raise ValueError(
            f"{feed.upper()} bars are unavailable through {through.isoformat()}"
        )

    # The exchange audit already accounts for names without a previous close
    # or a premarket print. Intraday levels require that premarket context.
    names = {
        row["symbol"]: float(row["previous_close"])
        for row in symbols
        if not row["listing_reasons"]
        and row["previous_close"]
        and row["last_bar_at"]
        and row["cutoff_exclusion"] != "split_effective"
    }
    bars = client.stocks(
        "bars",
        sorted(names),
        start,
        through - MINUTE - lag_seconds * SECOND,
        feed=feed,
        timeframe="1Min",
        adjustment="raw",
        asof="-",
        batch_size=200,
    )
    return {
        "schema": 1,
        "date": day,
        "feed": feed,
        "premarket_only": premarket_only,
        "start": start.isoformat(),
        "cutoff": cutoff.isoformat(),
        "open": opening.isoformat(),
        "through": through.isoformat(),
        "bar_lag_seconds": lag_seconds,
        "listing_rows": manifest["listing_rows"],
        "eligible_listings": manifest["eligible_listings"],
        "scan_config": manifest["scan_config"],
        "directory_sha256": manifest["directory_sha256"],
        "split_status": manifest["split_status"],
        "previous_closes": names,
        "bars": bars,
    }


def _measurements(candle: Candle, prior: deque[Candle]) -> dict:
    bar = candle.bar
    span = bar.high - bar.low
    features = dict(
        bar_open=candle.open,
        bar_vwap=candle.vwap,
        bar_range_bps=10_000 * span / candle.open,
        bar_body_bps=10_000 * (bar.close / candle.open - 1),
        upper_wick_bps=10_000 * (bar.high - max(bar.close, candle.open)) / candle.open,
        lower_wick_bps=10_000 * (min(bar.close, candle.open) - bar.low) / candle.open,
        close_in_range=(bar.close - bar.low) / span if span else None,
        average_shares_per_trade=bar.volume / candle.trades if candle.trades else None,
        close_to_bar_vwap_bps=10_000 * (bar.close / candle.vwap - 1)
        if candle.vwap
        else None,
    )
    for minutes in FEATURE_WINDOWS:
        window = [c for c in prior if bar.start - c.bar.start <= minutes * MINUTE]
        volume = sum(c.bar.volume for c in window)
        notional = sum(c.notional for c in window)
        trades = [c.trades for c in window if c.trades is not None]
        usable_volume = [c.bar.volume for c in window if c.bar.volume]
        median_volume = median(usable_volume) if usable_volume else None
        median_trades = median(trades) if trades else None
        high = max((c.bar.high for c in window), default=None)
        low = min((c.bar.low for c in window), default=None)
        vwap = notional / volume if volume else None
        metrics = dict(
            observed_bars=len(window),
            share_volume=volume,
            dollar_volume=notional,
            trade_count=sum(trades) if len(trades) == len(window) and trades else None,
            median_bar_volume=median_volume,
            median_bar_trades=median_trades,
            high=high,
            low=low,
            range_bps=10_000 * (high / low - 1) if low else None,
            estimated_vwap=vwap,
            close_vs_first_bps=10_000 * (bar.close / window[0].bar.close - 1)
            if window
            else None,
            close_vs_vwap_bps=10_000 * (bar.close / vwap - 1) if vwap else None,
            close_vs_high_bps=10_000 * (bar.close / high - 1) if high else None,
            signal_volume_multiple=bar.volume / median_volume
            if median_volume
            else None,
            signal_trade_multiple=candle.trades / median_trades
            if candle.trades is not None and median_trades
            else None,
        )
        features.update(
            {f"prior_{minutes}m_{key}": value for key, value in metrics.items()}
        )
    return features


def _event_rows(
    symbol: str,
    candles: list[Candle],
    levels: list[float | None],
    pattern: str,
    day: str,
    feed: str,
    lag: pd.Timedelta,
    rules: IndicatorRules,
) -> list[dict]:
    rows = []
    seen: set[str] = set()
    pending = None
    recent: deque[Candle] = deque()
    for candle, level in zip(candles, levels, strict=True):
        bar = candle.bar
        while (
            recent and bar.start - recent[0].bar.start > max(FEATURE_WINDOWS) * MINUTE
        ):
            recent.popleft()
        recent_setup = [
            c.bar
            for c in recent
            if bar.start - c.bar.start <= rules.rolling_minutes * MINUTE
        ]
        previous = recent_setup[-1] if recent_setup else None
        volumes = [item.volume for item in recent_setup if item.volume > 0]
        baseline = median(volumes) if len(volumes) >= rules.minimum_prior_bars else None
        multiple = bar.volume / baseline if baseline else None
        cases = []
        if pending is not None:
            broken_level, broken_at = pending
            if bar.start == broken_at + MINUTE and bar.close > broken_level:
                cases.append(("hold_confirmed", broken_level))
            pending = None
        if level is not None:
            if (
                "close_break" not in seen
                and bar.close > level
                and (previous is None or previous.close <= level)
            ):
                cases.append(("close_break", level))
                pending = level, bar.start
            elif bar.high > level >= bar.close and bar.low <= level:
                cases.append(("wick_rejection", level))
            elif (
                bar.high <= level
                and 0 <= (1 - bar.close / level) * 10_000 <= rules.near_miss_bps
            ):
                cases.append(("near_miss", level))
        for case, trigger in cases:
            if case in seen:
                continue
            seen.add(case)
            margin = (bar.close / trigger - 1) * 10_000
            selected = (
                case == "close_break"
                and margin >= rules.break_buffer_bps
                and multiple is not None
                and multiple >= rules.minimum_bar_volume_multiple
                and candle.trades is not None
                and candle.trades >= rules.minimum_signal_trades
            )
            rows.append(
                dict(
                    case_id=f"{day}/{feed}/{symbol}/{pattern}/{case}",
                    date=day,
                    feed=feed,
                    symbol=symbol,
                    pattern=pattern,
                    case=case,
                    observation=(bar.available_at + lag).isoformat(),
                    bar_start=bar.start.isoformat(),
                    level=trigger,
                    close=bar.close,
                    high=bar.high,
                    low=bar.low,
                    close_margin_bps=round(margin, 4),
                    bar_trade_count=candle.trades,
                    bar_volume=bar.volume,
                    bar_dollar_volume=round(candle.notional, 2),
                    median_prior_bar_volume=baseline,
                    recent_bar_volume_multiple=multiple,
                    traded_through_level=bar.low <= trigger <= bar.high,
                    volume_confirmed=selected,
                    **_measurements(candle, recent),
                )
            )
        recent.append(candle)
    return rows


def _rolling_levels(
    candles: list[Candle], tape: dict, rules: IndicatorRules
) -> list[float | None]:
    recent: deque[Candle] = deque()
    cumulative = 0.0
    config = tape["scan_config"]
    levels = []
    for candle in candles:
        bar = candle.bar
        while (
            recent and bar.start - recent[0].bar.start > rules.rolling_minutes * MINUTE
        ):
            recent.popleft()
        cumulative += candle.notional
        in_scope = config["minimum_price"] <= bar.close <= config["maximum_price"]
        level = (
            max(item.bar.high for item in recent)
            if in_scope
            and cumulative >= config["minimum_dollar_volume"]
            and len(recent) >= rules.minimum_prior_bars
            else None
        )
        levels.append(level)
        recent.append(candle)
    return levels


def evaluate_tape(
    tape: dict, rules: IndicatorRules
) -> tuple[list[dict], list[dict], dict]:
    if tape.get("schema") != 1:
        raise ValueError("Unsupported indicator tape schema")
    start, cutoff, opening, through = (
        timestamp(tape[k]) for k in ("start", "cutoff", "open", "through")
    )
    day, feed = tape["date"], tape["feed"]
    lag = tape["bar_lag_seconds"] * SECOND
    range_end = opening + rules.opening_range_minutes * MINUTE
    entry_end = opening + rules.opening_scan_minutes * MINUTE
    events, outcomes = [], []
    seen_symbols = 0
    config = tape["scan_config"]
    for symbol, raw in sorted(tape["bars"].items()):
        candles = sorted(
            (Candle.read(row) for row in raw), key=lambda item: item.bar.start
        )
        by_minute = {c.bar.start: c for c in candles}
        if len(by_minute) != len(candles):
            raise ValueError(f"Duplicate minutes in {day} {symbol} tape")
        premarket = [
            c
            for c in candles
            if start <= c.bar.start and c.bar.available_at + lag <= cutoff
        ]
        if not premarket:
            continue
        seen_symbols += 1
        cases = _event_rows(
            symbol,
            premarket,
            _rolling_levels(premarket, tape, rules),
            "premarket_rolling_high",
            day,
            feed,
            lag,
            rules,
        )
        if not tape["premarket_only"] and (
            sum(c.notional for c in premarket) >= config["minimum_dollar_volume"]
            and config["minimum_price"]
            <= premarket[-1].bar.close
            <= config["maximum_price"]
        ):
            post = [
                c
                for c in candles
                if opening <= c.bar.start and c.bar.available_at + lag <= entry_end
            ]
            initial = [c for c in post if c.bar.start < range_end]
            levels = [max(c.bar.high for c in premarket)] * len(post)
            cases.extend(
                _event_rows(
                    symbol,
                    post,
                    levels,
                    "premarket_high_at_open",
                    day,
                    feed,
                    lag,
                    rules,
                )
            )
            if len(initial) == rules.opening_range_minutes:
                level = max(c.bar.high for c in initial)
                levels = [level if c.bar.start >= range_end else None for c in post]
                cases.extend(
                    _event_rows(
                        symbol,
                        post,
                        levels,
                        "opening_range_high",
                        day,
                        feed,
                        lag,
                        rules,
                    )
                )
        for row in cases:
            event_start = timestamp(row["bar_start"])
            row.update(
                previous_close=tape["previous_closes"][symbol],
                cutoff=cutoff.isoformat(),
            )
            events.append(row)
            forward = {"case_id": row["case_id"]}
            for horizon in HORIZONS:
                future = by_minute.get(event_start + horizon * MINUTE)
                limit = (
                    cutoff if row["pattern"] == "premarket_rolling_high" else through
                )
                known = future is not None and future.bar.available_at + lag <= limit
                forward[f"close_return_{horizon}m_bps"] = (
                    round((future.bar.close / row["close"] - 1) * 10_000, 4)
                    if known
                    else None
                )
            outcomes.append(forward)
    return (
        events,
        outcomes,
        {
            "date": day,
            "feed": feed,
            "listing_rows": tape["listing_rows"],
            "eligible_listings": tape["eligible_listings"],
            "requested_symbols": len(tape["previous_closes"]),
            "symbols_with_premarket": seen_symbols,
            "case_count": len(events),
            "split_status": tape["split_status"],
            "directory_sha256": tape["directory_sha256"],
            "premarket_only": tape["premarket_only"],
            "bar_lag_seconds": tape["bar_lag_seconds"],
        },
    )


def summarize(results: list[tuple], rules: IndicatorRules, review_size: int) -> dict:
    if review_size < 1:
        raise ValueError("Review size must be positive")
    cases, outcomes, sessions = [], [], []
    for rows, forward, coverage in results:
        cases.extend(rows)
        outcomes.extend(forward)
        sessions.append(coverage)
    if len({(s["date"], s["feed"]) for s in sessions}) != len(sessions):
        raise ValueError("Supply each date and feed once")
    cases.sort(key=lambda row: (row["observation"], row["case_id"]))
    sessions.sort(key=lambda row: (row["date"], row["feed"]))
    outcomes.sort(key=lambda row: row["case_id"])
    counts = Counter((row["pattern"], row["case"]) for row in cases)
    by_id = {row["case_id"]: row for row in outcomes}
    followthrough = defaultdict(list)
    for row in cases:
        value = by_id[row["case_id"]]["close_return_3m_bps"]
        if value is None:
            continue
        pattern, case = row["pattern"], row["case"]
        followthrough[f"{pattern}/{case}"].append(value)
        if row["volume_confirmed"]:
            followthrough[f"{pattern}/volume_confirmed"].append(value)
    groups = defaultdict(list)
    for row in cases:
        key = row["pattern"], row["case"], row["date"], row["feed"]
        groups[key].append(row)
    buckets = [
        deque(
            sorted(
                group, key=lambda row: hashlib.sha256(row["case_id"].encode()).digest()
            )
        )
        for _, group in sorted(
            groups.items(),
            key=lambda item: hashlib.sha256("/".join(item[0]).encode()).digest(),
        )
    ]
    active = deque(buckets)
    sample = []
    while active and len(sample) < review_size:
        bucket = active.popleft()
        sample.append({**bucket.popleft(), "review_label": "", "review_note": ""})
        if bucket:
            active.append(bucket)
    return dict(
        cases=cases,
        outcomes=outcomes,
        review=sample,
        summary=dict(
            schema=1,
            rules=asdict(rules),
            sessions=sessions,
            case_count=len(cases),
            review_count=len(sample),
            volume_confirmed_count=sum(r["volume_confirmed"] for r in cases),
            patterns={f"{p}/{c}": count for (p, c), count in sorted(counts.items())},
            three_minute_followthrough={
                name: dict(
                    observations=len(values),
                    median_bps=median(values),
                    positive_fraction=sum(x > 0 for x in values) / len(values),
                )
                for name, values in sorted(followthrough.items())
            },
        ),
    )
