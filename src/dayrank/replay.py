"""An event-time, long-only simulator for opening breakouts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from dayrank.data import MINUTE, SECOND, Quote, Session, SymbolData
from dayrank.scan import ScanConfig, scan

SETUPS = ("opening-range", "premarket-high")


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    capital: float = 2_000.0
    risk_fraction: float = 0.01
    max_exposure_fraction: float = 0.50
    stop_fraction: float = 0.04
    max_spread_to_stop: float = 0.35
    max_chase_fraction: float = 0.03
    cost_bps: float = 5.0
    latency_ms: int = 250
    entry_window_seconds: int = 15
    opening_range_minutes: int = 5
    last_entry_minutes: int = 30
    exit_minutes: int = 60
    top: int = 10
    round_trips_per_five: int | None = None

    def __post_init__(self) -> None:
        positive = (
            self.capital,
            self.risk_fraction,
            self.max_exposure_fraction,
            self.stop_fraction,
            self.max_spread_to_stop,
            self.entry_window_seconds,
            self.opening_range_minutes,
            self.last_entry_minutes,
            self.exit_minutes,
            self.top,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Capital, sizing, and timing settings must be positive")
        if (
            max(self.risk_fraction, self.max_exposure_fraction) > 1
            or self.stop_fraction >= 1
            or not math.isfinite(self.max_chase_fraction)
            or self.max_chase_fraction < 0
            or not 0 <= self.cost_bps < 10_000
            or self.latency_ms < 0
            or not self.opening_range_minutes
            < self.last_entry_minutes
            < self.exit_minutes
            or (self.round_trips_per_five is not None and self.round_trips_per_five < 1)
        ):
            raise ValueError("Invalid sizing, latency, holding period, or trade limit")

    @property
    def latency(self) -> pd.Timedelta:
        return pd.Timedelta(milliseconds=self.latency_ms)

    @property
    def cost(self) -> float:
        return self.cost_bps / 10_000


@dataclass(frozen=True, slots=True)
class Signal:
    time: pd.Timestamp
    level: float
    bar_close: float


def signal_for(
    session: Session,
    data: SymbolData,
    candidate: dict,
    setup: str,
    config: ReplayConfig,
) -> tuple[Signal | None, str]:
    opening = session.market_open
    range_end = opening + config.opening_range_minutes * MINUTE
    entry_end = opening + config.last_entry_minutes * MINUTE
    bars = [b for b in data.bars if opening <= b.start < entry_end]
    if setup == "opening-range":
        initial = [b for b in bars if b.start < range_end]
        if len(initial) != config.opening_range_minutes:
            return None, "incomplete_opening_range"
        level = max(b.high for b in initial)
        first = range_end
    else:
        level = candidate["premarket_high"]
        first = opening
    if level is None:
        return None, "no_trigger_level"
    for bar in bars:
        if bar.start >= first and bar.close > level and bar.available_at < entry_end:
            return Signal(bar.available_at, level, bar.close), ""
    return None, "no_breakout"


def _entry(
    data: SymbolData,
    signal: Signal,
    deadline: pd.Timestamp,
    config: ReplayConfig,
) -> tuple[Quote | None, str]:
    reason = "no_quote_after_order"
    arrival = signal.time + config.latency
    max_offer = signal.level * (1 + config.max_chase_fraction)
    for q in data.quotes:
        if q.time <= arrival:
            continue
        if q.time > deadline:
            break
        if not q.two_sided:
            reason = "one_sided_or_crossed_quote"
        elif q.ask < signal.level:
            reason = "breakout_retraced"
        elif q.ask > max_offer:
            reason = "offer_above_chase_limit"
        elif (q.ask - q.bid) / (
            q.ask * config.stop_fraction
        ) > config.max_spread_to_stop:
            reason = "spread_too_large_for_stop"
        else:
            return q, ""
    return None, reason


def _exit(
    data: SymbolData,
    bought: Quote,
    shares: int,
    time_exit: pd.Timestamp,
    config: ReplayConfig,
) -> tuple[str, int, float, pd.Timestamp | None]:
    stop = bought.ask * (1 - config.stop_fraction)
    exit_trigger = time_exit
    reason = "time_exit"
    quotes = [
        q
        for q in data.quotes
        if q.time > bought.time
        and q.bid > 0
        and q.bid_size > 0
        and (q.ask == 0 or q.ask >= q.bid)
    ]
    for q in quotes:
        if q.time >= time_exit:
            break
        if q.bid <= stop:
            reason = "stop"
            exit_trigger = q.time
            break
    remaining = shares
    proceeds = 0.0
    last_fill = None
    used_bid = None
    for q in quotes:
        if q.time <= exit_trigger + config.latency or q.bid_key == used_bid:
            continue
        if q.time > data.execution_until:
            break
        filled = min(remaining, q.bid_size)
        remaining -= filled
        proceeds += filled * q.bid * (1 - config.cost)
        last_fill = q.time
        used_bid = q.bid_key
        if remaining == 0:
            break
    return reason, remaining, proceeds, last_fill


def _iso(value: pd.Timestamp | None) -> str | None:
    return value.isoformat() if value is not None else None


def replay(
    sessions: list[Session],
    selection: str,
    setup: str,
    config: ReplayConfig,
    scan_config: ScanConfig | None = None,
) -> dict:
    if setup not in SETUPS or not sessions:
        raise ValueError("Use a supported setup and at least one session")
    scan_config = scan_config if scan_config is not None else ScanConfig()
    sessions = sorted(sessions, key=lambda s: s.date)
    if (
        len({s.date for s in sessions}) != len(sessions)
        or len({s.feed for s in sessions}) != 1
    ):
        raise ValueError("Replay requires unique trading dates from one feed")
    settled = config.capital
    unsettled: list[tuple[str, float]] = []
    entries: list[str] = []
    ledger = []
    daily = []
    closed_pnl = 0.0
    unresolved_shares = 0
    unknown_execution = 0
    incomplete_news = sum(
        selection == "catalyst-rvol"
        and session.provenance.get("news_status") != "available"
        for session in sessions
    )
    for session in sessions:
        settled += sum(amount for due, amount in unsettled if due <= session.date)
        unsettled = [(due, amount) for due, amount in unsettled if due > session.date]
        busy_until = session.market_open
        day_ledger = []
        signals = []
        candidate_rows = [
            row for row in scan(session, selection, scan_config) if row["eligible"]
        ]
        required_end = session.market_open + config.exit_minutes * MINUTE
        for candidate in candidate_rows[: config.top]:
            symbol = candidate["symbol"]
            data = session.symbols[symbol]
            item = {"date": session.date, "symbol": symbol, "rank": candidate["rank"]}
            day_ledger.append(item)
            if data.execution_until is None or data.execution_until <= required_end:
                item["status"] = "insufficient_execution_capture"
                unknown_execution += 1
                continue
            signal, reason = signal_for(session, data, candidate, setup, config)
            if signal is None:
                item["status"] = reason
                continue
            item.update(
                signal_at=_iso(signal.time),
                trigger=signal.level,
                signal_close=signal.bar_close,
            )
            signals.append((signal.time, candidate["rank"], signal, data, item))

        recent = {*session.previous_sessions[-4:], session.date}
        for _, _, signal, data, item in sorted(signals, key=lambda s: s[:2]):
            if unresolved_shares:
                item["status"] = "blocked_by_unresolved_position"
                continue
            if signal.time < busy_until:
                item["status"] = "other_order_or_position_active"
                continue
            if config.round_trips_per_five is not None and (
                sum(day in recent for day in entries) >= config.round_trips_per_five
            ):
                item["status"] = "five_session_trade_limit"
                continue
            deadline = min(
                signal.time + config.entry_window_seconds * SECOND,
                session.market_open + config.last_entry_minutes * MINUTE,
            )
            item["order_at"] = _iso(signal.time + config.latency)
            q, reason = _entry(data, signal, deadline, config)
            if q is None:
                item["status"] = reason
                busy_until = deadline
                continue
            equity = settled + sum(amount for _, amount in unsettled)
            cost_per_share = q.ask * (1 + config.cost)
            risk_per_share = q.ask * config.stop_fraction + 2 * config.cost * q.ask
            wanted = math.floor(
                min(
                    settled / cost_per_share,
                    equity * config.max_exposure_fraction / cost_per_share,
                    equity * config.risk_fraction / risk_per_share,
                )
            )
            if wanted <= 0:
                item["status"] = "insufficient_settled_cash_or_risk_budget"
                busy_until = q.time
                continue
            shares = min(wanted, q.ask_size)
            debit = shares * cost_per_share
            settled -= debit
            reason, remaining, proceeds, last_fill = _exit(
                data, q, shares, required_end, config
            )
            entries.append(session.date)
            if proceeds:
                unsettled.append((session.settlement_date, proceeds))
            unresolved_shares += remaining
            if not remaining:
                closed_pnl += proceeds - debit
            busy_until = last_fill or data.execution_until
            item.update(
                status="open_unresolved" if remaining else "closed",
                entry_at=_iso(q.time),
                entry_bid=q.bid,
                entry_ask=q.ask,
                entry_spread=q.ask - q.bid,
                spread_to_stop=(q.ask - q.bid) / (q.ask * config.stop_fraction),
                wanted_shares=wanted,
                shares=shares,
                remaining_shares=remaining,
                exit_at=_iso(last_fill),
                exit_reason=reason,
                cash_spent=debit,
                sale_proceeds=proceeds,
                pnl=None if remaining else proceeds - debit,
            )
        ledger.extend(day_ledger)
        daily.append(
            dict(
                date=session.date,
                settled_cash=settled,
                unsettled_cash=sum(amount for _, amount in unsettled),
                unresolved_shares=unresolved_shares,
                cash_and_receivables=settled + sum(amount for _, amount in unsettled),
            )
        )
    marks = [config.capital, *(row["cash_and_receivables"] for row in daily)]
    peak = marks[0]
    drawdown = 0.0
    for value in marks:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1)
    return dict(
        summary=dict(
            sessions=len(sessions),
            starting_capital=config.capital,
            closed_trades=sum(item["status"] == "closed" for item in ledger),
            entries=len(entries),
            closed_trade_pnl=closed_pnl,
            unresolved_shares=unresolved_shares,
            insufficient_captures=unknown_execution,
            incomplete_news_sessions=incomplete_news,
            ending_cash_and_receivables=marks[-1],
            ending_equity=marks[-1] if not unresolved_shares else None,
            return_fraction=marks[-1] / config.capital - 1
            if not (unresolved_shares or unknown_execution or incomplete_news)
            else None,
            max_session_end_drawdown=drawdown
            if not (unresolved_shares or unknown_execution or incomplete_news)
            else None,
        ),
        daily=daily,
        ledger=ledger,
    )
