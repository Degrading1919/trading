"""The dataset boundary for capture, screening, and replay."""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path

import pandas as pd

ET = "America/New_York"
MINUTE = pd.Timedelta(minutes=1)
SECOND = pd.Timedelta(seconds=1)


def timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts) or ts.tzinfo is None:
        raise ValueError(f"Timestamp requires a UTC offset: {value!r}")
    return ts.tz_convert("UTC")


def at(day: str, clock: str) -> pd.Timestamp:
    return (
        pd.Timestamp.combine(date.fromisoformat(day), time.fromisoformat(clock))
        .tz_localize(ET)
        .tz_convert("UTC")
    )


def number(value: float, field: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return value


@dataclass(frozen=True, slots=True)
class Bar:
    start: pd.Timestamp
    high: float
    low: float
    close: float
    volume: float

    @property
    def available_at(self) -> pd.Timestamp:
        return self.start + MINUTE

    @classmethod
    def read(cls, row: dict) -> Bar:
        bar = cls(
            timestamp(row["t"]), *(number(row[k], k) for k in ("h", "l", "c", "v"))
        )
        if not 0 < bar.low <= bar.close <= bar.high or bar.start != bar.start.floor(
            "min"
        ):
            raise ValueError(f"Invalid minute bar at {row['t']}")
        return bar


@dataclass(frozen=True, slots=True)
class Candle:
    bar: Bar
    open: float
    trades: float | None
    vwap: float | None
    notional: float

    @classmethod
    def read(cls, row: dict) -> Candle:
        bar = Bar.read(row)
        opening = number(row["o"], "open")
        if not bar.low <= opening <= bar.high:
            raise ValueError(f"Invalid opening price at {bar.start}")
        trades = number(row["n"], "trade count") if row.get("n") is not None else None
        vwap = number(row["vw"], "VWAP") if row.get("vw") is not None else None
        return cls(bar, opening, trades, vwap or None, (vwap or bar.close) * bar.volume)


@dataclass(frozen=True, slots=True)
class Quote:
    time: pd.Timestamp
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    bid_exchange: str
    ask_exchange: str

    @classmethod
    def read(cls, row: dict, size_multiplier: int) -> Quote:
        sizes = [number(row[k], k) for k in ("bs", "as")]
        if any(not size.is_integer() for size in sizes):
            raise ValueError("Quoted size must be a whole number")
        return cls(
            timestamp(row["t"]),
            number(row["bp"], "bid"),
            number(row["ap"], "ask"),
            *(int(size) * size_multiplier for size in sizes),
            str(row.get("bx", "")),
            str(row.get("ax", "")),
        )

    @property
    def two_sided(self) -> bool:
        return 0 < self.bid <= self.ask and min(self.bid_size, self.ask_size) > 0

    @property
    def bid_key(self) -> tuple[float, int, str]:
        return self.bid, self.bid_size, self.bid_exchange


@dataclass(frozen=True, slots=True)
class SymbolData:
    previous_close: float | None
    history_volumes: tuple[float | None, ...]
    bars: tuple[Bar, ...]
    quotes: tuple[Quote, ...]
    execution_until: pd.Timestamp | None
    split: bool
    headlines: tuple[dict, ...]


@dataclass(frozen=True, slots=True)
class Session:
    date: str
    previous_sessions: tuple[str, ...]
    settlement_date: str
    premarket: pd.Timestamp
    scan_at: pd.Timestamp
    market_open: pd.Timestamp
    market_close: pd.Timestamp
    observed_through: pd.Timestamp
    feed: str
    symbols: dict[str, SymbolData]
    provenance: dict

    @classmethod
    def read(cls, document: dict) -> Session:
        if document.get("schema") != 1:
            raise ValueError("Unsupported dayrank dataset schema")
        day = date.fromisoformat(document["date"]).isoformat()
        previous = tuple(
            date.fromisoformat(d).isoformat() for d in document["previous_sessions"]
        )
        settlement_date = date.fromisoformat(document["settlement_date"]).isoformat()
        if previous != tuple(sorted(set(previous))) or any(d >= day for d in previous):
            raise ValueError(
                "previous_sessions must be unique trading dates before the session"
            )
        if len(previous) < 4 or settlement_date <= day:
            raise ValueError(
                "An exchange calendar with four preceding sessions is required"
            )
        multiplier = document.get("quote_size_multiplier")
        if (
            not isinstance(multiplier, int)
            or isinstance(multiplier, bool)
            or multiplier < 1
        ):
            raise ValueError("quote_size_multiplier must be a positive integer")
        premarket, scan_at, opening, closing, through = (
            timestamp(document[k])
            for k in (
                "premarket",
                "scan_at",
                "market_open",
                "market_close",
                "observed_through",
            )
        )
        if not premarket < scan_at < opening < closing or through < scan_at:
            raise ValueError(
                "Invalid ordering of scan, open, close, or observation times"
            )
        if any(
            ts.tz_convert(ET).date().isoformat() != day
            for ts in (premarket, opening, through)
        ):
            raise ValueError(
                "Session timestamps must belong to the declared New York date"
            )

        symbols = {}
        for name, row in sorted(document["symbols"].items()):
            if name != name.strip().upper() or not name:
                raise ValueError(f"Invalid symbol: {name!r}")
            bars = tuple(
                sorted((Bar.read(b) for b in row["bars"]), key=lambda b: b.start)
            )
            if len({b.start for b in bars}) != len(bars):
                raise ValueError(f"{name} has duplicate bar timestamps")
            quotes = tuple(
                sorted(
                    (Quote.read(q, multiplier) for q in row["quotes"]),
                    key=lambda q: q.time,
                )
            )
            if any(
                b.start < premarket or b.available_at > through for b in bars
            ) or any(q.time < premarket or q.time > through for q in quotes):
                raise ValueError(
                    f"{name} contains observations outside the captured window"
                )
            execution_until = row.get("execution_until")
            execution_until = (
                timestamp(execution_until) if execution_until is not None else None
            )
            if execution_until is not None and not opening < execution_until <= through:
                raise ValueError(f"{name} has an invalid execution cutoff")
            history = tuple(
                None if v is None else number(v, "historical volume")
                for v in row["history_volumes"]
            )
            if len(history) != len(previous):
                raise ValueError(
                    f"{name} has a different number of volume and calendar sessions"
                )
            close = row["previous_close"]
            symbols[name] = SymbolData(
                number(close, "previous close") if close is not None else None,
                history,
                bars,
                quotes,
                execution_until,
                bool(row.get("split", False)),
                tuple(row.get("headlines", [])),
            )
        if not symbols:
            raise ValueError("Dataset must contain at least one symbol")
        return cls(
            day,
            previous,
            settlement_date,
            premarket,
            scan_at,
            opening,
            closing,
            through,
            document["feed"],
            symbols,
            document["provenance"],
        )


def read_session(path: Path) -> Session:
    return Session.read(json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(
                document, handle, allow_nan=False, sort_keys=True, separators=(",", ":")
            )
            handle.write("\n")
        except (OSError, TypeError, ValueError):
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
