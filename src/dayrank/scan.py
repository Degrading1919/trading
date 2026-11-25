"""Point-in-time premarket gates and comparable candidate rankings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import median

import pandas as pd

from dayrank.data import SECOND, Session, timestamp

SELECTIONS = ("gap-only", "rvol", "catalyst-rvol")
FINANCING = re.compile(
    r"\b(offering|private placement|dilution|registered direct|warrants?|reverse split|"
    r"at.the.market|convertible notes?)\b",
    re.I,
)
MATERIAL = re.compile(
    r"\b(reports? earnings|financial results|earnings beat|raises? guidance|cuts? guidance|"
    r"FDA|phase [123]|clinical trial|merger|acquisition|"
    r"contract award|government contract|regulatory approval)\b",
    re.I,
)


@dataclass(frozen=True, slots=True)
class ScanConfig:
    minimum_price: float = 1.0
    maximum_price: float = 20.0
    minimum_gap: float = 0.05
    minimum_rvol: float = 2.0
    minimum_dollar_volume: float = 100_000.0
    minimum_history: int = 10
    premarket_quote_age_seconds: int = 60
    news_hours: int = 36


def _news(rows: tuple[dict, ...], as_of: pd.Timestamp, hours: int) -> tuple[str, str]:
    known = []
    cutoff = as_of - pd.Timedelta(hours=hours)
    for row in rows:
        created = timestamp(row["created_at"])
        changed = timestamp(row["updated_at"])
        if cutoff <= created <= changed <= as_of:
            known.append((created, str(row["headline"]), str(row.get("id", ""))))
    seen: set[str] = set()
    newest = {"financing_or_split": "", "material_keyword": "", "other_news": ""}
    for _, headline, _ in sorted(known, reverse=True):
        key = re.sub(r"[^a-z0-9]+", "", headline.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        kind = "other_news"
        if FINANCING.search(headline):
            kind = "financing_or_split"
        elif MATERIAL.search(headline):
            kind = "material_keyword"
        if not newest[kind]:
            newest[kind] = headline
    return next(
        ((kind, headline) for kind, headline in newest.items() if headline),
        ("none_observed", ""),
    )


def scan(
    session: Session, selection: str, config: ScanConfig = ScanConfig()
) -> list[dict]:
    if selection not in SELECTIONS:
        raise ValueError(f"Unknown selection: {selection}")
    results = []
    for symbol, data in session.symbols.items():
        bars = [b for b in data.bars if b.available_at <= session.scan_at]
        latest = bars[-1].close if bars else None
        previous = data.previous_close
        gap = latest / previous - 1 if latest and previous else None
        volume = sum(b.volume for b in bars)
        dollar_volume = sum(b.close * b.volume for b in bars)
        history = [v for v in data.history_volumes if v is not None]
        expected = median(history) if len(history) >= config.minimum_history else None
        relative = volume / expected if expected else None
        kind, headline = _news(data.headlines, session.scan_at, config.news_hours)
        if (
            kind == "none_observed"
            and session.provenance.get("news_status") != "available"
        ):
            kind = "feed_unavailable"
        quote = next(
            (q for q in reversed(data.quotes) if q.time <= session.scan_at), None
        )
        quote_status = "missing"
        spread = None
        if quote is not None:
            if (
                session.scan_at - quote.time
                > config.premarket_quote_age_seconds * SECOND
            ):
                quote_status = "stale"
            elif not quote.two_sided:
                quote_status = "one_sided_or_crossed"
            else:
                quote_status = "two_sided"
                spread = (quote.ask - quote.bid) / ((quote.ask + quote.bid) / 2)

        reasons = []
        if data.split:
            reasons.append("split_effective")
        if latest is None:
            reasons.append("no_premarket_bars")
        elif not config.minimum_price <= latest <= config.maximum_price:
            reasons.append("price")
        if gap is None or gap < config.minimum_gap:
            reasons.append("gap")
        if dollar_volume < config.minimum_dollar_volume:
            reasons.append("premarket_dollar_volume")
        if selection != "gap-only" and (
            relative is None or relative < config.minimum_rvol
        ):
            reasons.append("rvol_unavailable" if relative is None else "rvol")
        if selection == "catalyst-rvol" and kind != "material_keyword":
            reasons.append("material_catalyst_unverified")
        results.append(
            dict(
                symbol=symbol,
                rank=None,
                eligible=not reasons,
                reasons="|".join(reasons),
                previous_close=previous,
                premarket_price=latest,
                premarket_high=max((b.high for b in bars), default=None),
                gap=gap,
                premarket_volume=volume,
                premarket_dollar_volume=dollar_volume,
                historical_sessions=len(history),
                expected_volume=expected,
                relative_volume=relative,
                catalyst=kind,
                headline=headline,
                quote_status=quote_status,
                spread_pct=spread,
            )
        )
    results.sort(
        key=lambda r: (
            not r["eligible"],
            -(
                (r["relative_volume"] or 0)
                if selection != "gap-only"
                else (r["gap"] or 0)
            ),
            -(r["gap"] or 0),
            r["symbol"],
        )
    )
    position = 0
    for row in results:
        if row["eligible"]:
            position += 1
            row["rank"] = position
    return results
