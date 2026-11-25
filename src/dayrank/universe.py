"""Freeze the Nasdaq Trader exchange-listed security directory."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import re
from datetime import date, datetime
from urllib.request import Request, urlopen

import pandas as pd

from dayrank.data import ET, timestamp

DIRECTORY = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqtraded.txt"
ARCHIVE = "https://raw.githubusercontent.com/supermodo/us-markets-timemachine"
EXCHANGES = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Q": "Nasdaq",
    "Z": "Cboe BZX",
    "V": "IEX",
}
SYMBOL = re.compile(r"[A-Z][A-Z0-9./-]{0,14}")
NON_COMMON = re.compile(
    r"\b(warrants?|rights?|units?|preferred|preference|debentures?|notes?|bonds?|"
    r"depositary|depository|ETNs?|closed.end fund|beneficial interests?|limited partnership)\b",
    re.I,
)
COMMON = re.compile(
    r"\b(common (stock|shares?)|ordinary shares?|capital stock|common equity)\b",
    re.I,
)


def parse_directory(
    raw: bytes, source: str, retrieved_at, archive_date: date | None = None
) -> dict:
    decoded = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
    lines = decoded.decode("utf-8-sig").splitlines()
    if len(lines) < 3:
        raise ValueError("Symbol directory is empty or truncated")
    footer = re.fullmatch(r"File Creation Time:\s*(\d{10}:\d{2})\|*", lines[-1].strip())
    if not footer:
        raise ValueError("Nasdaq directory is missing its file creation timestamp")
    published = (
        pd.Timestamp(datetime.strptime(footer[1], "%m%d%Y%H:%M"))
        .tz_localize(ET)
        .tz_convert("UTC")
    )
    retrieved = timestamp(retrieved_at)
    if published > retrieved or (
        archive_date is not None and published.tz_convert(ET).date() > archive_date
    ):
        raise ValueError("Directory publication is later than the declared source date")
    table = csv.DictReader(io.StringIO("\n".join(lines[:-1])), delimiter="|")
    required = {
        "Symbol",
        "Security Name",
        "Listing Exchange",
        "ETF",
        "Test Issue",
        "NextShares",
        "Nasdaq Traded",
    }
    if not required.issubset(table.fieldnames or []):
        raise ValueError("Nasdaq directory columns have changed")
    rows = []
    seen = set()
    for row in table:
        if None in row or any(row[k] is None for k in required):
            raise ValueError("Nasdaq directory contains a malformed row")
        symbol = row["Symbol"].strip()
        name = row["Security Name"].strip()
        security_description = name.rsplit(" - ", 1)[-1]
        exchange = row["Listing Exchange"].strip()
        reasons = []
        if row["Test Issue"] == "Y":
            reasons.append("test_issue")
        if row["ETF"] == "Y" or row["NextShares"] == "Y":
            reasons.append("fund_flag")
        if row["Nasdaq Traded"] != "Y" or exchange not in EXCHANGES:
            reasons.append("listing_outside_scope")
        if NON_COMMON.search(security_description):
            reasons.append("non_common_security")
        elif not COMMON.search(security_description):
            reasons.append("security_type_unverified")
        if not SYMBOL.fullmatch(symbol) or symbol in seen:
            reasons.append("unsupported_or_duplicate_symbol")
        if not reasons:
            seen.add(symbol)
        rows.append(
            dict(
                symbol=symbol,
                name=name,
                exchange=EXCHANGES.get(exchange, exchange),
                listing_reasons="|".join(reasons),
            )
        )
    return dict(
        schema=1,
        source=source,
        archived=archive_date is not None,
        archive_date=archive_date.isoformat() if archive_date is not None else None,
        published_at=published.isoformat(),
        retrieved_at=retrieved.isoformat(),
        available_at=(published if archive_date is not None else retrieved).isoformat(),
        sha256=hashlib.sha256(decoded).hexdigest(),
        rows=rows,
    )


def download_directory(archive_date: date | None = None, ref: str = "main") -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", ref):
        raise ValueError("Use a GitHub branch, tag, or commit without slashes")
    url = (
        DIRECTORY
        if archive_date is None
        else (
            f"{ARCHIVE}/{ref}/data/nasdaq/nasdaqtraded/{archive_date.year}/{archive_date}.txt.gz"
        )
    )
    with urlopen(
        Request(url, headers={"User-Agent": "Stock-Trader/dayrank"}), timeout=30
    ) as res:
        raw = res.read()
    document = parse_directory(raw, url, pd.Timestamp.now(tz="UTC"), archive_date)
    if len(document["rows"]) < 1_000:
        raise ValueError("Exchange directory appears truncated")
    return document
