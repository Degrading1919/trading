"""Read-only Alpaca adapter for archived and same-day snapshots."""

from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from dayrank.data import ET, MINUTE, Session, at, timestamp
from dayrank.scan import SELECTIONS, scan

DATA = "https://data.alpaca.markets"
PAPER = "https://paper-api.alpaca.markets"


class DataRequestError(RuntimeError):
    def __init__(self, path: str, status: int | None):
        super().__init__(
            f"Alpaca {path}: {'HTTP ' + str(status) if status else 'network failure'}"
        )
        self.status = status


class Alpaca:
    def __init__(self, key: str, secret: str):
        if not key or not secret:
            raise ValueError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    @classmethod
    def from_environment(cls) -> Alpaca:
        return cls(
            os.environ.get("APCA_API_KEY_ID", ""),
            os.environ.get("APCA_API_SECRET_KEY", ""),
        )

    @classmethod
    def from_windows_credential(cls, target: str) -> Alpaca:
        """Read an explicitly named generic credential without exposing its values."""
        if os.name != "nt":
            raise ValueError("--credential-target requires Windows Credential Manager")
        import ctypes
        from ctypes import wintypes

        class Credential(ctypes.Structure):
            _fields_ = [
                ("flags", wintypes.DWORD),
                ("kind", wintypes.DWORD),
                ("target", wintypes.LPWSTR),
                ("comment", wintypes.LPWSTR),
                ("written", wintypes.FILETIME),
                ("blob_size", wintypes.DWORD),
                ("blob", ctypes.c_void_p),
                ("persist", wintypes.DWORD),
                ("attribute_count", wintypes.DWORD),
                ("attributes", ctypes.c_void_p),
                ("target_alias", wintypes.LPWSTR),
                ("username", wintypes.LPWSTR),
            ]

        advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        pointer = ctypes.POINTER(Credential)()
        advapi.CredReadW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(type(pointer)),
        ]
        advapi.CredReadW.restype = wintypes.BOOL
        advapi.CredFree.argtypes = [ctypes.c_void_p]
        advapi.CredFree.restype = None
        if not advapi.CredReadW(target, 1, 0, ctypes.byref(pointer)):
            raise ValueError(f"Windows generic credential {target!r} is unavailable")
        try:
            row = pointer.contents
            secret = ctypes.string_at(row.blob, row.blob_size).decode("utf-16-le")
            return cls((row.username or "").strip(), secret.rstrip("\x00").strip())
        finally:
            advapi.CredFree(pointer)

    def get(self, host: str, path: str, **query):
        request = Request(f"{host}{path}?{urlencode(query)}", headers=self._headers)
        for attempt in range(3):
            try:
                with urlopen(request, timeout=30) as response:
                    return json.load(response)
            except HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise DataRequestError(path, error.code) from None
                wait = error.headers.get("Retry-After", "")
                time.sleep(min(float(wait), 60) if wait.isdecimal() else 2**attempt)
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise DataRequestError(path, None) from None
                time.sleep(2**attempt)

    def pages(self, path: str, **query):
        seen = set()
        while True:
            page = self.get(DATA, path, **query)
            yield page
            token = page.get("next_page_token")
            if not token:
                return
            if token in seen:
                raise ValueError(f"Repeated Alpaca pagination token for {path}")
            seen.add(token)
            query["page_token"] = token

    def stocks(self, endpoint: str, names: list[str], start, end, **query) -> dict:
        result: dict[str, list] = {name: [] for name in names}
        for index in range(0, len(names), 50):
            batch = names[index : index + 50]
            for page in self.pages(
                f"/v2/stocks/{endpoint}",
                symbols=",".join(batch),
                limit=10_000,
                start=start.isoformat(),
                end=end.isoformat(),
                **query,
            ):
                for name, items in (page.get(endpoint) or {}).items():
                    if name in result:
                        result[name].extend(items)
        return result


def _news(client: Alpaca, names: list[str], start, end) -> tuple[dict, str]:
    by_symbol: dict[str, dict] = {name: {} for name in names}
    try:
        for index in range(0, len(names), 50):
            for page in client.pages(
                "/v1beta1/news",
                symbols=",".join(names[index : index + 50]),
                limit=50,
                start=start.isoformat(),
                end=end.isoformat(),
                include_content="false",
            ):
                for item in page.get("news") or []:
                    for name in set(item["symbols"]) & by_symbol.keys():
                        by_symbol[name][item["id"]] = {
                            k: item.get(k)
                            for k in (
                                "id",
                                "headline",
                                "created_at",
                                "updated_at",
                                "source",
                                "url",
                            )
                        }
    except DataRequestError as error:
        if error.status not in {403, 422}:
            raise
        return {}, f"http_{error.status}"
    return {
        name: list(items.values()) for name, items in by_symbol.items()
    }, "available"


def _splits(client: Alpaca, names: list[str], day: date) -> tuple[set[str], str]:
    found = set()
    try:
        for index in range(0, len(names), 50):
            for page in client.pages(
                "/v1/corporate-actions",
                symbols=",".join(names[index : index + 50]),
                types="forward_split,reverse_split",
                start=(day - timedelta(days=10)).isoformat(),
                end=(day + timedelta(days=7)).isoformat(),
                limit=1_000,
            ):
                for kind in ("forward_splits", "reverse_splits"):
                    for item in (page.get("corporate_actions") or {}).get(kind) or []:
                        effective = item.get("ex_date") or item.get("effective_date")
                        if effective == day.isoformat():
                            found.add(item["symbol"])
    except DataRequestError as error:
        if error.status not in {403, 422}:
            raise
        return set(), f"http_{error.status}"
    return found, "available"


def capture(
    client: Alpaca,
    day: date,
    names: list[str],
    *,
    feed: str = "sip",
    history_sessions: int = 20,
    scan_before: int = 5,
    quote_minutes: int = 75,
    shortlist: int = 10,
    scan_only: bool = False,
    quote_size_multiplier: int | None = None,
) -> dict:
    if (
        feed not in {"sip", "iex"}
        or not names
        or len(set(names)) != len(names)
        or not 10 <= history_sessions <= 60
        or not 1 <= scan_before <= 60
        or not 1 <= quote_minutes <= 240
        or shortlist < 1
        or (quote_size_multiplier is not None and quote_size_multiplier < 1)
    ):
        raise ValueError("Invalid feed, symbol list, or capture settings")
    calendar = client.get(
        PAPER,
        "/v2/calendar",
        start=(day - timedelta(days=history_sessions * 2 + 14)).isoformat(),
        end=(day + timedelta(days=10)).isoformat(),
    )
    current = next((r for r in calendar if r["date"] == day.isoformat()), None)
    if not current:
        raise ValueError(f"{day} is not a trading session")
    prior = sorted(
        (r for r in calendar if r["date"] < current["date"]), key=lambda r: r["date"]
    )
    prior = prior[-history_sessions:]
    following = sorted(r["date"] for r in calendar if r["date"] > current["date"])
    if len(prior) != history_sessions or not following:
        raise ValueError("Trade calendar is incomplete")
    opening = at(current["date"], current["open"])
    closing = at(current["date"], current["close"])
    start_hour = "04:00" if feed == "sip" else "08:00"
    premarket = at(current["date"], start_hour)
    scan_at = opening - scan_before * MINUTE
    through = scan_at if scan_only else min(closing, opening + quote_minutes * MINUTE)
    delay = 16 * MINUTE if feed == "sip" else pd.Timedelta(0)
    if pd.Timestamp.now(tz="UTC") < through + delay:
        raise ValueError(
            f"{feed.upper()} data through {through.isoformat()} is not available yet"
        )

    baseline = client.stocks(
        "bars",
        names,
        at(prior[0]["date"], "00:00"),
        at(prior[-1]["date"], "23:59"),
        feed=feed,
        timeframe="1Day",
        adjustment="raw",
        asof="-",
    )
    daily = {
        name: {
            timestamp(bar["t"]).tz_convert(ET).date().isoformat(): bar["c"]
            for bar in bars
        }
        for name, bars in baseline.items()
    }
    volumes: dict[str, list] = {name: [] for name in names}
    for record in prior:
        begin = at(record["date"], start_hour)
        end = at(record["date"], record["open"]) - scan_before * MINUTE
        minutes = client.stocks(
            "bars",
            names,
            begin,
            end - MINUTE,
            feed=feed,
            timeframe="1Min",
            adjustment="raw",
            asof="-",
        )
        for name in names:
            bars = [bar for bar in minutes[name] if begin <= timestamp(bar["t"]) < end]
            known_session = record["date"] in daily[name]
            volumes[name].append(
                sum(b["v"] for b in bars) if bars or known_session else None
            )

    bars = client.stocks(
        "bars",
        names,
        premarket,
        through - MINUTE,
        feed=feed,
        timeframe="1Min",
        adjustment="raw",
        asof="-",
    )
    news, news_status = _news(client, names, scan_at - 36 * 60 * MINUTE, scan_at)
    splits, split_status = _splits(client, names, day)
    multiplier = quote_size_multiplier
    if multiplier is None:
        multiplier = 100 if day < date(2025, 11, 3) else 1
    document = {
        "schema": 1,
        "date": current["date"],
        "previous_sessions": [r["date"] for r in prior],
        "settlement_date": current.get("settlement_date") or following[0],
        "premarket": premarket.isoformat(),
        "scan_at": scan_at.isoformat(),
        "market_open": opening.isoformat(),
        "market_close": closing.isoformat(),
        "observed_through": through.isoformat(),
        "feed": feed,
        "quote_size_multiplier": multiplier,
        "provenance": {
            "provider": "alpaca",
            "symbol_source": "caller_supplied",
            "capture_mode": "scan_only" if scan_only else "opening_replay",
            "news_status": news_status,
            "split_status": split_status,
            "settlement_source": "alpaca_calendar"
            if current.get("settlement_date")
            else "next_trading_session_assumption",
            "captured_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "quote_shortlist": shortlist,
        },
        "symbols": {
            name: {
                "previous_close": daily[name].get(prior[-1]["date"]),
                "history_volumes": volumes[name],
                "bars": [
                    b
                    for b in bars[name]
                    if premarket <= timestamp(b["t"]) <= through - MINUTE
                ],
                "split": name in splits,
                "headlines": news.get(name, []),
                "quotes": [],
                "execution_until": None,
            }
            for name in names
        },
    }
    parsed = Session.read(document)
    selected = sorted(
        {
            row["symbol"]
            for mode in SELECTIONS
            for row in scan(parsed, mode)
            if row["rank"] is not None and row["rank"] <= shortlist
        }
    )
    pre_quotes = client.stocks("quotes", selected, scan_at - MINUTE, scan_at, feed=feed)
    regular = (
        {}
        if scan_only
        else client.stocks("quotes", selected, opening, through, feed=feed)
    )
    for name in selected:
        row = document["symbols"][name]
        row["quotes"] = [
            q
            for q in pre_quotes[name] + regular.get(name, [])
            if premarket <= timestamp(q["t"]) <= through
        ]
        if not scan_only:
            row["execution_until"] = through.isoformat()
    Session.read(document)
    return document
