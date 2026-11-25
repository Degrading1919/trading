"""Command line for capturing, screening, and comparing intraday setups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from dayrank import METHODOLOGY_VERSION
from dayrank.alpaca import Alpaca, DataRequestError, capture
from dayrank.data import Session, read_session, write_json
from dayrank.replay import SETUPS, ReplayConfig, replay
from dayrank.scan import SELECTIONS, ScanConfig, scan


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="dayrank", description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("capture", help="Save an Alpaca session snapshot")
    collect.add_argument("--date", type=date.fromisoformat, required=True)
    symbols = collect.add_mutually_exclusive_group(required=True)
    symbols.add_argument("--symbols", help="Comma-separated symbols")
    symbols.add_argument(
        "--symbols-file", type=Path, help="Plain-text roster, one symbol per line"
    )
    collect.add_argument("--feed", choices=("sip", "iex"), default="sip")
    collect.add_argument("--history-sessions", type=int, default=20)
    collect.add_argument("--scan-before", type=int, default=5)
    collect.add_argument("--quote-minutes", type=int, default=75)
    collect.add_argument("--shortlist", type=int, default=10)
    collect.add_argument("--quote-size-multiplier", type=int)
    collect.add_argument("--scan-only", action="store_true")
    collect.add_argument(
        "--credential-target", help="Name of a Windows generic credential"
    )
    collect.add_argument("--out", type=Path, required=True)

    watch = commands.add_parser("scan", help="Export every candidate and exclusion")
    watch.add_argument("session", type=Path)
    watch.add_argument("--selection", choices=SELECTIONS, default="rvol")
    watch.add_argument("--out", type=Path, required=True)

    test = commands.add_parser(
        "replay", help="Run matching setup and selection comparisons"
    )
    test.add_argument("sessions", nargs="+", type=Path)
    test.add_argument("--selection", action="append", choices=SELECTIONS)
    test.add_argument("--setup", action="append", choices=SETUPS)
    test.add_argument("--capital", type=float, default=2_000)
    test.add_argument("--risk-fraction", type=float, default=0.01)
    test.add_argument("--max-exposure-fraction", type=float, default=0.50)
    test.add_argument("--stop-fraction", type=float, default=0.04)
    test.add_argument("--max-spread-to-stop", type=float, default=0.35)
    test.add_argument("--max-chase-fraction", type=float, default=0.03)
    test.add_argument("--cost-bps", type=float, default=5)
    test.add_argument("--latency-ms", type=int, default=250)
    test.add_argument("--last-entry-minutes", type=int, default=30)
    test.add_argument("--exit-minutes", type=int, default=60)
    test.add_argument("--top", type=int, default=10)
    test.add_argument("--round-trips-per-five", type=int)
    test.add_argument("--out", type=Path, required=True)
    return result


def _symbols(raw: str) -> list[str]:
    names = sorted(set(re.split(r"[\s,]+", raw.strip().upper())) - {""})
    if not names or any(
        not re.fullmatch(r"[A-Z][A-Z0-9./-]{0,14}", name) for name in names
    ):
        raise ValueError("Supply valid stock symbols separated by commas or whitespace")
    return names


def _csv(path: Path, rows: list[dict], minimum_fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(
        dict.fromkeys((*minimum_fields, *(key for row in rows for key in row)))
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        # Prevent untrusted headlines from becoming spreadsheet formulas.
        def safe(value):
            if isinstance(value, str) and value.startswith(
                ("=", "+", "-", "@", "\t", "\r", "\n")
            ):
                return "'" + value
            return value

        writer.writerows(
            {key: safe(value) for key, value in row.items()} for row in rows
        )


def _capture(args) -> None:
    symbols = (
        args.symbols
        if args.symbols is not None
        else args.symbols_file.read_text(encoding="utf-8-sig")
    )
    client = (
        Alpaca.from_windows_credential(args.credential_target)
        if args.credential_target
        else Alpaca.from_environment()
    )
    document = capture(
        client,
        args.date,
        _symbols(symbols),
        feed=args.feed,
        history_sessions=args.history_sessions,
        scan_before=args.scan_before,
        quote_minutes=args.quote_minutes,
        shortlist=args.shortlist,
        scan_only=args.scan_only,
        quote_size_multiplier=args.quote_size_multiplier,
    )
    write_json(args.out, document)
    quote_symbols = sum(bool(row["quotes"]) for row in document["symbols"].values())
    print(
        f"Captured {document['date']} {args.feed.upper()}: {len(document['symbols'])} symbols, "
        f"{quote_symbols} with quotes. {args.out}"
    )


def _scan(args) -> None:
    session = read_session(args.session)
    results = scan(session, args.selection)
    _csv(args.out, results, ("symbol", "rank"))
    print(f"{session.date} {session.feed.upper()} at {session.scan_at.isoformat()}")
    for row in [r for r in results if r["eligible"]][:10]:
        rvol = (
            f"{row['relative_volume']:.2f}x"
            if row["relative_volume"] is not None
            else "unknown"
        )
        print(
            f"{row['rank']:2}  {row['symbol']:6}  gap {row['gap']:+.1%}  "
            f"RVOL {rvol}  {row['catalyst']}  {row['quote_status']}"
        )
    print(
        f"{sum(row['eligible'] for row in results)} eligible of {len(results)}. {args.out}"
    )


def _replay(args) -> None:
    config = ReplayConfig(
        **{
            key: value
            for key, value in vars(args).items()
            if key in ReplayConfig.__dataclass_fields__
        }
    )
    sessions = []
    inputs = []
    for file in args.sessions:
        raw = file.read_bytes()
        session = Session.read(json.loads(raw))
        sessions.append(session)
        inputs.append(
            {
                "date": session.date,
                "feed": session.feed,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    inputs.sort(key=lambda item: item["date"])
    variants = []
    trades = []
    balances = []
    for selection in dict.fromkeys(args.selection or SELECTIONS):
        for setup in dict.fromkeys(args.setup or SETUPS):
            result = replay(sessions, selection, setup, config)
            name = f"{selection}/{setup}"
            variants.append({"variant": name, **result["summary"]})
            trades.extend({"variant": name, **row} for row in result["ledger"])
            balances.extend({"variant": name, **row} for row in result["daily"])
    args.out.mkdir(parents=True, exist_ok=True)
    _csv(args.out / "summary.csv", variants, ("variant",))
    _csv(args.out / "trade-ledger.csv", trades, ("variant", "date", "symbol", "status"))
    _csv(args.out / "cash-ledger.csv", balances, ("variant", "date"))
    write_json(
        args.out / "manifest.json",
        {
            "methodology_version": METHODOLOGY_VERSION,
            "inputs": inputs,
            "scan_config": asdict(ScanConfig()),
            "replay_config": asdict(config),
            "variants": [row["variant"] for row in variants],
        },
    )
    for row in variants:
        result = row["return_fraction"]
        display = f"{result:+.2%}" if result is not None else "incomplete"
        print(
            f"{row['variant']}: {row['closed_trades']} closed trades; return {display}"
        )
    print(f"Replay artifacts: {args.out}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        {"capture": _capture, "scan": _scan, "replay": _replay}[args.command](args)
    except (DataRequestError, KeyError, TypeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
