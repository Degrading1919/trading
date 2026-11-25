"""Command line for capturing, screening, and comparing intraday setups."""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import re
import sys
import tempfile
from dataclasses import asdict
from datetime import date, time
from pathlib import Path

from dayrank import METHODOLOGY_VERSION
from dayrank.alpaca import Alpaca, DataRequestError, capture
from dayrank.data import Session, read_session, write_json
from dayrank.discovery import discover, premarket_window
from dayrank.indicators import IndicatorRules, collect_tape, evaluate_tape, summarize
from dayrank.recorder import read_recording, record
from dayrank.replay import SETUPS, ReplayConfig, replay
from dayrank.scan import SELECTIONS, ScanConfig, scan
from dayrank.universe import download_directory


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="dayrank", description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    universe = commands.add_parser(
        "universe", help="Freeze an exchange security directory"
    )
    universe.add_argument("--archive-date", type=date.fromisoformat)
    universe.add_argument(
        "--archive-ref", default="main", help="Git reference for a dated snapshot"
    )
    universe.add_argument("--out", type=Path, required=True)

    find = commands.add_parser(
        "discover", help="Audit premarket discovery for a frozen universe"
    )
    tape = commands.add_parser("record", help="Record premarket bar receipt times")
    for command, feed in ((find, "sip"), (tape, "iex")):
        command.add_argument("--date", type=date.fromisoformat, required=True)
        command.add_argument("--universe", type=Path, required=True)
        command.add_argument("--feed", choices=("sip", "iex"), default=feed)
        command.add_argument("--scan-before", type=int, default=5)
        command.add_argument(
            "--start-at", type=time.fromisoformat, help="Override window start in ET"
        )
        command.add_argument(
            "--credential-target", help="Name of a Windows generic credential"
        )
        command.add_argument("--out", type=Path, required=True)
    find.add_argument("--history-sessions", type=int, default=20)
    find.add_argument("--bar-lag-seconds", type=int, default=35)
    find.add_argument("--recording", type=Path, help="Replay actual bar receipt times")

    indicators = commands.add_parser(
        "indicators", help="Audit breakout indicators and build a historical review set"
    )
    indicators.add_argument(
        "inputs", type=Path, nargs="+", help="Discovery folders or saved tapes"
    )
    indicators.add_argument(
        "--premarket-only",
        action="store_true",
        help="Collect only through discovery cutoff",
    )
    indicators.add_argument("--review-size", type=int, default=72)
    indicators.add_argument(
        "--credential-target", help="Name of a Windows generic credential"
    )
    indicators.add_argument("--out", type=Path, required=True)

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


def _client(args) -> Alpaca:
    return (
        Alpaca.from_windows_credential(args.credential_target)
        if args.credential_target
        else Alpaca.from_environment()
    )


def _universe(args) -> None:
    directory = download_directory(args.archive_date, ref=args.archive_ref)
    write_json(args.out, directory)
    eligible = sum(not row["listing_reasons"] for row in directory["rows"])
    print(
        f"Froze {eligible:,} eligible listings from {directory['available_at']}. {args.out}"
    )


def _discover(args) -> None:
    raw = args.universe.read_bytes()
    document = discover(
        _client(args),
        json.loads(raw),
        args.date,
        feed=args.feed,
        lookback=args.history_sessions,
        scan_before=args.scan_before,
        bar_lag_seconds=args.bar_lag_seconds,
        start_at=args.start_at,
        recorded=read_recording(args.recording) if args.recording else None,
    )
    _csv(args.out / "symbols.csv", document["audit"], ("symbol", "cutoff_exclusion"))
    _csv(args.out / "alerts.csv", document["alerts"], ("symbol", "rule", "observed_at"))
    cutoff_names = sorted(
        {row["symbol"] for row in document["audit"] if row["cutoff_rules"]}
    )
    (args.out / "eligible-symbols.txt").write_text(
        "".join(f"{name}\n" for name in cutoff_names),
        encoding="utf-8",
    )
    manifest = document["manifest"]
    manifest["universe_json_sha256"] = hashlib.sha256(raw).hexdigest()
    write_json(args.out / "manifest.json", manifest)
    print(
        f"{manifest['date']} {manifest['feed'].upper()} audited {manifest['listing_rows']:,} listings."
    )
    for rule, first in manifest["first_alerts"].items():
        print(
            f"{rule}: {first} first alerts, {manifest['cutoff_candidates'][rule]} at cutoff"
        )
    print(f"Discovery artifacts: {args.out}")


def _record(args) -> None:
    client = _client(args)
    directory = json.loads(args.universe.read_bytes())
    window = premarket_window(
        client, args.date, args.feed, 10, args.scan_before, start_at=args.start_at
    )
    print(
        f"Recording {args.feed.upper()} through {window.cutoff.isoformat()}.",
        flush=True,
    )
    count = asyncio.run(record(client, directory, window, args.feed, args.out))
    print(f"Recorded {count:,} bar events. {args.out}")


def _indicators(args) -> None:
    rules = IndicatorRules()
    if args.review_size < 1:
        raise ValueError("Review size must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    client = None
    for source in args.inputs:
        if source.is_dir():
            manifest = json.loads((source / "manifest.json").read_bytes())
            with (source / "symbols.csv").open(encoding="utf-8", newline="") as handle:
                names = list(csv.DictReader(handle))
            if client is None:
                client = _client(args)
            tape = collect_tape(client, manifest, names, args.premarket_only, rules)
            folder = args.out / "tapes"
            folder.mkdir(exist_ok=True)
            target = folder / f"{tape['date']}-{tape['feed']}.json.gz"
            with tempfile.NamedTemporaryFile(dir=folder, delete=False) as handle:
                temporary = Path(handle.name)
            try:
                with gzip.open(temporary, "wt", encoding="utf-8") as handle:
                    json.dump(tape, handle, allow_nan=False, separators=(",", ":"))
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            with gzip.open(source, "rt", encoding="utf-8") as handle:
                tape = json.load(handle)
        result = evaluate_tape(tape, rules)
        results.append(result)
        print(
            f"{tape['date']} {tape['feed'].upper()} produced {len(result[0]):,} cases.",
            flush=True,
        )
    report = summarize(results, rules, args.review_size)
    _csv(
        args.out / "cases.csv",
        report["cases"],
        ("case_id", "date", "symbol", "pattern", "case", "observation"),
    )
    _csv(
        args.out / "review.csv",
        report["review"],
        ("case_id", "review_label", "review_note"),
    )
    _csv(args.out / "forward.csv", report["outcomes"], ("case_id",))
    summary = report["summary"]
    write_json(args.out / "summary.json", summary)
    print(
        f"Saved {summary['case_count']:,} cases and {summary['review_count']} review rows. {args.out}"
    )


def _capture(args) -> None:
    symbols = (
        args.symbols
        if args.symbols is not None
        else args.symbols_file.read_text(encoding="utf-8-sig")
    )
    document = capture(
        _client(args),
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
        commands = {
            "universe": _universe,
            "discover": _discover,
            "record": _record,
            "indicators": _indicators,
            "capture": _capture,
            "scan": _scan,
            "replay": _replay,
        }
        commands[args.command](args)
    except (DataRequestError, KeyError, TypeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
