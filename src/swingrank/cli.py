"""Command-line entry point for CSV-based screening runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from swingrank.config import ScreeningConfig
from swingrank.engine import ScreeningEngine
from swingrank.exceptions import SwingRankError
from swingrank.providers.csv import CsvDataProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swingrank",
        description="Run the fixed, explainable swing-trading equity screen.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing universe.csv, prices.csv, and optional earnings.csv",
    )
    parser.add_argument(
        "--as-of",
        required=True,
        help="Timezone-aware decision timestamp, e.g. 2024-06-28T21:00:00Z",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="screen")
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Print only the first N candidates; artifacts always contain all candidates",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        as_of = pd.Timestamp(args.as_of)
        if as_of.tzinfo is None:
            raise ValueError("--as-of must include a timezone or UTC offset")
        provider = CsvDataProvider(args.data_dir)
        result = ScreeningEngine(provider, ScreeningConfig()).run(as_of)
        paths = result.write(args.output_dir, args.prefix)
    except (SwingRankError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    display = (
        result.candidates if args.top is None else result.candidates.head(args.top)
    )
    if display.empty:
        print("No securities passed the screen.")
    else:
        print(display.to_string(index=False))
    print(f"Candidates: {paths['candidates']}")
    print(f"Exclusions: {paths['exclusions']}")
    print(f"Manifest: {paths['manifest']}")
    print(f"Run fingerprint: {result.manifest['run_fingerprint']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
