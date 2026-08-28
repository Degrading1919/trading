"""Command-line entry point for side-by-side baseline/adaptive CSV runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from swingrank.adaptive.config import AdaptiveConfig
from swingrank.adaptive.engine import AdaptiveRankingEngine
from swingrank.adaptive.providers import CsvTrainingDataProvider
from swingrank.config import ScreeningConfig
from swingrank.exceptions import SwingRankError
from swingrank.providers.csv import CsvDataProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swingrank-adaptive",
        description=(
            "Run the fixed baseline and point-in-time adaptive probability ranker."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing current universe/prices/earnings CSV files",
    )
    parser.add_argument(
        "--training-csv",
        type=Path,
        required=True,
        help="Versioned historical baseline cohorts and complete 42-session outcomes",
    )
    parser.add_argument(
        "--as-of",
        required=True,
        help="Timezone-aware decision timestamp, e.g. 2025-01-31T21:00:00Z",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="screen")
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Print only N candidates; output artifacts always contain the full universe",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        as_of = pd.Timestamp(args.as_of)
        if as_of.tzinfo is None:
            raise ValueError("--as-of must include a timezone or UTC offset")
        result = AdaptiveRankingEngine(
            CsvDataProvider(args.data_dir),
            CsvTrainingDataProvider(args.training_csv),
            baseline_config=ScreeningConfig(),
            adaptive_config=AdaptiveConfig(),
        ).run(as_of)
        paths = result.write(args.output_dir, args.prefix)
    except (SwingRankError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    columns = [
        "adaptive_rank",
        "symbol",
        "probability_outperformance",
        "confidence_from_50pct",
        "baseline_rank",
        "baseline_score",
    ]
    display = result.candidates.loc[:, columns]
    if args.top is not None:
        display = display.head(args.top)
    print(display.to_string(index=False))
    print(f"Adaptive candidates: {paths['adaptive_candidates']}")
    print(f"Learned weights: {paths['adaptive_weights']}")
    print(f"Adaptive manifest: {paths['adaptive_manifest']}")
    print(f"Baseline candidates: {paths['baseline_candidates']}")
    print(f"Run fingerprint: {result.manifest['run_fingerprint']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
