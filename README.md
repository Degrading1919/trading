# swingrank

`swingrank` is an initial, research-oriented U.S. equity screening and ranking
engine for roughly one-to-two-month candidate holding periods. It calculates
market data; it never asks an LLM to supply, infer, or repair prices,
fundamentals, classifications, or timestamps.

The package is intentionally narrow:

- point-in-time common-stock eligibility and tradability filters;
- six fixed, literature-motivated factor contributions;
- deterministic cross-sectional normalization and ranking;
- provider-neutral raw-data interfaces;
- candidate, exclusion, and run-manifest artifacts suitable for audit or replay.

It is a research screen, not a portfolio constructor, execution system, return
forecast, or investment recommendation.

## Method at a glance

The score combines non-overlapping 12-to-7-month and 6-to-1-month total returns,
52-week-high proximity, leave-one-out sector momentum, standardized unexpected
quarterly earnings, and a 63-session low-volatility term. Volume is used as a
liquidity filter because its directional relationship with momentum is
conditional rather than simply monotonic.

All weights and thresholds are declared in `ScreeningConfig`. They are fixed
research priors, not values selected from this project's historical results.
Missing factors receive a zero contribution and their weights are not
redistributed. See [the full methodology](docs/METHODOLOGY.md) and
[the data contract](docs/DATA_CONTRACT.md).

## Install

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Python 3.11 or newer is required.

## Run a CSV snapshot

Prepare `universe.csv`, `prices.csv`, and optionally `earnings.csv` according to
the data contract, then run after the market data intended for the decision is
actually available:

```powershell
swingrank `
  --data-dir C:\point-in-time-feed\2024-06-28 `
  --as-of 2024-06-28T21:00:00Z `
  --output-dir C:\screen-output `
  --top 25
```

The command writes:

- `screen.candidates.csv`: ranks, raw factor values, normalized values, and
  additive contributions;
- `screen.exclusions.csv`: every rejected symbol and its reason(s);
- `screen.manifest.json`: methodology/configuration, exact input hashes,
  point-in-time discard counts, dependency versions, and a run fingerprint.

The reference CSV adapter favors clarity over throughput. Implement
`MarketDataProvider` for a database, columnar store, or commercial feed; the
screening methodology does not change.

## Test

```powershell
python -m ruff check src tests
python -m pytest
```

The test suite covers exact factor formulas, cross-sectional normalization,
eligibility filters, fixed-weight score arithmetic, future-data isolation,
point-in-time universe membership, stable ordering, and run reproducibility.

## Package map

```text
src/swingrank/
  providers/       raw provider protocol and CSV/in-memory adapters
  schema.py        timestamp, revision, and point-in-time enforcement
  factors.py       pure signal calculations
  normalization.py deterministic cross-sectional ranks
  engine.py        filtering, scoring, ranking, and audit orchestration
  result.py        output artifacts
docs/
  METHODOLOGY.md
  DATA_CONTRACT.md
tests/
```

