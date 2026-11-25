# Stock-Trader

## Day trading research

Use `dayrank` to capture historical Alpaca data, build a premarket watchlist,
and compare opening breakouts with quote-based fills and settled cash.
[Run the day trading workflow](docs/DAY_TRADING.md).

## Swing research

`swingrank` is an initial, research-oriented U.S. equity screening and ranking
engine for roughly one-to-two-month candidate holding periods. It calculates
market data; it never asks an LLM to supply, infer, or repair prices,
fundamentals, classifications, or timestamps.

The fixed baseline is preserved and an optional adaptive layer now sits beside it:

- point-in-time common-stock eligibility and tradability filters;
- six fixed, literature-motivated factor contributions;
- deterministic cross-sectional normalization and ranking;
- a rolling, L2-regularized logistic model over those same six features;
- probability, learned coefficient, and per-stock log-odds explanations;
- provider-neutral raw-data interfaces;
- candidate, exclusion, and run-manifest artifacts suitable for audit or replay.

It is a research screen, not a portfolio constructor, execution system, expected
return estimator, or investment recommendation. The adaptive probability is a
relative event forecast whose calibration must be established out of sample.

## Method at a glance

The unchanged baseline score combines non-overlapping 12-to-7-month and
6-to-1-month total returns,
52-week-high proximity, leave-one-out sector momentum, standardized unexpected
quarterly earnings, and a 63-session low-volatility term. Volume is used as a
liquidity filter because its directional relationship with momentum is
conditional rather than simply monotonic.

All weights and thresholds are declared in `ScreeningConfig`. They are fixed
research priors, not values selected from this project's historical results.
Missing factors receive a zero contribution and their weights are not
redistributed. See [the full methodology](docs/METHODOLOGY.md) and
[the data contract](docs/DATA_CONTRACT.md).

The adaptive layer does not replace that score or invent new features. It learns
six regularized logistic coefficients from complete, materialized historical
baseline cohorts inside a rolling five-year window. Its event is outperformance
of the formation cohort's median delisting-inclusive total return over 42 market
sessions. Each decision date gets a fresh point-in-time fit; future or incomplete
outcomes cannot enter. The output always retains fixed and adaptive ranks for
direct comparison.

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

## Build adaptive training cohorts

Training rows are created only after the full 42-session outcome is known. Start
from a retained fixed-baseline result and an independently calculated outcome
table containing every formation candidate, including explicit delisting returns:

```python
from swingrank.adaptive import build_training_cohort

cohort = build_training_cohort(
    baseline_result,
    forward_returns,  # columns: symbol, forward_return
    label_end_session="2025-03-31",
    label_end_at="2025-03-31T21:05:00Z",
    available_at="2025-03-31T21:10:00Z",
)
```

Append each returned cohort to a versioned training store; never overwrite older
versions. The exact contract and survivorship requirements are in
[the data contract](docs/DATA_CONTRACT.md#7-adaptive-training-panel-contract).

## Run baseline and adaptive rankings together

With current raw market CSVs plus the historical training panel:

```powershell
swingrank-adaptive `
  --data-dir C:\point-in-time-feed\2025-01-31 `
  --training-csv C:\research-store\adaptive-training.csv `
  --as-of 2025-01-31T21:30:00Z `
  --output-dir C:\screen-output `
  --top 25
```

This writes the untouched baseline candidates/exclusions/manifest alongside:

- adaptive candidates with probability, both ranks, and six additive log-odds
  contributions;
- learned coefficients, odds ratios, relative absolute weights, and fixed weights;
- an adaptive manifest containing the exact training window, input hash, model
  state, diagnostics, discarded-future counts, and run fingerprint.

Programmatic walk-forward replay uses
`AdaptiveRankingEngine.run_walk_forward(decision_dates)`. Every timestamp is fit
independently from samples available at that timestamp.

## Test

```powershell
python -m ruff check src tests
python -m pytest
```

The test suite covers exact factor formulas, cross-sectional normalization,
eligibility filters, fixed-weight score arithmetic, adaptive coefficient and
probability arithmetic, full-cohort enforcement, revision isolation,
future-market/training-data isolation, walk-forward cutoffs, stable ordering, and
run reproducibility.

## Package map

```text
src/swingrank/
  adaptive/        rolling training contract, models, engine, and artifacts
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
