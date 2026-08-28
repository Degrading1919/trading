# Point-in-time data contract

## 1. Provider boundary

`MarketDataProvider` supplies raw observations in three tables. It must not supply
precomputed ranks or factors. The engine owns filtering, formulas, normalization,
and scoring.

Every versioned observation has an `available_at` timestamp: the earliest instant
the strategy could actually have consumed that exact version. Timestamps must be
parseable with an explicit offset and are converted to UTC. Session and security
effective dates use the configured market timezone (`America/New_York` by
default). The engine selects the latest version known at the run's `as_of`
timestamp.

The CSV adapter may return a superset of rows. The engine performs defense-in-depth
time filtering and records discarded future rows. A database adapter may push down
the same predicates for performance, but must return equivalent content.

## 2. `universe.csv`

Required columns:

| Column | Meaning |
|---|---|
| `symbol` | Stable security symbol for this dataset; uppercase normalization is applied |
| `security_type` | Canonical type such as `COMMON_STOCK` |
| `country` | Issuer country code (`US` or `USA` by default) |
| `exchange` | Canonical primary exchange |
| `currency` | Trading currency |
| `sector` | Point-in-time sector classification |
| `industry` | Point-in-time industry classification; reported but not scored in v1 |
| `market_cap` | Point-in-time USD market cap known for this version |
| `is_primary_listing` | Boolean |
| `effective_from` | First active date for this security-master version/state |
| `effective_to` | Exclusive inactive date, blank while active |
| `available_at` | When this version became knowable |

Multiple rows per symbol are expected when membership, classification, or market
cap changes. `symbol + available_at` must identify a unique version. The engine
first chooses the latest known version and then evaluates effective dates; this
prevents an old active record from resurfacing after a delisting update.

For historical work, the table must include securities that later delisted or
changed symbols. A permanent security identifier should replace ticker as the join
key in a production provider; the CSV adapter uses symbol only as an initial,
explicit limitation.

## 3. `prices.csv`

Required columns:

| Column | Meaning |
|---|---|
| `symbol` | Security key matching the universe |
| `session_date` | Exchange trading session represented by the completed daily bar |
| `close` | Raw close used with volume for dollar liquidity and the price screen |
| `split_adjusted_close` | Close adjusted for splits only, used for 52-week-high proximity |
| `total_return_index` | Positive, consistently based cumulative total-return index used for returns |
| `volume` | Nonnegative raw share volume consistent with `close` |
| `available_at` | When this exact row/version became usable |

All price/index values must be positive and finite. Volume must be nonnegative.
When revisions exist, provide multiple versions; the latest version available at
`as_of` wins for each `symbol + session_date`.

`total_return_index` must incorporate distributions and corporate actions without
using corporate-action knowledge unavailable at the historical decision time. A
vendor's present-day adjusted-close download is not automatically a valid
point-in-time historical series. Likewise, split adjustments and volume must remain
internally consistent. The engine intentionally does not guess missing actions or
replace bad observations with model output.

Daily bars should be timestamped as available only after completion and after the
provider's ingestion delay. A run at 15:00 New York time must not see that day's
official close simply because its `session_date` equals the run date.

## 4. `earnings.csv` (optional)

Required columns when supplied:

| Column | Meaning |
|---|---|
| `symbol` | Security key matching the universe |
| `fiscal_period_end` | Quarter end represented by the EPS value |
| `reported_eps` | Originally reported diluted quarterly EPS on a consistent basis |
| `announced_at` | Public earnings-release timestamp |
| `available_at` | Timestamp this exact version reached the system |

`available_at` cannot precede `announced_at`, and the fiscal period cannot end
after its announcement. Multiple versions support corrections/restatements; the
latest one known at `as_of` is selected. Historical replay must retain the original
version rather than overwrite it with today's restated history.

EPS must be consistent through splits as they were knowable at the decision time.
Do not mix GAAP and vendor-defined adjusted EPS within a symbol's history. Missing
earnings are allowed and produce a zero earnings contribution with reduced coverage.

## 5. Reproducibility artifacts

The manifest stores:

- exact UTC `as_of`, engine version, and shared reference market session;
- full serialized configuration and its SHA-256 hash;
- canonical SHA-256 hashes for the filtered universe, price, and earnings frames;
- a combined input hash and deterministic run fingerprint;
- input, known-row, candidate, exclusion, and discarded-future counts;
- Python, pandas, and NumPy versions.

Canonical frame hashing ignores input row and column order. Future observations
discarded by the point-in-time boundary are not included in the input hash. The
same known information, as-of timestamp, and configuration therefore produce the
same fingerprint and ranking.

## 6. Future provider checklist

Before accepting a new feed, validate that it can answer all of these questions:

1. Can it reconstruct the security universe, classifications, and delistings as
   they were known on an arbitrary historical date?
2. Does it expose observation and revision availability timestamps?
3. Can it provide original quarterly facts rather than only latest-restated facts?
4. Are prices, volume, distributions, splits, symbol changes, and delisting returns
   internally consistent?
5. Can a frozen provider extract be checksummed and retained with a research run?
6. Does replaying a prior `as_of` after appending newer data leave the result
   unchanged?

If any answer is no, the feed may be useful for current research but is not valid
for unbiased historical replay.
