# Day trading research with dayrank

Dayrank records premarket watchlists and evaluates long-only opening breakouts
using the same session data for each comparison. It makes read-only requests to
Alpaca and writes local artifacts. All defaults are starting assumptions for
research.

## Run a historical session

Install from the repository root with `python -m pip install -e ".[dev]"`.
Supply paper API credentials through `APCA_API_KEY_ID` and
`APCA_API_SECRET_KEY` in your local environment, or pass
`--credential-target` to read a named Windows generic credential.

```powershell
dayrank capture --date 2026-11-23 --symbols SNDL,PLUG,OPEN,LCID --out runs/2026-11-23.json
dayrank scan runs/2026-11-23.json --selection rvol --out runs/watchlist.csv
dayrank replay runs/2026-11-23.json --capital 2000 --out runs/replay
```

`--symbols-file` accepts a plain-text roster. Choose each historical roster
using contemporaneous exchange listings or a frozen watchlist. A roster picked
from the day's later winners makes the result unsuitable for assessing an edge.
The capture is limited to the supplied symbols.

Replay accepts multiple session JSON files on the command line. Its comparison
matrix includes three selection methods and two entry setups. Use `--selection`
or `--setup` to focus the comparison. Useful sensitivity runs include
`--cost-bps 20`, `--latency-ms 1000`, and `--round-trips-per-five 3`.
Use a distinct output directory for each assumption set. A replay writes
summary, trade ledger, cash ledger, and a manifest with the input SHA-256
values and settings.

For an on-demand watchlist on the free plan, use
`capture --date YYYY-MM-DD --feed iex --scan-only` after the configured
scan cutoff. IEX premarket begins at 8:00 a.m. ET. SIP capture starts at
4:00 a.m. ET and uses a 16-minute buffer for the free historical entitlement.
All history and quotes in a capture come from the same feed. The exchange
calendar provides the session open, close, and settlement date. Capture requests
quotes only for the union of the top `--shortlist` candidates in each
selection method. A larger replay `--top` can require a new capture.

## What the watchlist measures

- Five minutes before the exchange open, the scanner freezes completed
  premarket bars, compares the last price with the previous daily close from
  the same feed, and records the high, traded volume, and estimated dollar
  volume. The default gates are $1–$20, a 5% gap, and $100,000 of premarket
  dollar volume.
- Relative volume compares the cumulative premarket volume with the median
  over the same time window of prior sessions. The scanner requires ten
  observations and records an unknown ratio for a missing or zero baseline.
  The RVOL variants require at least 2×. Known same-day splits are excluded.
- The gap-only list ranks by gap percentage. RVOL ranks by relative volume
  with gap as a tie-break. Catalyst + RVOL also requires a fresh headline
  with a material-event keyword. Financing or split keywords take priority
  for review. Repeated normalized headlines are deduplicated, and both the
  creation and last update must precede the decision. A keyword is an
  annotation and an experimental filter. Coverage is limited to the sourced
  feed, and float is unavailable in this version.
- All names retain eligibility and exclusion reasons. A wide, stale,
  one-sided, or unavailable premarket quote stays visible as context.
  Order eligibility is tested against quotes at entry.

## What the replay assumes

- A five-minute opening range requires all five completed bars. The
  premarket-high setup uses the high frozen with the watchlist. A completed
  minute must close above the trigger during the entry window. Default
  orders arrive 250 ms later and evaluate distinct quote events strictly
  after arrival. Historical minute bars can contain later vendor revisions.
- Buys use the ask of a valid two-sided quote and are limited by displayed
  size, a price-chase threshold, and spread divided by stop distance. The
  remaining buy quantity is canceled. A sell uses the bid after its own
  latency and may require multiple bid updates. An unchanged bid and size
  cannot be filled repeatedly. Alpaca quote sizes are shares from November
  3, 2025. Older captures default to 100 shares per lot, with an override
  for different listing conventions.
- The stop is measured from the entry offer. A time exit is submitted
  60 minutes after the open by default. Missing quotes can leave an
  unresolved position, which blocks further simulated entries and leaves
  portfolio return unreported. Displayed depth is a proxy for an executable
  fill. Replay does not reconstruct order queues, venue routing, market
  impact, or exchange halt messages. Fills cross the spread. Modeled costs apply to each executed side.
- The simulation uses whole shares and settled cash, with one pending order
  or position at a time. Proceeds settle on the calendar's settlement date.
  If the feed omits that date, the capture records a next-trading-session
  assumption. The optional five-session cap is a scenario parameter. The
  report withholds a return when there are unresolved positions, insufficient
  execution captures, or an unavailable news feed in a catalyst run.

These choices follow the public [Stocks in Play workflow](https://www.smbtraining.com/blog/how-to-day-trade),
[execution reconciliation practice](https://lineyourownpockets.com/episodes/sizing-up-a-trading-strategy/transcript),
and [market replay research](https://www.man.com/insights/simulating-financial-markets).
The [opening-range study](https://alexandria.unisg.ch/server/api/core/bitstreams/3c2989c4-688d-4d78-8a71-f02690990d51/content)
used a different universe and order model. The [Alpaca feed and timing
constraints](https://docs.alpaca.markets/us/docs/market-data-faq) and
[quote-size schema](https://docs.alpaca.markets/us/reference/stockquotesingle-1)
document the inputs used here.