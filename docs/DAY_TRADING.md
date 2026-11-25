# Day trading research with dayrank

Dayrank records premarket watchlists and evaluates long-only opening breakouts
using the same session data for each comparison. It makes read-only requests to
Alpaca and writes local artifacts. All defaults are starting assumptions for
research.

Install from the repository root with `python -m pip install -e ".[dev]"`.
Supply paper API credentials through `APCA_API_KEY_ID` and
`APCA_API_SECRET_KEY` in your local environment, or pass
`--credential-target` to read a named Windows generic credential.

## Discover stocks from a dated directory

Freeze a directory published before the feed's premarket window. Without
`--archive-date`, the command downloads the current Nasdaq Trader directory and
records when it was retrieved. For a historical run, choose an archived
snapshot and keep the exported JSON. The archived file's Nasdaq creation
timestamp serves as its historical availability estimate. The scanner
requires a fresh directory from the previous trading date or later, available
before the scan window. The directory hash and source URL go into the run
manifest. [Nasdaq directory definitions](https://nasdaqtrader.com/Trader.aspx?id=SymbolDirDefs), [directory archive](https://github.com/supermodo/us-markets-timemachine)

```powershell
dayrank universe --archive-date 2026-11-22 --archive-ref a80aad7731337939fa5fda3c70f3f1480d0d9e3f --out runs/2026-11-23-universe.json
dayrank discover --date 2026-11-23 --universe runs/2026-11-23-universe.json --feed sip --out runs/2026-11-23-sip
dayrank discover --date 2026-11-23 --universe runs/2026-11-23-universe.json --feed sip --start-at 08:00 --out runs/2026-11-23-sip-08
dayrank discover --date 2026-11-23 --universe runs/2026-11-23-universe.json --feed iex --out runs/2026-11-23-iex
```

- `symbols.csv` retains every directory row, including listing exclusions,
  missing market data, and qualifying names. Eligibility starts with
  supported exchange-listed common-stock descriptions. The first-alert
  fields survive later fading or staleness at the cutoff.
- `alerts.csv` reports each symbol's first qualifying bar event for each
  selection rule. The scanner compares cumulative premarket volume to the
  same minute over prior sessions from the same feed. It requires a
  positive baseline and at least ten observations. Historical headlines
  are evaluated at bar events using vendor timestamps.
- `eligible-symbols.txt` feeds the cutoff roster into the existing
  `capture --symbols-file` command for separate execution research.
- `manifest.json` describes the window, directory, feed, gates, data
  availability, alert counts, and cutoff counts. The window starts at
  4:00 a.m. ET for SIP and 8:00 a.m. ET for IEX by default. Set
  `--start-at 08:00` on SIP for a same-window comparison.

Historical REST bars can include later revisions. Discovery models
availability as the end of each minute plus 35 seconds by default. Use
`--bar-lag-seconds` to test that assumption. The model also excludes prices
whose last minute ended more than two minutes before the decision.
Historical feed coverage and a first qualification count measure discovery.
Executable entries and returns require a separate quote-based replay.

To measure actual timing, freeze the current directory before premarket and
start `record` before the chosen feed window. The recorder saves original
minute bars and updates with local UTC receipt timestamps. Leave the recorder
running through the cutoff with the system clock synchronized. Afterwards,
attach the completed tape to `discover` with `--recording` and the same
directory, date, feed, and window. The replay applies each update as it was
received. Interrupted recordings leave a `.partial` file and the replay
requires a completion record. [Alpaca stock stream](https://docs.alpaca.markets/us/docs/real-time-stock-pricing-data), [Alpaca feed access](https://docs.alpaca.markets/us/docs/about-market-data-api), [IEX hours](https://www.iex.io/resources/trading/trading-hours-holidays)

```powershell
dayrank universe --out runs/universe.json
dayrank record --date YYYY-MM-DD --universe runs/universe.json --feed iex --out runs/iex.jsonl
dayrank discover --date YYYY-MM-DD --universe runs/universe.json --feed iex --recording runs/iex.jsonl --out runs/iex-recorded
```

## Sample the market without indicator rules

`snapshots` starts from a frozen exchange directory. Every structurally
eligible common stock gets a row at every requested time, including stocks
with no prior close, no minute bars, or a stale last print. The command
applies no gap, price, volume, or breakout condition to sample membership.
`listings.csv` accounts for every directory row and its eligibility reason.

```powershell
dayrank snapshots --date 2026-11-23 --universe runs/2026-11-23-universe.json --out runs/2026-11-23-market
dayrank snapshots --date 2026-11-23 --universe runs/2026-11-23-universe.json --feed iex --at 08:45 --at 09:25 --out runs/2026-11-23-iex-market
```

Defaults are 8:15, 8:45, 9:15, 9:25, 9:35, and 9:45 a.m. ET. Each row in
`snapshots.csv.gz` uses values observable by its decision time under the
historical bar lag. The 5-, 15-, and 30-minute feature windows measure activity,
recent price ranges, and distances to the high and estimated VWAP. Zero bar
counts indicate no observations in the window. Prices and price changes stay
blank when the necessary data is missing. Split effects may appear in changes
from the unadjusted prior close.

Relative strength compares the move from the same-feed prior close with
the median of other fresh stocks in the same prior-day price and dollar-volume
bands. The prior-day bands and peer coverage appear in the manifest. This
comparison uses up to two-minute-old bar observations and requires five peers.

The default quote sample is 100 tickers per session. It is chosen from the
entire eligible roster using a hash of the date and symbol, before examining
any price move. Use `--quote-sample-size 0` for a bar-only run or increase the
sample to extend coverage. Other symbols still get feature rows. Quotes are
the last event in the minute up to the decision time. Quote event time serves
as an availability estimate, and the output distinguishes unsampled symbols,
missing quotes, and invalid books.
[Alpaca historical bars](https://docs.alpaca.markets/us/reference/stockbars),
[Alpaca historical quotes](https://docs.alpaca.markets/us/reference/stockquotes-1)

`manifest.json` records data coverage, source roster, clocks, lag, and quote
sample design. Choose one output directory per dated feed run. The command
uses historical market data and the same SIP entitlement buffer as discovery.
Snapshot rows contain observations through the decision time.

## Detect unusual behavior

`dayrank anomalies` reads a target snapshot folder and earlier folders for
the same feed and sampling design. It scores each observation against up
to 30 prior sessions at the same Eastern clock time. Each metric needs
20 comparable prior observations by default. No LLM or Alpaca connection
is required for offline scoring.

```powershell
$history = Get-ChildItem runs/baseline -Directory | Select-Object -ExpandProperty FullName
dayrank anomalies --target runs/2026-11-23-market --history @history --out runs/2026-11-23-anomalies
```

- Price dislocations use the recent five-minute return. A current move
  that aligns with a sufficiently populated peer group is retained in the
  evidence and excluded from the stock alert list.
- Participation surges require unusually high dollar volume **and** trade
  count against the stock's own history. They can surface before a large
  price move. Market and price/liquidity peer-group alerts test changes in
  the median recent return against that group's earlier sessions.
- Raw prior-close gaps are provisional offline. Add `--verify-splits` and
  Alpaca credentials to check candidate symbols and exclude reported
  effective splits. Corporate actions may reach the feed with a delay.
  As-reported splits help diagnose historical anomalies. The feed does not
  prove that a split check would have returned the same result at alert time.
  [Alpaca corporate actions](https://docs.alpaca.markets/us/reference/corporateactions-1)

Scores use a median and median absolute deviation with a minimum scale
for thinly traded stocks. Each candidate also needs a small empirical
tail fraction in the historical sample. This fraction is a count of
similarly extreme past observations with a finite-sample adjustment. It
does not estimate trade profitability. Stale observations, isolated
prints, missing baselines, and quote coverage remain visible.

The command writes `alerts.csv`, `candidates.csv` with suppression reasons,
`audit.csv.gz` with every target stock snapshot, and `manifest.json` with the
baseline dates, settings, and coverage. It ranks up to 15 alerts per clock by
default and keeps the first alert per symbol, anomaly type, and direction
across the supplied target clocks. One symbol can use at most one alert slot
at the same clock. Peer groups use prior-day price and dollar-volume bands.

## Build an indicator review set

`indicators` reads one or more saved discovery folders. It accounts for the
full directory and fetches minute bars only for in-scope names that had a
previous close and a premarket print. Known split exclusions carry over from
discovery. The indicator census covers a 15-minute premarket rolling high,
the premarket high frozen at the discovery cutoff, and the five-minute
opening-range high. It records the first close through each level, the next
consecutive close holding that level when available, wick rejections, and
near misses. The current volume confirmation thresholds are research
hypotheses. All measurements use the ticker's elapsed time and only the bars
available at the event. Market open resets the rolling feature history.

```powershell
dayrank indicators runs/2026-11-23-sip runs/2026-11-24-sip --review-size 72 --out runs/indicator-study
dayrank indicators runs/indicator-study/tapes/2026-11-23-sip.json.gz --out runs/indicator-repeat
```

- `cases.csv` contains the full census with per-bar observations and
  trailing 5-, 15-, and 30-minute feature windows.
- `review.csv` contains a repeatable sample across dates, patterns, and case
  types with blank review labels. Add labels after inspecting each case.
- `forward.csv` measures observed close changes after 1, 3, and 5 minutes.
  Missing future candles remain blank. Keep this file out of inputs to feature
  development and blind review. The changes are price observations.
- `summary.json` reports coverage, rule definitions, outcome counts, and
  median three-minute follow-through. It is small enough for a scheduled
  process to pass to a model after the local computations finish.
- `tapes/` retains the exact minute-bar response and timestamps used in the
  study. Reprocessing these compressed files works without Alpaca credentials.

For a scheduled pre-open audit, freeze the current directory before the
feed's start, run `discover` with a dated output directory, and pass that
directory to `indicators --premarket-only` after the discovery cutoff.
The Python command runs without a model. Historical SIP requires the
existing 16-minute entitlement buffer, so an Alpaca Basic run before the
bell uses IEX. The indicator census currently models observation time using
the discovery bar lag. It does not consume live receipt tapes, NBBO quotes,
or venue-level executions.

## Run a historical session with a supplied roster

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
