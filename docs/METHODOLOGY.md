# Methodology specification

## 1. Objective and scope

This version ranks liquid U.S.-listed common stocks for further human or
AI-assisted research over a prospective holding period of roughly one to two
months. It is a cross-sectional candidate screen. It does not decide position
size, execution, exits, portfolio exposure, or whether the market regime is
suitable for momentum.

The methodology is versioned as `1.0.0`. Its parameters were chosen before any
project backtest and are intentionally few. Changes to definitions, thresholds,
or weights require a new methodology version and out-of-sample evaluation.

## 2. Evidence used

The design emphasizes results with clear definitions and long research histories:

1. Jegadeesh and Titman document continuation following three-to-twelve-month
   formation periods. This establishes medium-horizon cross-sectional momentum:
   [Returns to Buying Winners and Selling Losers (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x).
2. Novy-Marx finds that months 12 through 7 carry particularly strong predictive
   content, motivating a separate intermediate-horizon leg rather than treating
   all past months as interchangeable:
   [Is Momentum Really Momentum? (2012)](https://doi.org/10.1016/j.jfineco.2011.05.003).
3. George and Hwang find that nearness to the 52-week high contains information
   beyond conventional past returns:
   [The 52-Week High and Momentum Investing (2004)](https://doi.org/10.1111/j.1540-6261.2004.00695.x).
4. Moskowitz and Grinblatt document strong momentum in industry components of
   stock returns, supporting an explicit peer-group term:
   [Do Industries Explain Momentum? (1999)](https://doi.org/10.1111/0022-1082.00146).
   The initial engine uses the broader provider-supplied sector field to avoid
   unstable, sparse industry groups. This is a conservative implementation
   choice, not a claim that sectors and industries are identical.
5. Chan, Jegadeesh, and Lakonishok find that past returns and earnings surprise
   independently predict subsequent drift. They also document sluggish analyst
   response:
   [Momentum Strategies (1996)](https://doi.org/10.1111/j.1540-6261.1996.tb05222.x).
   Version 1 implements the earnings-surprise leg, which can be reconstructed
   from originally reported point-in-time EPS. Analyst revision history is left
   for a later provider capability because backfilled consensus series are a
   common leakage risk.
6. Ang, Hodrick, Xing, and Zhang document a negative cross-sectional relation
   between high idiosyncratic volatility and future returns:
   [The Cross-Section of Volatility and Expected Returns (2006)](https://doi.org/10.3386/w10852).
   This engine uses simpler realized volatility as a small quality/risk term; it
   does not claim that the two measures are equivalent.
7. Lee and Swaminathan show that past trading volume conditions the magnitude
   and persistence of price momentum, with high- and low-volume winners behaving
   differently over time:
   [Price Momentum and Trading Volume (2000)](https://doi.org/10.1111/0022-1082.00280).
   Because the evidence is conditional, volume is not assigned a monotonic alpha
   score. Median dollar volume and zero-volume frequency instead enforce
   tradability.
8. Korajczyk and Sadka show why price impact and liquidity matter to implementable
   momentum returns:
   [Are Momentum Profits Robust to Trading Costs? (2004)](https://doi.org/10.1111/j.1540-6261.2004.00656.x).
9. Momentum can suffer severe state-dependent crashes, especially around market
   rebounds following stressed declines:
   [Daniel and Moskowitz, Momentum Crashes (2016)](https://doi.org/10.1016/j.jfineco.2015.12.002).
   A high rank is therefore not a complete risk decision.

The research process also treats bias control as part of the model. Shumway
documents economically large omitted delisting returns
([The Delisting Bias in CRSP Data](https://doi.org/10.1111/j.1540-6261.1997.tb03818.x)),
while Harvey, Liu, and Zhu show why a large factor search needs a much higher
evidentiary hurdle
([…and the Cross-Section of Expected Returns](https://doi.org/10.1093/rfs/hhv059)).
Those concerns motivate point-in-time membership, explicit delisting requirements
for future return evaluation, and the absence of parameter optimization in this
initial engine.

## 3. Decision time and information set

Every run has one timezone-aware `as_of` timestamp. An observation can influence
the run only when its `available_at <= as_of`. Timestamps are compared in UTC;
session and security effective dates are evaluated in the configured U.S. market
timezone (`America/New_York` by default). Additional event-time rules apply:

- a daily price row must have `session_date <= as_of`;
- an earnings row must have `announced_at <= as_of`;
- the latest security-master version known at `as_of` is selected before its
  effective dates are evaluated;
- a revised row supersedes an earlier version only when that revision was already
  available at `as_of`.

The engine repeats these checks even if the provider claims to have filtered the
data. Future rows are discarded and counted in the manifest. Appending data that
became available after `as_of` cannot alter the candidate table or input hash.

For historical replay, providers must preserve historical versions, original
announcement availability, delisted securities, and the contemporaneous security
master. Using today's constituents or today's fully backfilled fundamentals is a
data-contract violation.

## 4. Eligibility screen

The default universe includes primary, USD-denominated records that the provider
classifies as common-stock listings of U.S. issuers on NYSE, Nasdaq, NYSE American,
or NYSE Arca. ETFs, ETNs, preferreds, funds, ADRs, limited partnerships, and
secondary listings are rejected when the provider taxonomy identifies them as
such. SPAC common equity is not separately excluded in version 1, so provider
classification quality remains important.

At the formation timestamp a security must also satisfy:

- active according to the latest known security-master version;
- point-in-time market capitalization of at least USD 300 million;
- latest raw close of at least USD 5;
- median raw-close times volume of at least USD 5 million over 20 market sessions;
- nonzero volume on at least 95% of those 20 sessions;
- a current bar on the shared latest market session;
- all required momentum anchors and all recent 64 observations;
- at least 98% coverage of the required 253-session price window.

These thresholds are transparent engineering priors intended to avoid the least
tradable tail. They are not universal constants and were not selected by maximizing
past returns. Historical validation must model spreads, impact, commissions,
turnover, taxes where relevant, and delisting returns; the screen itself does not.

## 5. Raw factors

Let `TRI(t-k)` be the point-in-time total-return index at the shared market session
lag `k`, and let `P*(t)` be split-adjusted, dividend-unadjusted close.

### 5.1 Intermediate momentum: 20%

```text
MOM_12_7 = TRI(t-126) / TRI(t-252) - 1
```

This isolates the older half of the conventional momentum formation year.

### 5.2 Recent medium-horizon momentum: 15%

```text
MOM_6_1 = TRI(t-21) / TRI(t-126) - 1
```

The most recent 21 market sessions are skipped to avoid deliberately loading on
the short-term reversal month. The two momentum windows do not overlap.

### 5.3 52-week-high proximity: 15%

```text
HIGH_52W = P*(t) / max(P*(t-251), ..., P*(t))
```

The value is at most one. A consistent split-adjusted price series is required;
a dividend-adjusted total-return index is not substituted for the historical high.

### 5.4 Sector momentum: 15%

For stock `i`, first define:

```text
PEER_BASE_i = 0.5 * MOM_12_7_i + 0.5 * MOM_6_1_i
```

The factor is the current-point-in-time-market-cap-weighted mean `PEER_BASE` of
other eligible stocks in the same provider-supplied sector. Stock `i` is excluded
from its own sector value. A sector needs at least five eligible members. This
leave-one-out rule prevents a large stock's own return from mechanically feeding
back into its score.

### 5.5 Standardized unexpected earnings: 20%

For originally reported diluted quarterly EPS `E_q`, define the seasonal change:

```text
D_q = E_q - E_(q-4)
SUE_q = (D_q - mean(D_(q-8), ..., D_(q-1)))
        / sample_std(D_(q-8), ..., D_(q-1))
```

The engine requires 13 consecutive fiscal-quarter observations, a nonzero baseline
standard deviation, and a latest announcement no older than 120 calendar days.
Negative EPS is valid. Missing, nonconsecutive, stale, or constant histories do not
receive an invented value.

### 5.6 Low realized volatility: 15%

```text
VOL_63 = sample_std(diff(log(TRI)), last 63 returns) * sqrt(252)
LOW_VOL = -VOL_63
```

The sign is reversed so that higher raw factor values are always preferable. This
term is both a modest defensive preference and a guard against letting the most
explosive stocks dominate a momentum-only screen.

## 6. Normalization, missing values, and score

Each raw factor is ranked only among eligible stocks for which that factor is
available. With average rank `r` among `n` observations:

```text
normalized = 2 * (r - 1) / (n - 1) - 1
```

This maps the cross-section to `[-1, 1]`, uses average ranks for ties, and is robust
to extreme raw magnitudes. For the return factors, this cross-sectional step is the
relative-strength comparison; subtracting the same market return from every stock
would not change their ranks. An entirely tied factor maps to zero. At least three
observations are required by default.

The final score and contribution are:

```text
contribution_f = 100 * weight_f * normalized_f
score = sum(contribution_f)
```

The score therefore lies in `[-100, 100]` when all factors are observed. Missing
factors contribute exactly zero, weights are not redistributed, and
`score_coverage` reports the sum of observed factor weights. The default requires
coverage of at least 0.65, which corresponds to both price-momentum windows,
52-week proximity, and volatility. Missing earnings or a sparse sector is visible
rather than silently imputed or rewarded.

Ranks sort by descending score and then ascending symbol. Contributions sum to the
reported score; there is no opaque model state.

## 7. Fixed weights and non-optimization policy

The factor-family allocation is:

| Family | Components | Weight |
|---|---|---:|
| Stock price state | 12-7 momentum, 6-1 momentum, 52-week high | 50% |
| Peer context | sector momentum | 15% |
| Fundamental news | earnings SUE | 20% |
| Risk quality | low realized volatility | 15% |

These are coarse ex ante allocations: price information is central, independent
earnings news is material, and peer/risk terms are meaningful but smaller. No
historical optimizer was used. A future research process should preregister any
candidate change, keep a trial log, use walk-forward evaluation, and reserve a
truly untouched test period. Bailey et al. provide a formal treatment of selection
bias from trying many backtests:
[The Probability of Backtest Overfitting](https://doi.org/10.21314/JCF.2016.322).

## 8. Known limitations

- Sector classifications and market capitalization must themselves be
  point-in-time; many inexpensive feeds backfill current classifications.
- Correct total-return histories require point-in-time corporate-action handling.
- Reported EPS definitions can vary by provider and firm. The provider must keep a
  consistent diluted-EPS basis and revision history.
- The engine has no borrow, spread, impact, tax, portfolio, or turnover model.
- It has no market-regime overlay, despite documented momentum crash risk.
- A rank is comparative, not a calibrated probability or expected return.
- Factor evidence may weaken after publication and varies across subperiods.

These limitations are reasons to build replay and execution layers carefully, not
reasons to hide complexity inside the initial ranker.
