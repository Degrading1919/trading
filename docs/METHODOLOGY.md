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

## 9. Adaptive probabilistic layer

### 9.1 Separation from the fixed baseline

The fixed methodology above remains version `1.0.0` and is not reimplemented by
the adaptive code. At every decision timestamp, `AdaptiveRankingEngine` first
runs the existing `ScreeningEngine`. Its eligible universe and six normalized
features are the complete current prediction input. The fixed score remains in
the output as `baseline_score` and `baseline_rank`.

The adaptive layer has its own methodology version, also initially `1.0.0`, and
answers a different question: conditional on the fixed baseline information set,
what is the estimated probability that a stock will outperform the median stock
in that same investable formation cohort over the next 42 completed market
sessions? The probability is a research estimate, not a promise of profit and not
a portfolio weight.

This boundary creates three distinct layers:

1. the immutable, literature-motivated fixed baseline;
2. a purely numerical adaptive probability model trained on frozen baseline
   artifacts and realized outcomes;
3. future AI-assisted research or judgment that may consume the two artifacts but
   cannot alter features, labels, coefficients, probabilities, or rank arithmetic.

Gu, Kelly, and Xiu show that statistical learning can help forecast the
cross-section of equity returns and that momentum, liquidity, and volatility
remain important predictors. They also preserve temporal ordering in validation:
[Empirical Asset Pricing via Machine Learning (2020)](https://doi.org/10.1093/rfs/hhaa009).
Haddad, Kozak, and Santosh provide evidence that factor expected returns can vary
over time, while also emphasizing the need to discipline a large factor-timing
search:
[Factor Timing (2020)](https://doi.org/10.3386/w26708).
These findings motivate testing time-varying weights; they do not establish that
this particular implementation will outperform the fixed baseline.

### 9.2 Frozen training cohort and event label

A training cohort begins with a completed fixed-baseline run at `formation_at`.
For every candidate, the cohort stores the six `normalized_*` values, baseline
methodology/configuration hashes, run fingerprint, reference session, and declared
formation-universe size. Those feature values are immutable.

After exactly 42 shared market sessions, a return pipeline supplies one finite
total return for every formation candidate:

```text
FORWARD_RETURN(i,t) = TRI(i,t+42) / TRI(i,t) - 1
```

The outcome pipeline must incorporate distributions, corporate actions, and any
delisting return. A cohort with even one missing outcome is rejected rather than
silently converted into a survivor-only sample. `horizon_sessions=42` is an ex
ante approximation to two trading months and is inside the requested one-to-two
month research horizon; it was not selected from this project's results.

Within each complete formation cohort, the binary event is:

```text
Y(i,t) = 1 if FORWARD_RETURN(i,t) > median_j(FORWARD_RETURN(j,t)); else 0
```

The strict comparison gives ties a zero label. A cross-sectional median target is
robust to extreme realized returns, remains meaningful across market regimes, and
aligns the probability with the ranking task. It is not the probability of a
positive absolute return or of beating a named market index.

### 9.3 Point-in-time and revision rules

Every materialized training row has `formation_at`, `label_end_at`, and
`available_at`. A row can train a model at decision timestamp `D` only when:

```text
formation_at < label_end_at <= available_at <= D
formation_at >= D - training_window_calendar_days
```

The adaptive schema filters these rules independently of the provider. It selects
the latest row version known at `D`. A revision may correct only
`forward_return`; it cannot rewrite the frozen baseline feature values, cohort
metadata, or baseline fingerprints. A revision that became available after `D`
cannot affect a replay at `D`.

All cohorts must match the current fixed baseline's methodology version and full
configuration hash. This prevents mixing features built under different filters,
factor definitions, or normalization rules. The training input hash is calculated
after point-in-time version selection and rolling-window filtering, so appending a
future row leaves the earlier fit and run fingerprint unchanged.

Point-in-time correctness ultimately depends on the provider retaining historical
universe membership, delisted securities, and original data versions. The schema
can reject incomplete or internally inconsistent artifacts; it cannot recover
facts a provider never preserved.

### 9.4 Initial probability model

The reference adaptive method is a pooled L2-regularized logistic regression:

```text
logit(P(Y(i,t)=1)) = intercept + sum_f(beta_f * X(i,t,f))
```

`X` contains only the six baseline normalized factors in their canonical order.
A missing normalized factor enters as zero, exactly matching its neutral
fixed-baseline contribution. No LLM, text embedding, generated market datum, or
post-formation feature enters the fit.

The fitted parameters minimize:

```text
weighted_mean_binary_log_loss + 0.5 * lambda * sum_f(beta_f^2)
```

The intercept is not penalized. Each formation cohort receives equal total
likelihood weight so changes in universe size do not cause a large cohort to
dominate the fit. L2 shrinkage reduces unstable coefficient magnitudes without
performing factor selection. Le Cessie and van Houwelingen show how ridge
estimators can improve logistic prediction when unrestricted estimates overfit:
[Ridge Estimators in Logistic Regression (1992)](https://doi.org/10.2307/2347628).

The default, preregistered engineering choices are:

| Setting | Default |
|---|---:|
| Forward horizon | 42 market sessions |
| Rolling formation window | 1,826 calendar days (about five years) |
| Minimum complete cohorts | 24 |
| Minimum observations | 500 |
| Minimum observations in each class | 25 |
| L2 penalty `lambda` | 0.10 |
| Maximum Newton iterations | 100 |
| Convergence tolerance | `1e-9` |

The window balances adaptation against the very low signal-to-noise ratio of
individual-stock returns. The numerical values are explicit priors, not the
winners of a historical grid search. They require sensitivity analysis and truly
out-of-sample evaluation before any use. A failed convergence or insufficient
sample fails the adaptive run; the engine does not substitute a hidden fallback.

### 9.5 Walk-forward operation and explainability

For each decision timestamp, the engine independently:

1. runs the fixed baseline at that timestamp;
2. selects only complete training rows available at that timestamp and inside the
   rolling window;
3. fits a fresh model without carrying fitted state forward;
4. predicts the current fixed-baseline candidate universe; and
5. sorts descending probability with symbol as the deterministic tie-break.

`run_walk_forward` requires strictly increasing decision timestamps and repeats
that process. Consequently, an outcome becoming known between two timestamps may
affect the later coefficients, but never the earlier model or rank.

For each stock, the output exposes the intercept, every additive feature
contribution in log-odds units, total log odds, probability, and both adaptive and
fixed ranks. The identity is exact:

```text
adaptive_log_odds = adaptive_base_log_odds
                    + sum_f(adaptive_contribution_f)
probability_outperformance = logistic(adaptive_log_odds)
```

The learned-weight artifact reports each coefficient, its odds ratios, its share
of total absolute coefficient magnitude, and the corresponding fixed baseline
weight. A coefficient may be negative: that is an auditable statement about the
recent training window, not a rewrite of the academic baseline.

`confidence_from_50pct = 2 * abs(probability - 0.5)` is provided as a compact
distance from an even-probability prediction. It is not a confidence interval,
standard error, or guarantee of calibration.

### 9.6 Validation and non-optimization policy

The manifest reports weighted in-sample Brier score and log loss only as fit
diagnostics, explicitly not as performance estimates. Brier introduced the
quadratic score for probability forecasts:
[Verification of Forecasts Expressed in Terms of Probability (1950)](https://doi.org/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2).
Probability calibration must be assessed on predictions that were genuinely
out of sample.

A research comparison should preregister decision dates and evaluate the fixed
and adaptive ranks over the same complete point-in-time universe. At minimum it
should report walk-forward Brier score, log loss, calibration by probability bin,
rank information coefficient, top-quantile return, turnover, capacity, costs,
drawdowns, and delisting-inclusive results. Overlapping 42-session outcomes are
serially and cross-sectionally dependent, so naive row-level standard errors are
not valid.

Alternative windows, penalties, labels, or model families are separate methods.
They must use the `ProbabilisticModel`/`BinaryOutcomeLabeler` boundaries, retain a
trial log, and be compared on untouched walk-forward periods. Repeatedly choosing
the historically best alternative would recreate the multiple-testing problem
the fixed baseline was designed to avoid.
