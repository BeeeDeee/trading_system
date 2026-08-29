# ADR-007: Static cluster caps and index beta instead of a covariance matrix

**Status:** Accepted. Amended for equities by §"Equity instantiation" below —
the reasoning is unchanged, the cluster map and beta reference change.

## Context

The brief requests correlation, sector exposure, market beta, and factor exposure
in the portfolio engine, correctly observing that "BTC long, ETH long, SOL long"
is effectively one position.

Implementing that with an estimated covariance matrix over 100 crypto assets
produces an object that is mostly estimation error. With a 90-bar window and 100
assets there are 4,950 pairwise covariances estimated from 90 observations each.
The matrix is near-singular, needs shrinkage (another parameter), and — worst — it
will confidently report that two assets are uncorrelated shortly before they go to
zero together, because crypto correlations converge to 1.0 exactly when it matters.

## Context that makes the simplification safe

Crypto's correlation structure is, to first order, **one factor (BTC) plus a
handful of stable thematic clusters**. That structure can be written down by hand
and it changes on a timescale of years, not weeks.

## Decision

Three mechanisms, no matrix:

1. **Static cluster map in config** (`MAJOR`, `L1`, `L2`, `DEFI`, `MEME`, `AI`,
   `EXCH`, `OTHER`) with `max_cluster_risk_pct` (default 1.0%, half the total heat
   cap) and `max_positions_per_cluster` (default 2).
2. **Rolling 90-bar beta to BTC** as the single continuous market-exposure
   measure: `net_beta_exposure_pct = Σ sign · open_risk_pct · beta`, capped at
   1.5%.
3. **Portfolio heat** — total open risk as a fraction of equity — capped at 2.0%,
   with a taper starting at 60% utilisation.

`corr_bench_90` is computed and recorded for research but does not enter decisions
in v1.

## Equity instantiation

The three mechanisms are unchanged. Two inputs are re-bound
([ADR-015](015-equities-first.md)):

1. **Cluster map becomes GICS sector**, taken from the data vendor's
   classification, not hand-maintained: `ENERGY`, `MATERIALS`, `INDUSTRIALS`,
   `CONS_DISC`, `CONS_STAPLES`, `HEALTHCARE`, `FINANCIALS`, `INFO_TECH`,
   `COMM_SVCS`, `UTILITIES`, `REAL_ESTATE`, plus `ETF_BROAD`, `ETF_SECTOR`,
   `ETF_INTL`, `ETF_BOND`, `ETF_COMMODITY`, `OTHER`.

   **A sector ETF and its constituents are the same cluster.** `XLK` is mapped to
   `INFO_TECH`, not `ETF_SECTOR`, precisely so that holding `XLK` plus four
   semiconductor names counts against one cap rather than two. Broad-market,
   international, bond, and commodity ETFs get their own clusters because their
   factor exposure is genuinely different.

2. **Beta reference becomes `SPY`** rather than BTC. `beta_bench_90` is a rolling
   90-session regression of the symbol's returns on `SPY`'s.

   This matters far more in equities than it did in crypto, and in the opposite
   direction. In crypto, BTC beta was a concentration measure. In equities, market
   beta is *the* thing a long-only equity strategy accidentally becomes: any
   long-biased equity system backtested over 2009–2021 shows a positive Sharpe
   from beta alone, with no skill whatsoever. So `net_beta_exposure_pct` is both a
   risk cap and the central diagnostic of
   [`12-RESEARCH_PROTOCOL.md`](../12-RESEARCH_PROTOCOL.md) — a strategy whose
   returns are explained by SPY beta has found nothing.

**Point-in-time caveat, stated honestly:** GICS sector classifications are
themselves revised — companies get reclassified, and the Real Estate and
Communication Services sectors were created in 2016 and 2018. The system uses the
vendor's *current* classification for all history. For an 11-way partition used
only to enforce a two-position cap, the resulting bias is immaterial; using it to
compute sector returns as a signal would not be acceptable. This limitation is
recorded in every run's `run.log`.

## Consequences

- Simpler *and* more robust than the estimated version. No shrinkage parameter, no
  singularity handling, no regime-dependent recalibration.
- Risk-weighting (rather than notional-weighting) the beta exposure means a
  tight-stopped position is not treated as a large exposure when its actual
  downside is small.
- The cluster cap at half the heat cap directly answers the brief's example: three
  correlated crypto longs are capped at 1.0% total risk, not 1.2% across three
  positions pretending to be independent.
- **Honest limitation:** in a genuine market-wide selloff all clusters go to
  correlation 1.0 and the cluster cap does nothing. What protects you then is the
  heat cap and the drawdown breaker. The cluster map guards against *thematic*
  concentration, which is the common case; the heat cap guards the tail. Both are
  needed; neither substitutes for the other.
  - For equities there is a second, momentum-specific version of this: a
    **momentum crash** reverses the entire top decile at once, across sectors.
    Neither the cluster cap nor beta helps. See
    [ADR-019 §5](019-cross-sectional-momentum-primary.md).
- Cost (crypto): the map must be maintained as new symbols list. Config validation
  warns with the names of unmapped symbols, so it fails loudly rather than pooling
  them silently into `OTHER`. For equities the map comes from the vendor, so the
  same validation applies to symbols with a missing sector field.

## Alternatives rejected

**Estimated covariance with Ledoit–Wolf shrinkage.** Better than raw sample
covariance, still dominated by estimation error at 100 assets and 90
observations, and it adds a parameter plus a substantial amount of code to
maintain and test.

**Correlation-based clustering, refitted periodically.** Cluster membership would
flip between refits, making the caps non-deterministic across runs and making
"why was this trade rejected" unanswerable.

**Principal-component or factor-model exposure.** The first PC of crypto returns
*is* BTC beta, and the first PC of equity returns *is* market beta, to a good
approximation in both cases. Item 2 already captures it at a fraction of the
complexity.

**A full Fama-French / Barra factor decomposition for equities.** Genuinely
informative, and it belongs in *research reporting* rather than in the decision
path. M3 reports the strategy's loading on market, size, value, and momentum
factors as a diagnostic; the portfolio layer caps only beta and clusters.

**Pairwise correlation cap between candidate and held positions.** Tempting and
cheap, but pairwise 90-bar correlation is unstable enough that the cap would bind
arbitrarily. The cluster map encodes the same information with far less noise.

## Revisit trigger

More than 15 concurrent positions, at which point hand-maintained clusters become
a real burden and the estimation error is spread over enough positions to be
worth estimating. v1 caps at 6, so this will not trigger.
