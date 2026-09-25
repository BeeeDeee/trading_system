# ADR-003: Binned empirical walk-forward edge estimator with a lower confidence bound

**Status:** Accepted

## Context

Ranking by expected value requires an estimate of expected value. The candidates
were: a hand-specified formula, a fitted model, or measured historical outcomes.

## Decision

**Measure it.** For each setup, look up the empirical distribution of realised R
from historically resolved setups in the same bin, and use a one-sided lower
confidence bound on its mean:

```python
sample   = resolved[(bin == k) & (resolution_ts < as_of)]   # strictly before
mean_r   = sample.realised_r_gross.mean()
std_r    = sample.realised_r_gross.std(ddof=1)
ev_r_lcb = mean_r - z * std_r / sqrt(len(sample))           # z = 1.28 (90%)
```

Bins: `(strategy_id, direction, vol_bucket)` — 12 in v1. Assets pooled.
Statistics precomputed on a monthly `as_of` grid, looked up with backward
rounding. Bins with `n < 100` are unusable and reject the candidate.

## Consequences

- **The estimator is walk-forward by construction.** There is no fitting step, so
  there is nothing to leak. The development period is already an honest
  walk-forward of the estimator; the holdout exists only to guard the *other*
  overfittable things (thresholds, strategy parameters, gates, bins).
- **Small-sample and noise penalties are automatic.** The `1/sqrt(n)` and `std_r`
  terms replace the brief's separate "confidence" and "volatility quality" score
  dimensions with no weights.
- **Timeouts and partial moves are handled for free**, because the estimate uses
  the realised distribution rather than a three-point win/loss/cost
  approximation.
- **Cold start is real:** the first 12–18 months of any run produce almost no
  trades. This is correct and must not be "fixed"; seeding bins from full-history
  statistics leaks the answer directly into the ranking.
- **Upgrade path is clean.** At M6 the estimator goes behind a Protocol and an ML
  variant returns the same `BinStats` shape. Nothing downstream changes.
- **Heavy-tail caveat:** `std_r / sqrt(n)` assumes approximate normality of the
  sampling distribution, which is optimistic below about `n = 500`. Mitigated by
  a bootstrap LCB option, which is mandatory for holdout runs.

## Alternatives rejected

**A hand-specified probability formula** (for example, "ER above 0.4 implies a 55%
win rate"). Unfalsifiable and tuned. This is what the brief's "Probability Model"
box would have become in practice.

**Logistic regression or gradient boosting.** Deferred to M6. Fitting a model
before the pipeline is validated makes any result unattributable, and if the
baseline has no edge the model will find the same absence plus an overfit.

**Per-asset statistics.** Sample size collapses. Per-asset separation requires
passing the six-part test in
[`07-EDGE_AND_SCORING.md §9`](../07-EDGE_AND_SCORING.md#9-should-an-asset-get-its-own-calibration),
which crypto sample sizes will almost never satisfy — the correct outcome.

**Recomputing statistics at every decision bar.** About 300× more expensive for
immaterial precision gain, and it introduces a boundary-error risk at exactly the
place where a leak would be most damaging.

**Estimating `p(win)` and reward separately, then combining.** Requires special
handling of timeouts and re-introduces a three-point approximation. `mean_r`
captures everything the combination was approximating.

## Revisit trigger

M6, gated on the M3 criteria in
[`07-EDGE_AND_SCORING.md §11`](../07-EDGE_AND_SCORING.md#11-when-ml-is-allowed-to-replace-the-estimator).
