# ADR-002: One ranking statistic in R units; no weighted score

**Status:** Accepted — the single most important decision in this design

## Context

The original brief proposed ranking opportunities by a weighted sum:

```text
FinalScore = w_technical * TechnicalScore + w_regime * RegimeScore
           + w_risk * RiskScore + w_sentiment * SentimentScore
           + w_context * ContextScore
```

alongside a separate expected-value formulation. The two are incompatible.

Problems, in order of severity:

1. **Dimensionally incoherent.** Expected value has units of money or risk;
   sentiment score is dimensionless in `[-1, 1]`. Their weighted sum has no unit,
   so it cannot be compared against a cost — and cost comparison is the entire
   purpose of a selective system.
2. **12–20 free parameters** (the weights plus each sub-score's normalisation),
   all fitted against the same history, all interacting. With 100 assets and
   6 strategies, the reachable configuration count is effectively unbounded.
3. **Double counting.** Regime quality, volatility quality, and technical signal
   strength largely measure recent realised volatility from three angles. Adding
   them over-weights what they share, producing a volatility bet in disguise.
4. **No probabilistic meaning.** "Final score 0.87" predicts nothing, so the
   trading threshold must be chosen by looking at the backtest.

## Decision

**Exactly one ranking statistic, with a unit, derived rather than tuned:**

```text
ev_net_r = ev_r_lcb - cost_r          [units: R]
```

Trade if `ev_net_r >= min_ev_net_r`. Rank by `ev_net_r / expected_bars_held`.

Every other input enters as exactly one of three mechanisms:

| Mechanism | Nature | Examples |
|---|---|---|
| Gate | binary, pre-scoring | liquidity, spread, data quality, regime, warm-up, minimum samples |
| Conditioning | selects the bin whose history applies | strategy, direction, volatility bucket |
| Size multiplier | multiplicative, `(0, 1]`, one-sided | sentiment penalty, heat taper, thin-liquidity discount |

**Weight count: zero.**

## Consequences

- Free parameters drop from 12–20 to 0 in the scoring layer.
- Every quantity in the decision is comparable, because every quantity is in R.
- The threshold is derived from cost-estimation uncertainty rather than fitted.
- The system becomes falsifiable: predicted `ev_net_r` versus realised R is a
  calibration plot, and a flat line means no information. Under a weighted score
  no equivalent check exists.
- Cost: some genuinely informative inputs can only enter as a gate or a
  multiplier, which is coarser than a weight. Accepted — a coarse mechanism that
  cannot be abused beats a precise one that will be.
- Cost: the mapping from "sentiment is positive" to "score improves" is no longer
  available. That is deliberate; see ADR-013.

## Alternatives rejected

**Weighted sum with weights fitted by walk-forward optimisation.** Still
dimensionally incoherent; the fitting would just be automated. Would also consume
the entire trial budget.

**Logistic regression producing a probability, then EV from that.** Defensible,
and it is the M6 upgrade path. Rejected for v1 because it requires a fitting step
with leakage risk before any baseline exists, and because a broken pipeline plus a
model is unattributable.

**Rank by expectancy alone, ignoring sample size.** Would systematically prefer
rare setup types with lucky small samples — the highest-variance bins would always
top the ranking. The lower confidence bound is what prevents this.

**Rank by `ev_net_r` without the holding-period normalisation.** Would prefer a
0.20 R trade held 30 bars over a 0.12 R trade held 8 bars, despite the second
earning more than twice as much per unit of the genuinely scarce resource.

## Revisit trigger

M6, and only if the M3 baseline passes its criteria. The replacement is a
calibrated probability model behind the `EdgeEstimator` Protocol — still producing
one number in R, never a weighted score.
