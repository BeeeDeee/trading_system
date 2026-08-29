# ADR-004: Regime is a causal feature plus a gate, not a pipeline stage

**Status:** Accepted — reaffirms the previous project's ADR-004

## Context

The brief places a "Regime Detector" as a pipeline stage between features and
strategies, emitting TREND / RANGE / CHOP plus direction, volatility state, and a
confidence in `[0, 1]`.

Discrete regime labels are among the least reliable objects in quantitative
finance. They are unstable at threshold boundaries, they lag by construction, and
carelessly implemented (any full-series fit — HMM, clustering, changepoint
detection) they *repaint*: today's new bar changes yesterday's label, which
invalidates every backtest that used it.

Making the label a *pipeline stage* means every downstream component takes a hard
dependency on a fragile categorical.

The previous iteration of this project already resolved this correctly. Its
ADR-004 stated: *"Market regime classification is owned by the Feature Engine as
derived features. It is an optional strategy input, not a pipeline stage and not a
trading prerequisite."* The new brief regresses on that.

## Decision

Regime lives in the feature layer as:

1. Two continuous causal features — `efficiency_ratio_20` (decision timeframe) and
   `efficiency_ratio_ctx` (context timeframe), Kaufman's Efficiency Ratio.
2. One derived label from a **pure, stateless function with five thresholds**
   (`classify_regime`), available as `FeatureRow.regime`.

The label has exactly one use in decisions: **gating which strategies may fire**,
via `Strategy.allowed_regimes`. It is also recorded on every `Setup` and
`DecisionRecord` for research, and it is available as a bin dimension if it ever
earns one.

There is no regime component, no state machine, no separate detector, and no
regime confidence score.

## Consequences

- No repainting is possible: every input is a backward-looking window.
- Five thresholds total, versus an HMM's state count, covariance structure,
  initialisation, and refit schedule — all of which would be fitted on the same
  data used to evaluate the strategy.
- **Label flapping is tolerable**, because a flap costs a missed entry, never a
  wrong-way position. This is precisely why the label must not be used for
  anything beyond gating. Using it for sizing or scoring would make instability
  expensive.
- CHOP is the default: anything not clearly trending or clearly ranging blocks
  both v1 strategies. The brief's "CHOP → usually NO TRADE" falls out for free.
- Multi-timeframe agreement (`er_ctx >= er_ctx_trend_min`) delivers what the brief
  wanted from multi-timeframe analysis at a cost of one threshold rather than a
  second decision timeframe.

## Alternatives rejected

**Hidden Markov model.** Repaints when fitted on the full series; unstable at
boundaries when fitted walk-forward, at exactly the transitions that matter. Adds
four categories of hyperparameter to a system whose whole discipline is bounding
the parameter count.

**ADX as the trend measure.** Three interacting smoothing parameters and a
definition that varies between implementations. ER has one parameter, is bounded
in `[0, 1]` with no normalisation, and measures directional progress per unit of
path length — which is literally the ratio of what you want to what stops you out.

**Regime as a continuous score entering the ranking.** Would reintroduce the
weighted-score problem from ADR-002.

**Volatility as a separate regime dimension.** It is `vol_bucket`, and it is a
*bin dimension* rather than a gate, which is the more useful position — it
conditions the statistics instead of blocking trades.

## Revisit trigger

If the M3 stability analysis shows performance concentrated in one regime while
the gate claims to admit several, revisit the thresholds — once, as a documented
trial, not as a sweep.
