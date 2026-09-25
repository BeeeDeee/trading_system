# ADR-014: Strategies do not report confidence

**Status:** Accepted

## Context

The brief specifies that each strategy should answer:

```text
Can I trade this regime?
Do I see a valid setup?
What is the expected direction?
What is the expected reward?
What is the expected risk?
How confident am I?
```

Five of those six are fine. The sixth is not.

A confidence number that a strategy asserts about itself is not evidence. It is a
free parameter with no ground truth, expressed as code:

```python
# The shape this always takes in practice
confidence = 0.5
if row.ema_spread_atr > 0.5:
    confidence += 0.2
if row.efficiency_ratio_20 > 0.4:
    confidence += 0.15
if row.atr_percentile_1y < 0.5:
    confidence += 0.1
return min(confidence, 1.0)
```

Four magic numbers, unfalsifiable, and they will be adjusted until the backtest
improves. Multiply that by the number of strategies and it is a larger free
parameter space than the entire rest of the system.

Worse, self-reported confidence is *correlated with the setup conditions* the edge
estimator already conditions on, so it double-counts the same information — the
same defect as the brief's weighted score.

## Decision

`Strategy.detect` returns `Setup | None`. A `Setup` contains geometry only:
direction, reference price, stop, target, max holding period, and a
`trigger_note` label for analysis.

There is **no** `confidence` field on `Setup` and no confidence method on
`Strategy`.

Confidence is **measured**, in `scoring/edge.py`, as the gap between the point
estimate and its lower confidence bound:

```text
ev_r_point = mean_r                         # what similar setups delivered
ev_r_lcb   = mean_r - z * std_r / sqrt(n)   # what we can defend

confidence  ~  the size of (ev_r_point - ev_r_lcb)
```

Narrow gap: many samples, consistent outcomes — high confidence. Wide gap: few
samples or dispersed outcomes — low confidence. Computed from data, no parameters.

## Consequences

- Strategies become genuinely simple: a boolean condition and four price levels.
  `donchian_breakout_v1` is about 30 lines.
- Strategies are trivially unit-testable — pure functions of one `FeatureRow`.
- The free parameter count stays bounded: 5 parameters for strategy A, 4 for
  strategy B, and none for confidence.
- Confidence becomes falsifiable. The calibration plot in
  [`07-EDGE_AND_SCORING.md §10.1`](../07-EDGE_AND_SCORING.md#10-required-m3-diagnostics)
  tests it directly. A self-reported confidence cannot be tested against anything.
- The system remains honest about the cold-start problem: a new strategy has no
  history, so it has no defensible confidence, so it does not trade until it has
  one. A self-reported confidence would let it trade from day one on an assertion.
- Cost: genuine strategy-author intuition about setup quality is discarded. If
  that intuition is real, it should be expressible as a *condition* — either the
  setup fires or it does not — or as a bin dimension whose value the data can
  confirm. Both routes are available and both are testable.

## Alternatives rejected

**Confidence as an extra `Setup` field, used as a size multiplier.** Better than
using it in the ranking, but still unfalsifiable and still tuned.

**Confidence as a bin dimension.** Legitimate: discretise the strategy's
confidence into buckets and let the empirical statistics decide whether it
predicts anything. Rejected for v1 on sample size — it would take 12 bins to 36.
This is the correct route if a strong intuition needs testing later.

**Multiple strategy variants instead of a confidence score** (for example
`donchian_breakout_strict_v1` and `donchian_breakout_loose_v1`). This is
*equivalent* to a discretised confidence and gets separate bin statistics, so the
data decides. Genuinely available if needed, at the cost of two trials and a
sample-size split.

## Revisit trigger

M4, and only as a bin dimension with pre-registered reasoning — never as a
self-reported number.
