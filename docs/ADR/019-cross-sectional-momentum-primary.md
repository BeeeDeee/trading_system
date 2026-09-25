# ADR-019: Cross-sectional momentum is the primary strategy; R generalises to a sizing unit

**Status:** Accepted

## Context

[`06-STRATEGIES.md`](../06-STRATEGIES.md) originally specified two strategies —
Donchian breakout and range fade — both expressed as a `Setup` with an entry, a
stop, and a target, resolved by triple-barrier labeling. That is a
futures-trader's framework, and it is the right shape for crypto.

It is not the shape with the strongest evidence in equities.

The best-documented cross-sectional anomaly in equities is momentum: rank all
names by trailing return, hold the top fraction, rebalance periodically. Thirty
years of out-of-sample literature, replicated across countries, market caps, and
asset classes, and it survived publication. [`06-STRATEGIES.md §6`](../06-STRATEGIES.md)
already flagged it as having the strongest evidence while fitting the architecture
worst. The equity pivot forces that tension to be resolved, and it must be
resolved in favour of the momentum strategy.

Three things make momentum awkward in the existing architecture:

1. **It is relative, not absolute.** A stock is not a buy because it rose 40%; it
   is a buy because it rose more than 90% of its peers. A `Strategy.detect()` sees
   one `FeatureRow` and cannot know a cross-sectional rank.
2. **It has no target.** The exit is time or rank, not a price level.
3. **Stops hurt it.** This is documented, not folklore: momentum returns are
   positively autocorrelated at the holding horizon, so a stop that exits on a
   drawdown systematically sells the positions about to recover. Yet the entire
   risk framework is denominated in R, which was defined as the stop distance.

## Decision

### 1. Cross-sectional rank becomes a feature, not a strategy concern

The feature layer sees the whole panel and the universe snapshot, so it can
legitimately compute cross-sectional statistics. Added to `FeatureRow`:

- `mom_126_xs_pct`, `mom_252_xs_pct` — percentile rank of trailing 126/252-session
  return **among symbols eligible at `t`**, in [0, 1].
- `vol_xs_pct` — percentile rank of realised volatility, same universe.

Strategies remain pure per-row functions. `xsec_momentum_v1.detect()` gates on
`mom_252_xs_pct >= 0.90`, which is a plain field read.

The rank must be computed over the eligible set **at `t`**, from data with
`close_time <= t`. This is enforced by `tests/unit/test_no_lookahead.py`, which
truncates the panel and asserts every cross-sectional feature is unchanged.

### 2. R generalises from "stop distance" to "volatility sizing unit"

**R is a sizing unit denominated in dollars of intended risk. It does not require
a stop order to exist.**

```
risk_per_unit = stop_atr * atr        # dollars per share
qty           = risk_capital_usd / risk_per_unit
1 R           = risk_capital_usd
```

This definition already produces the previous behaviour when a stop is placed at
`stop_atr * atr`, so nothing about the breakout strategy changes. It additionally
makes volatility-targeted sizing expressible without a stop, which is what
momentum needs. Every downstream consumer — `ev_r_lcb`, `cost_r`, portfolio heat,
calibration — is unaffected, because all of them only ever divide dollars by
`risk_capital_usd`.

### 3. `Setup.target_price` becomes optional; a disaster stop is still required

```python
stop_price:   float          # required — always. Disaster protection.
target_price: float | None   # None means "no profit target; exit on time or rank"
```

`xsec_momentum_v1` sets a **disaster stop at 5 ATR** and `target_price=None`. A 5
ATR stop on a daily bar is roughly a 9% adverse move for a typical large cap — far
enough out that it does not interfere with the momentum autocorrelation, close
enough to bound a single-name catastrophe. It is not a risk-management tool; it is
a fraud-and-fat-tail circuit breaker.

Triple-barrier labeling handles `target_price=None` by evaluating only the stop
and time barriers. `realised_r_gross` is then computed from the actual exit price
in either case, unchanged.

### 4. Strategy roster for v1

| Strategy | Shape | Regime | Hold | Exit | Role |
|---|---|---|---|---|---|
| `xsec_momentum_v1` | rank-gated, no target | market RISK_ON | 21 sessions | time, or rank falls below 0.70 | **Primary.** The documented anomaly |
| `donchian_breakout_v1` | bracket, target at 2.5 R | per-symbol TREND_UP/DOWN | ≤ 40 sessions | stop / target / time | Secondary. Retained as a contrast |

`range_fade_v1` is **dropped from v1**. Daily-bar equity mean reversion at the
top-1000 liquidity tier is where the bid-ask-bounce illusion lives, it is the most
crowded retail strategy in existence, and it needs intraday data to be modelled
honestly. Available as an M6 experiment against its own trial budget.

### 5. Momentum-specific portfolio treatment

Momentum's failure mode is not a single bad trade; it is a **momentum crash** —
a violent reversal in which the entire top decile falls together (April 2009,
January 2021, and the classic 1932 and 2009 episodes in the literature). Position
sizing cannot help, because every position fails at once.

Two defences, both already in the architecture:

- **Market regime gate.** `xsec_momentum_v1` is only active in market RISK_ON.
  Momentum crashes cluster in the first weeks of a sharp recovery from a market
  low, which RISK_OFF and NEUTRAL exclude.
- **Cluster caps by GICS sector.** A crash concentrated in one sector is bounded.

Neither is sufficient. M3 explicitly reports performance in the six worst known
momentum-crash windows, and a strategy that loses more than 15% in any of them
does not proceed regardless of its aggregate metrics
([`12-RESEARCH_PROTOCOL.md`](../12-RESEARCH_PROTOCOL.md)).

## Consequences

- The `ev_net_r` ranking statistic and the entire scoring, cost, portfolio, and
  research machinery are unchanged. This was the point of making R a unit.
- Momentum trades resolve mostly by **TIME**, so the bins will be dominated by
  `SetupOutcome.TIME` rather than the STOP/TARGET mix crypto expected. `ev_r_lcb`
  handles this without modification, but the `mean_bars_held` used for cost
  scaling will be nearly constant at 21, which is fine.
- Two strategies with different shapes exercise the abstraction properly. If
  `Setup` cannot express both cleanly, the abstraction is wrong and it is better to
  find that out in M2 than in M5.
- Cross-sectional features couple the feature layer to the universe snapshot.
  `compute_features` takes the snapshot as a parameter; it does not import the
  universe module. Layering is preserved.
- **Turnover rises.** Rank-based exits at a monthly cadence with a 1,000-name
  universe generate more churn than bracket exits. At 2.3 bps per side this is
  affordable, which it would not have been in crypto — this decision is downstream
  of ADR-015.

## Alternatives rejected

**Keep breakout and range fade; add momentum in M6.** Spends the M3 trial budget
on the two shapes with the weakest equity evidence, and reaches the strongest one
only after the holdout is partly consumed. Backwards.

**Replace the whole architecture with a monthly rank-and-hold rebalancer.** Would
be simpler and is closer to how the anomaly is published. Rejected because it
discards the cost model, the per-candidate audit trail, the LCB edge estimation,
and the portfolio layer — all of which are the parts most likely to reveal that a
published anomaly does not survive retail implementation. The value here is in
measuring the gap between the paper result and the tradable result, and that
requires the machinery.

**Stops on momentum positions at a normal 3 ATR.** Documented to hurt momentum.
The 5 ATR disaster stop is the compromise: rarely binding, bounds a single-name
catastrophe.

**Self-reported strategy confidence to distinguish "strong" from "weak" momentum.**
Forbidden by [ADR-014](014-no-self-reported-confidence.md). The bin statistics
answer this empirically, and `mom_252_xs_pct` is a bin dimension.

## Revisit trigger

If M3 shows `xsec_momentum_v1` with a flat calibration curve — predicted `ev_net_r`
carrying no information about realised R — then either the bin dimensions are wrong
or equity momentum at this horizon and cost level is exhausted. Diagnose in that
order.
