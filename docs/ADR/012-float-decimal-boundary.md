# ADR-012: Float/Decimal boundary

**Status:** Accepted

## Context

The global project rule requires `Decimal` or integer minor units at three
boundaries: order construction, balance and position accounting, and P&L or fee
accumulation. It also states that floats are fine for time-series data,
indicators, plotting, and analytics.

Applying that rule needs an unambiguous line, because the two failure modes are
symmetric and both are bad: a float ledger accumulates cent-level error over
thousands of trades and silently drifts from the exchange's numbers, while a
`Decimal` DataFrame is roughly 1000× slower and makes a 90-second backtest take
a day.

## Decision

### `Decimal` — mandatory

| Location | Fields |
|---|---|
| `portfolio/sizing.py`, from `size_position` onward | `qty`, `risk_usd`, notional |
| `domain/portfolio.py` — `Position` | `qty`, all prices, fees, funding |
| `domain/portfolio.py` — `PortfolioState` | all `*_usd` fields |
| `domain/execution.py` — `OrderIntent`, `Fill` | `qty`, prices, `fee_usd` |
| `domain/results.py` — `ClosedTrade` | all `*_usd` fields |
| `backtest/ledger.py` | every arithmetic operation |
| `execution/*` | everything sent to an exchange |
| `data/universe/assets.parquet` | `tick_size`, `step_size`, `min_notional_usd` — stored as **strings**, parsed to `Decimal` on load |

### `float` — mandatory

`MarketPanel`, `FeaturePanel`, every indicator, `Setup` prices, `BinStats`,
`Opportunity`, `CostEstimate`, `realised_r` / `mae_r` / `mfe_r`,
`DecisionRecord`, all metrics and plots.

### The crossing point

**`portfolio/sizing.py::size_position`** is the boundary. It receives float
`Setup` prices and float `Opportunity` statistics, and returns `Decimal` `qty` and
`risk_usd`. Everything upstream is float; everything downstream is `Decimal`.

### Conversion and rounding rules

```python
d = Decimal(str(f))     # ALWAYS via str. Decimal(0.1) is
                        # 0.1000000000000000055511151231257827.
f = float(d)            # only when writing an analytics record

QTY:    ROUND_DOWN to step_size          # never round up into extra risk
PRICE:  entries and targets round to the conservative side
STOPS:  round AWAY from entry, never closer -- rounding a stop closer
        silently tightens risk below the mandate and increases stop-outs
CENTS = Decimal("0.01")
```

## Consequences

- The ledger is exact and reconciles with the exchange to the cent.
- The feature and analytics path stays fast: 5.3M float rows.
- Performance cost of `Decimal` is negligible where it is used — roughly 1,000
  trades × 20 operations is microseconds.
- One clear rule for implementers: *"if it is money or an order, `Decimal`; if it
  is a statistic or a time series, `float`."*
- Cost: conversions at the boundary. Localised to one function, so it is testable
  in one place.
- `tick_size` and `step_size` stored as strings looks odd but is required. A
  `step_size` of 0.001 stored as a double is not 0.001, and rounding a quantity to
  a not-quite-right step produces exchange rejections that are maddening to
  diagnose.

## Alternatives rejected

**`Decimal` everywhere.** Roughly 1000× slower on the panel; a 90-second backtest
becomes a day. Also incompatible with numpy and pandas, so every indicator would
need reimplementing.

**`float` everywhere.** Cent-level drift accumulating over thousands of trades,
divergence from the exchange's accounting, and eventual order rejections from
quantities that are not valid multiples of the step size.

**Integer minor units (satoshis, cents).** Exact and fast, and defensible. Rejected
because crypto step sizes vary by many orders of magnitude across symbols
(BTC at 0.001, SHIB at 1000), so a single integer scale does not exist and
per-symbol scaling factors are more error-prone than `Decimal`.

**`Decimal` for `Setup` prices.** `Setup` is geometric intent, not an order, and
it is produced inside the hot per-bar path. Converting at sizing is the right
boundary.

## Revisit trigger

None foreseen.
