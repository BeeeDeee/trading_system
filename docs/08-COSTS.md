# Execution Costs

> One implementation, shared by backtest, paper, and live. Module:
> `src/scout/costs/model.py`

A strategy that works before costs and fails after costs is not a strategy. The
cost model is therefore built to be **pessimistic by default**: every estimate
that could go either way goes against you.

**Scope: US-listed common stocks and ETFs.** The crypto model is in
[§6](#6-appendix-crypto-perpetuals-m7-optional).

This document contains the single most important number in the project. The
7× cost reduction derived in [§3](#3-worked-examples) is the reason equities are
the primary scope ([ADR-015](ADR/015-equities-first.md)).

---

## 1. The single entry point

```python
def estimate_cost(
    setup: Setup,
    universe_entry: UniverseEntry,
    notional_usd: float,
    risk_capital_usd: float,
    expected_bars_held: float,          # SESSIONS
    cfg: CostConfig,
) -> CostEstimate:
```

There is exactly one implementation and no mode flag. If backtest and live could
differ, they eventually would, and the backtest would be the optimistic one.
`SimBroker` calls this function to compute fills; `LiveBroker` calls it to
pre-check that a trade still clears the threshold at the current spread.

---

## 2. Components

All in basis points of notional unless stated.

### 2.1 Commission

Unlike crypto, this is not a percentage. It is per share, with a floor, which
changes its behaviour completely.

```python
shares = notional_usd / price_raw

per_side_usd = min(
    max(cfg.commission_per_share_usd * shares, cfg.commission_min_usd),
    cfg.commission_max_pct_of_notional * notional_usd,
)
commission_bps = 2.0 * per_side_usd / notional_usd * 1e4
```

Defaults model **IBKR Pro**: `$0.0035` per share, `$0.35` minimum per order, capped
at 1% of trade value. Zero-commission brokers (Schwab, Fidelity, IBKR Lite) would
justify `0.0`, and the model is deliberately more expensive than the cheapest
available option.

Two non-obvious consequences that the percentage-based crypto model did not have:

- **The `$0.35` minimum penalises small positions.** On a $7,400 notional it is
  0.9 bps; on a $1,000 notional it is 7.0 bps. This system wants **at least
  $50,000 of capital** to be cost-efficient at these thresholds, or a
  zero-commission broker. Below that, use a zero-commission broker and set
  `commission_per_share_usd: 0.0`.
- **Per-share pricing penalises low-priced stocks.** A $22 stock incurs ten times
  the commission in basis points of a $220 stock at the same notional. Combined
  with wider spreads at the low end, this is a second reason for the `$5.00`
  minimum-price gate in
  [`04-DATA_AND_UNIVERSE.md §7.2`](04-DATA_AND_UNIVERSE.md#72-eligibility-rules).

### 2.2 Spread

```python
spread_bps = universe_entry.spread_bps_est      # Corwin-Schultz, floored by tier
spread_component_bps = 2.0 * spread_bps
```

Crossed once per side, so it appears twice. Estimator and per-tier floors:
[`04-DATA_AND_UNIVERSE.md §7.3`](04-DATA_AND_UNIVERSE.md#73-the-spread-estimate).

Charging the full spread on both sides is conservative for the entry, because a
market-on-open order participates in the **opening auction**, which clears at a
single price with no bid-ask spread to cross. Keeping the charge is deliberate:
the auction has its own imbalance-driven price concession, and it is not worth
introducing a separate parameter for something the spread term already covers at
roughly the right magnitude.

### 2.3 Slippage

Two additive pieces, each per side.

```python
slip_fixed_bps = cfg.slippage_fixed_bps                   # default 1.0
slip_vol_bps   = cfg.slippage_vol_coef * atr_pct * 1e4    # coef default 0.02

slippage_bps = 2.0 * (slip_fixed_bps + slip_vol_bps)
```

**The volatility coefficient is 0.02 for equities against 0.05 for crypto, and the
reason is structural, not a tuning choice.** In crypto, the entry is a market
order at the open of the next 4h bar, so the fill lands up to four hours after the
decision and the price has drifted. In equities, the entry is a
market-on-open order in the opening auction: a single clearing price, at a known
instant, with no drift between decision and fill and no spread to cross.

There is of course a large price change between the decision close and the next
open — the overnight gap. **That is not slippage.** It is unbiased market movement,
it is symmetric, and it is already captured where it belongs: the labeling pass
re-anchors R on the *actual* entry price
([`07-EDGE_AND_SCORING.md`](07-EDGE_AND_SCORING.md)), so a gap in your favour
shrinks the R denominator and a gap against you widens it. Charging it as a cost
would be double-counting; ignoring it in the labeling would be lookahead. Neither
happens.

The residual 0.02 coefficient covers auction imbalance and the fact that the
opening print in a volatile name is a worse execution than in a quiet one.

### 2.4 Market impact

```python
participation = notional_usd / max(universe_entry.adv_usd_60, 1.0)
impact_bps = cfg.impact_coef * atr_pct * 1e4 * sqrt(participation)   # coef default 1.0
```

The square-root law: fractional impact scales as `Y · σ · sqrt(participation)`,
where `σ` is the asset's volatility and `Y` is a dimensionless coefficient
typically 0.5–1.5 for liquid markets. **The volatility term is not optional** — a
bare `coef · sqrt(participation)` has no unit and produces nonsense across assets
with different volatilities. `impact_coef` defaults to 1.0, the pessimistic end of
the literature range; it cannot be calibrated without live fills, so it is
recalibrated from real fills at M5.

Applied once (entry only). Exits are stops, targets, or scheduled market orders,
whose impact is partly reflected in the fill assumptions in
[`11-BACKTEST_ENGINE.md §5`](11-BACKTEST_ENGINE.md#5-fill-rules).

For a retail account this term is near zero at the top of the universe (0.4 bps at
$1.5B ADV) and material at the bottom (7.1 bps at the $8M ADV floor). That spread
is the whole justification for the liquidity rank in
[ADR-018](ADR/018-liquidity-rank-universe.md), and it is why the apparent edge in
illiquid names is almost always impact you did not model.

### 2.5 Borrow and dividends (shorts only)

This replaces crypto funding, and it behaves very differently.

```python
if direction is Direction.SHORT:
    years_held  = expected_bars_held / 252.0
    borrow_bps  = universe_entry.borrow_bps_per_year * years_held
    dividend_bps = universe_entry.dividend_yield_annual * 1e4 * years_held
else:
    borrow_bps   = 0.0      # longs have no borrow cost
    dividend_bps = 0.0      # longs RECEIVE dividends; not credited here (conservative)
```

**Longs are not credited for dividends received**, even though the ledger applies
them ([`04-DATA_AND_UNIVERSE.md §4.3`](04-DATA_AND_UNIVERSE.md#43-dividends-as-ledger-cash-flows)).
This is the required conservative asymmetry (project rule 4): the cost estimate
that gates the decision never counts a benefit that might not materialise, while
the ledger records what actually happened. The result is that realised long P&L is
slightly better than the estimate, which is the correct direction of error.

Magnitudes, and the surprise:

| Component | 21-session short of a typical large cap |
|---|---|
| Borrow at 30 bps/yr (easy-to-borrow) | 2.5 bps |
| Dividend at a 1.5% annual yield | 12.5 bps |

**Dividend cost dominates borrow cost by 5×.** This is counter-intuitive and it is
the item most often omitted from retail equity backtests. Dividends are also
*discrete*: over a 21-session window there is roughly a one-in-three chance of an
ex-date, and when it lands it is a single ~50 bps event, not 12.5 bps smeared out.
The cost model charges the expectation; `SimBroker` applies the realisation.

**Hard-to-borrow is a gate, not a cost.** Borrow rates on genuinely hard-to-borrow
names run to 20%+ per year, and at that level the position is not viable. Rather
than model it, `evaluate_gates` rejects shorts when the passed-in borrow rate
exceeds the cap (default 300 bps/yr). `UniverseEntry` does not carry borrow;
the engine supplies `borrow_bps_per_year`. `gates.skip_hard_to_borrow` defaults
to true because v1 has no historical borrow file.

If historical borrow-rate data is unavailable — and for most vendors it is —
`borrow_bps_per_year` falls back to `cfg.borrow_bps_per_year_default` and the
`HARD_TO_BORROW` gate cannot fire (`skip_hard_to_borrow: true`, or a `None`
rate). That is a **known optimism** in the short book
and it is recorded as a limitation in every run's `run.log`. It is also a reason to
read the short-side results more sceptically than the long side.

### 2.6 Short-sale restrictions: not modelled

Regulation SHO's alternative uptick rule bans short sales at or below the national
best bid for the remainder of the session and the following session after a stock
falls 10% from the prior close. This is **not modelled**. Its effect is to prevent
some short entries immediately after a large decline — a case where the backtest
will show a fill that live trading might not achieve.

Stated as a limitation rather than modelled because doing it properly needs
intraday quote data. Its practical impact on this system is small: entries are
market-on-open, and `xsec_momentum_v1` shorts weak names rather than crashing ones.
Recorded in `run.log`.

---

## 3. Worked examples

```python
total_bps = (commission_bps
             + 2.0 * spread_bps
             + slippage_bps
             + impact_bps
             + borrow_bps
             + dividend_bps)

cost_usd = total_bps / 1e4 * notional_usd
cost_r   = cost_usd / risk_capital_usd
```

Derivation of the `cost_r` conversion:
[`07-EDGE_AND_SCORING.md §6`](07-EDGE_AND_SCORING.md#6-step-4--costs-in-r).

Common to all three: `equity = $100,000`, `risk_fraction = 0.004` →
`risk_capital = $400`, `stop_atr = 3.0`, `expected_bars_held = 21` sessions.

### 3.1 Example A — liquid large cap, long

`atr_pct = 0.018`, `price_raw = $230`, `adv_usd_60 = $1,500,000,000`,
spread tier `> $500M` → 1.0 bps.

```text
risk_per_unit_pct  = 3.0 * 0.018                        = 0.054
notional           = 400 / 0.054                        = $7,407
shares             = 7,407 / 230                        = 32

commission  2 * max(32*0.0035, 0.35) = 2 * $0.35 = $0.70 =  0.94 bps
spread      2 * 1.0                                     =  2.00 bps
slippage    2 * (1.0 + 0.02 * 0.018 * 1e4 = 3.6)        =  9.20 bps
impact      1.0 * 0.018 * 1e4 * sqrt(7,407 / 1.5e9)     =  0.40 bps
borrow      long                                        =  0.00 bps
dividend    long, not credited                          =  0.00 bps
                                                          ────────
total                                                   = 12.54 bps

cost_usd = 0.001254 * 7,407                             = $9.29
cost_r   = 9.29 / 400                                   = 0.023 R
```

### 3.2 Example B — bottom of the universe, long

`atr_pct = 0.030`, `price_raw = $22`, `adv_usd_60 = $8,000,000`,
spread tier `else` → 12.0 bps estimated.

```text
risk_per_unit_pct  = 3.0 * 0.030                        = 0.090
notional           = 400 / 0.090                        = $4,444
shares             = 4,444 / 22                         = 202

commission  2 * max(202*0.0035, 0.35) = 2 * $0.71       =  3.19 bps
spread      2 * 12.0                                    = 24.00 bps
slippage    2 * (1.0 + 0.02 * 0.030 * 1e4 = 6.0)        = 14.00 bps
impact      1.0 * 0.030 * 1e4 * sqrt(4,444 / 8e6)       =  7.07 bps
                                                          ────────
total                                                   = 48.26 bps

cost_usd = 0.004826 * 4,444                             = $21.45
cost_r   = 21.45 / 400                                  = 0.054 R
```

**2.3× the cost of Example A, and still a third of crypto's best case.** Note that
the per-share commission is 3.4× worse in bps despite a smaller notional, purely
because the share price is 10× lower.

### 3.3 Example C — liquid large cap, short

Example A, short, with a 1.5% annual dividend yield and 30 bps/yr borrow.

```text
subtotal from Example A                                 = 12.54 bps
borrow      30.0 * (21/252)                             =  2.50 bps
dividend    0.015 * 1e4 * (21/252)                      = 12.50 bps
                                                          ────────
total                                                   = 27.54 bps

cost_usd = 0.002754 * 7,407                             = $20.40
cost_r   = 20.40 / 400                                  = 0.051 R
```

**Shorts cost 2.2× longs**, and the dividend term is the largest single component
after slippage. A long/short system is not symmetric in cost, and any result where
the short book contributes as much as the long book should be checked against this
first.

### 3.4 What this means: the required edge, and whether it is plausible

Comparison against the crypto baseline (`08-COSTS.md` at the previous revision):

| | Crypto perp, BTCUSDT | Equity large cap, long |
|---|---|---|
| `cost_r` | **0.159 R** | **0.023 R** |
| Ratio | — | **6.9× cheaper** |

Now the question that actually matters. Working backwards from a target Sharpe
rather than forwards from a hope:

```text
Assume: 6 concurrent positions, 0.4% equity risk each, per-trade realised-R
standard deviation ~1.2 R, average pairwise correlation among simultaneous
momentum positions ~0.4, 21-session holds (~12 rebalances/year).

Per-rebalance portfolio return std
  = 0.4% * 1.2 * sqrt(6 + 6*5*0.4)  = 0.48% * 4.24        = 2.04%
Annualised vol
  = 2.04% * sqrt(12)                                       = 7.1%

For a net Sharpe of 0.5:
  required annual return                                   = 3.5%
  per rebalance                                            = 0.30%
  per position                                             = 0.05% of equity
  in R units (0.05% / 0.4%)                                = 0.125 R net
  gross, adding cost_r                                     = 0.148 R
```

**So: gross `mean_r ≈ 0.15 R` per trade buys a net Sharpe of about 0.5.** In price
terms that is a 0.8% average gross gain over 21 sessions on a 5.4% risk unit —
squarely inside the range that the published cross-sectional momentum literature
reports for top-decile-minus-market returns. The arithmetic hangs together.

The equivalent crypto calculation needed **0.33 R gross** for the same Sharpe, of
which nearly half was consumed by cost before any edge appeared. That is the
difference between a plausible target and an implausible one, and it is the entire
argument of [ADR-015](ADR/015-equities-first.md) in one line.

Three further implications, stated plainly:

- **Cost is now 15% of the required gross edge, not 48%.** Cost-estimation *error*
  is therefore no longer capable of deciding the outcome, which is a qualitatively
  different situation.
- **The dominant cost is slippage (73% of Example A)**, and it is the component
  estimated with the least evidence. It is also the one most improved by real fill
  data, which is why §4 exists and why `slippage_vol_coef` is recalibrated at M4
  before the go-live decision.
- **Wider stops are cheaper per unit of risk.** `stop_atr = 4.0` reduces notional
  per unit of risk and therefore reduces `cost_r` proportionally, to about 0.017 R.
  This is a genuine, non-obvious result of doing the arithmetic before
  implementing, and it partly justifies the 5 ATR disaster stop on
  `xsec_momentum_v1` ([ADR-019](ADR/019-cross-sectional-momentum-primary.md)) being
  cheap as well as appropriate. It is worth one of the M4 parameter trials.

---

## 4. Validation of the cost model

Required before M5, and the reason the model records itemised components rather
than a single number:

1. Collect at least 50 paper-trade fills at M4.
2. For each, compare the actual fill price to the pre-trade `CostEstimate`.
3. Fit realised slippage against `atr_pct` and participation; update
   `slippage_vol_coef` and `impact_coef`.
4. **If realised cost exceeds modelled cost, re-run the M3 holdout with the
   updated model before going live.** A cost model that was optimistic invalidates
   the go-live decision, not just the cost model.

Equity-specific items to check at M4:

- **Opening auction fills against the official opening print.** If the modelled
  slippage is systematically wrong in one direction, this is where it shows.
- **At least 10 short fills**, to check the borrow and dividend assumptions, which
  are the least well-evidenced part of the model.
- **At least one ex-dividend date inside a holding period**, verifying the ledger
  cash flow has the correct sign. A sign error here is invisible in aggregate and
  systematically flatters shorts.

---

## 5. Configuration

```yaml
costs:
  # --- equities (primary) ---
  commission_per_share_usd: 0.0035
  commission_min_usd: 0.35
  commission_max_pct_of_notional: 0.01
  slippage_fixed_bps: 1.0
  slippage_vol_coef: 0.02             # x atr_pct x 1e4, per side
  impact_coef: 1.0                    # x atr_pct x sqrt(participation) x 1e4
  borrow_bps_per_year_default: 30.0
  hard_to_borrow_max_bps_per_year: 300.0
  dividend_yield_source: historical   # historical | default
  dividend_yield_default: 0.015

  # --- crypto (M7, unused before then) ---
  taker_fee_bps: 5.0
  maker_fee_bps: 2.0
  funding_source: default             # default | historical
  funding_rate_default_per_8h: 0.0001

  # --- research lever ---
  cost_multiplier: 1.0                # robustness sweeps only; NEVER below 1.0
```

`spread_bps_est` arrives on the `UniverseEntry`, so `spread_floor_bps_by_tier`
lives under **`universe:`**, not here
([`14-CONFIG.md §3`](14-CONFIG.md#3-configbaseyaml--the-complete-default)). The
floor is a property of the *estimator*, applied when the snapshot is built, so no
consumer can ever see an optimistic spread.

`cost_multiplier` is a research lever for the sensitivity analysis in
[`07-EDGE_AND_SCORING.md §10`](07-EDGE_AND_SCORING.md#10-required-m3-diagnostics).
Config validation rejects values below 1.0. There is no legitimate reason to model
lower costs than your best estimate, and the ability to do so is a foot-gun aimed
at your own account.

Given that equity costs are 7× lower, the sensitivity sweep runs at
`cost_multiplier ∈ {1.0, 1.5, 2.0, 3.0}` — a wider range than crypto used, because
`cost_r` is now small enough that a 3× error would still leave a viable strategy,
and demonstrating that is worth more than demonstrating survival at 1.5×.

---

## 6. Appendix: crypto perpetuals (M7, optional)

Retained for the optional M7 sleeve. Do not implement before an equity holdout
result exists.

- **Fees:** taker on both sides, `taker_fee_bps` default 5.0. The entry is a
  market order by construction; assuming the resting take-profit earns the maker
  fee assumes your order was at the front of the queue when price arrived.
- **Slippage:** `slippage_fixed_bps` 2.0, `slippage_vol_coef` **0.05** — 2.5× the
  equity value, because a 4h bar means the fill lands up to four hours after the
  decision (§2.3).
- **Funding**, which replaces borrow and dividends entirely:

```python
funding_intervals = expected_bars_held * bars_to_hours(timeframe) / 8.0
rate = (mean_funding_rate(symbol, trailing_window) if cfg.funding_source == "historical"
        else cfg.funding_rate_default_per_8h)
funding_bps = direction.sign * rate * funding_intervals * 1e4
```

  Signed, so it can be a benefit; do not clamp to zero. But note the asymmetry:
  crypto perpetual funding is positive most of the time, making it a persistent tax
  on longs and a persistent subsidy for shorts. A backtest ignoring funding
  systematically overstates long-biased strategies, and in a 2019–2021 sample that
  is most of the apparent edge. Historical rates are **mandatory** before any
  crypto holdout run.
- **Reference total:** BTCUSDT at `atr_pct = 0.012`, `stop_atr = 2.0`,
  `adv_usd_30 = $8B`, 20 bars (80 hours), long → 38.17 bps → **`cost_r = 0.159 R`**.
  This is the number every equity figure in §3 is compared against.
