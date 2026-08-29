# Portfolio and Risk

> The stage that turns a ranked list into at most a few sized orders. Module:
> `src/scout/portfolio/`

This is the component that decides whether "AAPL long, MSFT long, NVDA long" is
three positions or one. The answer is one (INFO_TECH), and the design follows
from that.

---

## 1. Inputs and outputs

```python
def select_and_size(
    ranked: Sequence[Opportunity],            # sorted by ev_per_bar_r desc
    state: PortfolioState,
    sentiment: Mapping[str, SentimentView],
    assets: Mapping[str, Asset],
    cfg: PortfolioConfig,
    risk_cfg: RiskConfig,
    sentiment_cfg: SentimentConfig,
    cost_cfg: CostConfig,
    min_ev_net_r: float,                      # from scoring config
) -> tuple[TradeDecision, ...]:
    """Returns one TradeDecision per input Opportunity, accepted or rejected.
    Rejections are returned, not dropped: the audit trail needs them."""
```

What this stage sees: `ev_net_r`, `direction`, `cluster`, `beta_bench_90`, the risk
distance, and liquidity. What it does **not** see: which strategy produced the
setup, what the trigger was, or any feature beyond those listed. This is the
brief's "no strategy-specific portfolio logic" enforced by the type signature
rather than by discipline.

---

## 2. The greedy algorithm

```python
accepted: list[TradeDecision] = []
provisional = state                     # mutated copy as we accept

for rank, opp in enumerate(ranked, start=1):

    # --- global blocks: checked first, cheapest, and most important ---
    if not risk_cfg.trading_enabled:                  reject(KILL_SWITCH)
    if breaker_tripped(provisional, risk_cfg):        reject(CIRCUIT_BREAKER)

    # --- per-opportunity blocks ---
    if opp.symbol in provisional.positions:           reject(ALREADY_IN_POSITION)
    if len(provisional.positions) >= cfg.max_positions:      reject(MAX_POSITIONS)
    if rank > cfg.top_n:                              reject(BELOW_TOP_N)

    # --- sentiment: veto or penalty, never a bonus ---
    mult_sentiment = sentiment_multiplier(sentiment.get(opp.symbol), opp.direction,
                                          sentiment_cfg)
    if mult_sentiment <= 0.0:                         reject(SENTIMENT_VETO)

    # --- size, then check caps against the ACTUAL size ---
    mult = mult_sentiment * liquidity_multiplier(opp, cfg) * heat_multiplier(provisional, cfg)
    qty, risk_usd = size_position(opp, provisional, assets[opp.symbol], mult, cfg)

    if qty <= 0:                                      reject(SIZE_BELOW_MIN_NOTIONAL)
    if notional(qty, opp) < assets[opp.symbol].min_notional_usd:
                                                      reject(SIZE_BELOW_MIN_NOTIONAL)

    # --- portfolio caps, evaluated on the post-trade state ---
    after = simulate_add(provisional, opp, qty, risk_usd)
    if after.open_risk_pct > cfg.max_portfolio_heat_pct:     reject(PORTFOLIO_HEAT_CAP)
    if after.cluster_risk_pct(opp.cluster) > cfg.max_cluster_risk_pct: reject(CLUSTER_CAP)
    if abs(after.net_beta_exposure_pct) > cfg.max_net_beta_pct:        reject(BETA_CAP)
    if after.gross_exposure_pct > cfg.max_gross_exposure_pct: reject(GROSS_EXPOSURE_CAP)

    # --- re-check EV with the final cost at the final size ---
    final_cost = estimate_cost(opp.setup, ..., notional(qty, opp), risk_usd, ...)
    if opp.ev_r_lcb - final_cost.cost_r < min_ev_net_r: reject(BELOW_EV_THRESHOLD)

    accept(opp, qty, risk_usd, mult, final_cost)
    provisional = after
```

Two properties of this loop matter more than the individual caps:

1. **Caps are checked against `provisional`, which includes decisions already
   accepted in this same cycle.** Checking against `state` alone would let three
   simultaneously-accepted trades jointly breach the heat cap while each
   individually passes. This is the most common portfolio-gate bug, it only
   appears on cycles with multiple signals, and those cycles are precisely the
   high-correlation moments when the cap matters most.
2. **Rejection reasons are recorded in the order the checks run**, so the order
   is fixed and must not be changed casually — funnel analysis compares reason
   counts across runs.

Rejecting a candidate never terminates the loop. A lower-ranked opportunity in a
different cluster can still be accepted after a higher-ranked one was blocked by
a cluster cap, which is the desired diversifying behaviour.

---

## 3. Position sizing

```python
def size_position(opp, state, asset, multiplier, cfg) -> tuple[Decimal, Decimal]:
    """Risk-based sizing. Decimal from here on."""

    equity = state.equity_usd                                   # Decimal
    risk_frac = Decimal(str(cfg.risk_fraction_per_trade))       # 0.004
    mult = Decimal(str(multiplier))                             # (0, 1]

    risk_usd = (equity * risk_frac * mult).quantize(CENTS, ROUND_DOWN)

    entry = Decimal(str(opp.setup.reference_price))
    stop = Decimal(str(opp.setup.stop_price))
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        raise ScoutError("setup with non-positive risk reached sizing")

    qty_raw = risk_usd / risk_per_unit

    # Notional caps, applied BEFORE step rounding
    max_notional = min(
        equity * Decimal(str(cfg.max_position_notional_pct)),
        Decimal(str(opp.adv_usd_30 * cfg.max_pct_of_adv)),
    )
    qty_raw = min(qty_raw, max_notional / entry)

    # Step rounding: ALWAYS down. Rounding up increases risk beyond the mandate.
    qty = (qty_raw / asset.step_size).to_integral_value(ROUND_DOWN) * asset.step_size

    actual_risk = qty * risk_per_unit
    return qty, actual_risk
```

Note the two notional caps beyond the risk-based size:

- `max_position_notional_pct` (default 0.25) bounds leverage. Risk-based sizing
  with a tight stop implies enormous notional: a 0.4% risk with a 2.4% stop
  distance is 41× the risk capital, or 17% of equity. In a low-volatility regime
  where `atr_pct` is 0.4%, the same formula asks for 50% of equity in one symbol.
  Risk-based sizing without a notional cap produces catastrophic positions
  precisely when volatility is lowest, which is when it is about to rise.
- `max_pct_of_adv` (default 0.005) bounds market impact and, more importantly,
  bounds *exit* impact. You must be able to get out.

---

## 4. Portfolio heat

```python
open_risk_pct = Σ over positions of open_risk_usd(mark) / equity_usd
```

where `open_risk_usd` is the distance from the current mark to the stop, times
quantity, floored at zero. It decreases as a trade moves in your favour, so
winners free up capacity — the intended behaviour.

`max_portfolio_heat_pct`, default **0.020** (2%).

With `risk_fraction_per_trade = 0.004`, that permits about 5 concurrent full-risk
positions. In a correlated drawdown where everything stops out together, the loss
is 2% of equity. Combined with the cluster cap, the realistic worst single-day
outcome is bounded near 2%, and surviving 20 consecutive such days is what
"staying in the game" means.

### Heat multiplier

```python
def heat_multiplier(state, cfg) -> float:
    """Taper size as the portfolio fills up, rather than allowing full size
    right up to a hard wall."""
    used = state.open_risk_pct / cfg.max_portfolio_heat_pct
    if used <= cfg.heat_taper_start:                 # default 0.60
        return 1.0
    remaining = (1.0 - used) / (1.0 - cfg.heat_taper_start)
    return max(cfg.heat_min_multiplier, remaining)   # floor default 0.35
```

A hard wall means the fifth position gets full size and the sixth gets nothing.
Tapering makes the marginal position smaller as concentration rises, which is
both smoother and closer to what a risk manager would actually do.

### Liquidity multiplier

The third and last multiplier. It exists so that a symbol just above the ADV gate
is not treated identically to BTC:

```python
def liquidity_multiplier(opp: Opportunity, cfg: PortfolioConfig) -> float:
    """Taper size for symbols near the liquidity floor. Returns (0, 1]."""
    ratio = opp.adv_usd_30 / cfg.liquidity_full_size_adv_usd     # default 200e6
    return min(1.0, max(cfg.liquidity_min_multiplier, sqrt(ratio)))  # floor 0.50
```

Square-root scaling because impact scales with the square root of participation
([`08-COSTS.md §2.4`](08-COSTS.md#24-market-impact)), so halving size in a market
with a quarter the depth restores the same impact. A symbol at the $20M floor gets
`sqrt(0.1) = 0.32`, clipped to the 0.50 floor; anything above $200M ADV gets full
size.

The three multipliers compose: `mult = sentiment × liquidity × heat`. All are in
`(0, 1]`, so the product can only reduce size — property-tested.

---

## 5. Clusters

Static map in config. No estimated covariance matrix
([`ADR/007-cluster-caps-not-covariance.md`](ADR/007-cluster-caps-not-covariance.md)).

```yaml
# Cluster *names* for equities. Membership comes from the vendor's GICS (or
# equivalent) at snapshot time — not a hand-maintained symbol list.
clusters:
  INFO_TECH: []
  FINANCE: []
  HEALTH: []
  CONSUMER_DISC: []
  CONSUMER_STAPLES: []
  INDUSTRIALS: []
  ENERGY: []
  MATERIALS: []
  COMM_SVCS: []
  UTILITIES: []
  REAL_ESTATE: []
  ETF_BROAD: []
  ETF_SECTOR: []      # unused for XLK etc. — sector ETFs join the GICS cluster
  ETF_INTL: []
  ETF_BOND: []
  ETF_COMMODITY: []
  OTHER: []

portfolio:
  max_cluster_risk_pct: 0.010           # half the total heat cap
  max_positions_per_cluster: 2
```

A symbol not present in any cluster falls into `OTHER`, and config validation
**warns** with the list of unmapped symbols. Unmapped symbols silently pooling
into one bucket would make `OTHER` the largest cluster and defeat the cap.

**A sector ETF and its constituents are the same cluster.** `XLK` maps to
`INFO_TECH`, not `ETF_SECTOR`, so holding XLK plus four semiconductor names is
capped as one theme. International, bond, and commodity ETFs get their own
clusters because their residual risk is not the US equity market.

`max_cluster_risk_pct` at half the total heat cap means no single theme can be
more than half the portfolio's risk. That is the single most valuable constraint
in this document: five info-tech longs are capped at 1% total risk rather than
2% across five positions pretending to be independent.

### The honest limitation

In a genuine risk-off cascade, all equity clusters go to correlation 1.0 and
the cluster cap does nothing. What protects you then is `max_portfolio_heat_pct`,
the SPY market-regime gate, and the drawdown circuit breaker. The cluster map is
protection against *thematic* concentration, which is the common case; the heat
cap is protection against the tail. Both are needed and neither substitutes for
the other.

---

## 6. Net beta exposure

```python
net_beta_exposure_pct = Σ over positions of (
    direction.sign * open_risk_usd / equity_usd * beta_bench_90
)
```

`max_net_beta_pct`, default **0.015**.

This is the entire market-exposure model: risk-weighted, beta-adjusted, signed,
against **SPY**. It permits a fully long-biased book up to 1.5% of equity in
SPY-equivalent risk and gives credit for genuine hedges.

Using `open_risk_usd` rather than notional as the weight is deliberate. Notional
weighting would treat a tight-stopped position as a huge exposure when its actual
downside is small.

---

## 7. The float/Decimal boundary

The global rule says `Decimal` at three boundaries. Here is exactly where, so
nobody Decimal-ifies a pandas DataFrame and nobody floats a ledger.

### `Decimal` — mandatory

| Location | Fields |
|---|---|
| `portfolio/sizing.py`, from `size_position` onward | `qty`, `risk_usd`, notional |
| `domain/portfolio.py` — `Position` | `qty`, `entry_price`, `stop_price`, `target_price`, fees, funding |
| `domain/portfolio.py` — `PortfolioState` | `equity_usd`, `cash_usd`, `peak_equity_usd`, all P&L |
| `domain/execution.py` — `OrderIntent`, `Fill` | `qty`, prices, `fee_usd` |
| `domain/results.py` — `ClosedTrade` | all `*_usd` fields |
| `backtest/ledger.py` | every arithmetic operation |
| `execution/*` | every price and quantity sent to an exchange |

### `float` — mandatory

| Location | Why |
|---|---|
| `MarketPanel`, `FeaturePanel` | pandas/numpy. Decimal here would be 1000× slower and buys nothing. |
| Every indicator | same |
| `Setup` prices | geometric intent, not an order. Converted to Decimal at sizing. |
| `BinStats`, `Opportunity`, `CostEstimate` | statistics; `cost_r` of 0.159 needs no cent-exactness |
| `realised_r`, `mae_r`, `mfe_r` | analytics |
| `DecisionRecord` | analytics table |
| Metrics, plots | analytics |

### Conversion rules

```python
# float -> Decimal: ALWAYS via str. Decimal(0.1) is
# 0.1000000000000000055511151231257827, which defeats the purpose.
d = Decimal(str(f))

# Decimal -> float: only when writing an analytics record.
f = float(d)

# Rounding
QTY:    ROUND_DOWN to step_size     # never round up into extra risk
PRICE:  entries and targets ROUND toward the conservative side;
        stops ROUND toward the conservative side (further from entry never
        closer) -- rounding a stop closer to entry silently tightens risk
        below the mandate and increases stop-out frequency
CENTS = Decimal("0.01")             # for USD amounts
```

Ledger arithmetic uses `Decimal` throughout. Cost per backtest: at 1,000 trades
with roughly 20 Decimal operations each, that is 20,000 operations, which is
microseconds. The float DataFrame path handles 5.3M rows. Both are in the right
place.

---

## 8. Kill switch and circuit breakers

Mandatory before every order send, per the global rule. In backtest they are
active too, because a backtest that ignores its own risk limits is not measuring
the system you intend to run.

```yaml
risk:
  trading_enabled: true               # env override: SCOUT_TRADING_ENABLED=0
  max_daily_loss_pct: 0.030           # realised + unrealised vs day-start equity
  max_drawdown_pct: 0.150             # from peak equity
  max_trades_per_day: 8
  breaker_cooldown_bars: 30           # bars of no new entries after a trip
```

```python
def breaker_tripped(state: PortfolioState, cfg: RiskConfig) -> RejectionReason | None:
    if state.day_loss_pct() >= cfg.max_daily_loss_pct:  return CIRCUIT_BREAKER
    if state.drawdown_pct() >= cfg.max_drawdown_pct:    return CIRCUIT_BREAKER
    if state.trades_today >= cfg.max_trades_per_day:    return CIRCUIT_BREAKER
    if state.bars_since_breaker < cfg.breaker_cooldown_bars: return CIRCUIT_BREAKER
    return None
```

**Breakers block opening risk only.** Reduce-only exits, stops, targets, and the
time stop always proceed. This is the asymmetric failure policy
([`01-ARCHITECTURE.md §7`](01-ARCHITECTURE.md#7-failure-policy)) applied here: a
tripped breaker must never trap you in a position.

`trading_enabled` reads from `SCOUT_TRADING_ENABLED` first and config second, so
it can be flipped without a deploy. In live mode it is re-read **before every
order send**, not cached at startup — a kill switch you have to restart the
process to use is not a kill switch.

### Why breakers run in the backtest

The 15% drawdown breaker with a 30-bar cooldown changes the equity curve, and it
changes it most in exactly the periods you most want to understand. Reporting a
backtest without breakers and then deploying with them means the deployed system
is not the one you validated. Report both if you like, but the headline number
must include them.

---

## 9. Exits

Every position is opened as a bracket: entry, protective stop, take-profit. The
stop and target are **exchange-native conditional orders**, submitted with the
entry ([`ADR/008-bracket-exits.md`](ADR/008-bracket-exits.md)).

Consequences:

- Exits do not need the decision pipeline, so a daily decision clock does not
  leave positions unprotected overnight. Stops are live while the cash session
  is closed, which is when most equity gap risk arrives.
- If the process dies, protection survives. This is the single most important
  operational property of the design.
- If the stop cannot be placed, the entry is closed immediately. A naked
  position is never acceptable, and `submit_bracket` enforces this.

The only pipeline-driven exit is the **time stop** at `max_hold_bars`, evaluated
at the start of each cycle before new entries are considered.

**No trailing stops, no partial exits, no pyramiding, no stop-to-breakeven in
v1.** Each adds parameters and interacts with the labeling pass — the edge
statistics describe a fixed bracket, so changing the exit rule invalidates them
and requires a full relabel plus a version bump. Exit-rule variants are M4
experiments, each spending trial budget.

---

## 10. Full configuration

```yaml
portfolio:
  initial_equity_usd: 100000
  risk_fraction_per_trade: 0.004        # 0.4% of equity at risk per trade
  max_positions: 6
  max_positions_per_cluster: 2
  top_n: 3                              # max new entries per cycle
  max_portfolio_heat_pct: 0.020
  max_cluster_risk_pct: 0.010
  max_net_beta_pct: 0.015
  max_gross_exposure_pct: 1.50          # sum of notionals / equity
  max_position_notional_pct: 0.25
  max_pct_of_adv: 0.005
  heat_taper_start: 0.60
  heat_min_multiplier: 0.35
  liquidity_full_size_adv_usd: 200000000
  liquidity_min_multiplier: 0.50
```

The final-size EV re-check in §2 reads **`scoring.min_ev_net_r`**, not a separate
portfolio threshold. There is one threshold and it lives in the `scoring` section
([`14-CONFIG.md §3`](14-CONFIG.md#3-configbaseyaml--the-complete-default)).
Duplicating it here would allow the two to drift.

`risk_fraction_per_trade` at 0.4% sits in the middle of the brief's suggested
0.25–0.50% band. It is **not a parameter to optimise**: it scales the equity
curve without changing risk-adjusted performance, so sweeping it only inflates
the trial count while teaching you nothing. Set it once from the drawdown you can
tolerate and leave it. The one thing worth checking is that it interacts sanely
with `max_portfolio_heat_pct`, which it does at 5 concurrent positions.

`top_n = 3` implements the brief's funnel directly: however many candidates clear
the EV threshold, at most 3 new entries per cycle.
