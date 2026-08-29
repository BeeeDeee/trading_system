# Strategies

> Two strategies for v1, of deliberately different shapes. Complete rules, no
> ambiguity. Module: `src/scout/strategies/`

**Scope: US-listed common stocks and ETFs, daily decision bars.** "Bars" means
**sessions** everywhere in this document
([`04-DATA_AND_UNIVERSE.md §5`](04-DATA_AND_UNIVERSE.md#5-the-trading-calendar)).

---

## 1. What a strategy is and is not

A strategy is a **pure geometric claim**: given one `FeatureRow`, either there is a
setup here or there is not, and if there is, the entry reference, stop, optional
target, and maximum holding period are these numbers.

A strategy does **not**:

- estimate probability, expected value, or confidence — measured in `scoring`
- know about costs, equity, size, or the portfolio
- know about other symbols — but it *may* read cross-sectional features, which the
  feature layer computed from the whole panel (§2.1)
- know whether it is running in a backtest or live
- hold state between calls
- read config at call time (params are bound at construction)

The brief asked strategies to answer "what is the expected reward / expected risk
/ how confident am I". Reward and risk are geometric and *are* answered here.
Confidence is not: a strategy asserting its own confidence is an unfalsifiable free
parameter and it will be tuned until the backtest improves
([ADR-014](ADR/014-no-self-reported-confidence.md)). Confidence in this system is
the width of the empirical interval around setups like this one, computed from
data.

### 1.1 R is a sizing unit, not necessarily a stop distance

```
risk_per_unit = stop_atr * atr        # dollars per share
qty           = risk_capital_usd / risk_per_unit
1 R           = risk_capital_usd
```

`stop_atr` defines the **volatility sizing unit**. A physical stop order at that
distance is one way to use it, and `donchian_breakout_v1` does exactly that. But
`xsec_momentum_v1` sizes on a 5 ATR unit while placing its stop as a
rarely-binding disaster stop, and every downstream consumer — `ev_r_lcb`, `cost_r`,
portfolio heat, calibration — is indifferent, because all of them only ever divide
dollars by `risk_capital_usd`.
[ADR-019](ADR/019-cross-sectional-momentum-primary.md) has the full reasoning.

Consequently **`Setup.target_price` may be `None`** (no profit target; exit on time
or rank) but **`Setup.stop_price` may never be**. A naked position is never
acceptable.

---

## 2. Strategy A — `xsec_momentum_v1` (primary)

**Hypothesis.** Stocks that have outperformed their peers over the past twelve
months, excluding the most recent month, continue to outperform over the following
month, by enough to cover costs. This is cross-sectional momentum, the
best-documented anomaly in equities: Jegadeesh & Titman (1993) and thirty years of
out-of-sample replication across countries, market-cap segments, and asset
classes.

**Why it should persist.** Three mechanisms, none of which requires anyone to be
stupid: slow diffusion of information through a fragmented analyst and investor
base; institutional mandates that constrain how fast capital can rotate; and a
genuine risk premium for holding an asset whose crash risk is concentrated in
sharp market reversals — precisely when investors least want it. The third
mechanism is the one that makes the strategy uncomfortable to hold, which is also
the reason it has not been arbitraged away.

**Why it might not.** Published Sharpe ratios of 0.5–0.8 from the 1990s literature
are closer to 0.2–0.4 post-2003. Momentum is now packaged in ETFs with a few basis
points of fee, so any excess must survive competition from products that
implement it better and cheaper than you can. And momentum crashes are brutal:
April 2009, January 2021.

**Allowed market regimes:** `{RISK_ON, NEUTRAL}`. Not `RISK_OFF`.

### 2.1 The cross-sectional feature

A strategy sees one `FeatureRow` and cannot compute a rank. The feature layer
sees the whole panel and the universe snapshot, and can:

```python
# In features/cross_sectional.py, per timestamp t:
#   over symbols ELIGIBLE at t, using only data with close_time <= t
row.mom_252_skip21          # 12-month total return, skipping the last 21 sessions
row.mom_252_xs_pct          # percentile rank of the above, in [0, 1]
row.vol_xs_pct              # percentile rank of realised vol, in [0, 1]
```

Two things about `mom_252_skip21` that are decisions, not parameters:

- **Twelve months.** The horizon at which the effect is documented most robustly.
- **Skip the most recent 21 sessions.** Short-horizon returns exhibit *reversal*,
  not continuation, so including the last month contaminates the signal with the
  opposite effect. Skipping it is standard in the literature and it is baked in
  rather than exposed as a tunable, because exposing it invites fitting.

The rank must be computed over symbols eligible **at `t`**, from data with
`close_time <= t`. `tests/unit/test_no_lookahead.py` truncates the panel and
asserts every cross-sectional feature is bit-identical.

### 2.2 Rules

```python
class XSecMomentum:
    strategy_id = "xsec_momentum_v1"
    allowed_market_regimes = frozenset({MarketRegime.RISK_ON, MarketRegime.NEUTRAL})
    required_warmup_bars = 273          # 252 + 21 skip

    # params, all from config
    xs_threshold: float = 0.90          # enter above this percentile (symmetric for shorts)
    exit_xs_threshold: float = 0.70     # exit when rank decays past this
    stop_atr: float = 5.0               # DISASTER stop, and the sizing unit
    max_hold_bars: int = 21             # ~1 month

    def detect(self, row: FeatureRow) -> Setup | None:
        if not row.is_warm:
            return None

        # LONG: top decile of 12-1 momentum
        if row.mom_252_xs_pct >= self.xs_threshold:
            stop = row.close - self.stop_atr * row.atr_14
            return Setup(
                asset_id=row.asset_id, symbol=row.symbol, ts=row.ts,
                strategy_id=self.strategy_id,
                direction=Direction.LONG, regime=row.regime,
                reference_price=row.close,
                stop_price=stop,
                target_price=None,                  # exit on time or rank decay
                max_hold_bars=self.max_hold_bars,
                trigger_note="mom_252_xs_pct>=thr",
            )

        # SHORT: bottom decile — exact mirror
        if row.mom_252_xs_pct <= (1.0 - self.xs_threshold):
            stop = row.close + self.stop_atr * row.atr_14
            return Setup(..., direction=Direction.SHORT,
                         target_price=None,
                         trigger_note="mom_252_xs_pct<=1-thr")

        return None
```

### 2.3 The rank exit

Unique to this strategy and evaluated by the engine, not by `detect`:

```python
# In portfolio, at the start of each cycle, before new entries:
if position.strategy_id == "xsec_momentum_v1":
    aligned_pct = (pct if position.direction is Direction.LONG else 1.0 - pct)
    if aligned_pct < strategy.exit_xs_threshold:
        close_at_next_open(exit_reason="RANK_DECAY")
```

The hysteresis — enter at 0.90, exit at 0.70 — exists to stop a position
oscillating in and out as its rank jitters around a single threshold. Without it,
turnover roughly doubles for no change in exposure, and at 12.5 bps per round trip
that is a pure loss.

### 2.4 Parameter count: 4

`xs_threshold`, `exit_xs_threshold`, `stop_atr`, `max_hold_bars`. The short
threshold is `1 - xs_threshold` by construction, not a fifth parameter — momentum
is a symmetric cross-sectional claim, and allowing the two sides to differ would
be fitting the long/short balance to the sample.

### 2.5 Design notes

- **The 5 ATR stop is not risk management.** At a 1.8% daily `atr_pct` it sits about
  9% away, so it binds rarely. Its jobs are to bound a single-name catastrophe
  (fraud, a failed acquisition, a bankruptcy filing) and to define the sizing unit.
  A tighter stop would *hurt*: momentum returns are positively autocorrelated at
  this horizon, so exiting on a drawdown systematically sells the positions about
  to recover. This is documented, not folklore.
- **Wide stops are also cheaper.** A 5 ATR unit means less notional per unit of
  risk, so `cost_r` falls proportionally — about 0.017 R rather than 0.023 R
  ([`08-COSTS.md §3.4`](08-COSTS.md#34-what-this-means-the-required-edge-and-whether-it-is-plausible)).
  Wider stops being both appropriate and cheaper is a happy accident and worth
  noticing.
- **Most trades will resolve as `TIME`**, not `STOP` or `TARGET`. The bin
  statistics handle this without modification, and `mean_bars_held` will sit near
  21, which makes cost scaling nearly constant.
- **Expect the short leg to disappoint.** It carries most of momentum's crash risk,
  it costs 2.2× the long leg
  ([`08-COSTS.md §3.3`](08-COSTS.md#33-example-c--liquid-large-cap-short)), and the
  hard-to-borrow gate cannot fire without borrow data. Read short-side results more
  sceptically than long-side ones.
- **The market-regime gate does the heavy lifting on drawdown.** Excluding
  `RISK_OFF` is the single most robust risk control in the equity literature
  ([`05-FEATURES_AND_REGIME.md`](05-FEATURES_AND_REGIME.md)), and it is also the
  main defence against momentum crashes, which cluster in the first weeks of a
  sharp recovery from a market low. It is not sufficient — see
  [ADR-019 §5](ADR/019-cross-sectional-momentum-primary.md) and the mandatory
  momentum-crash reporting in
  [`12-RESEARCH_PROTOCOL.md`](12-RESEARCH_PROTOCOL.md).

---

## 3. Strategy B — `donchian_breakout_v1` (secondary)

**Hypothesis.** A stock making a new 55-session extreme, while its own structure
is efficiently trending in the same direction, continues more often than a random
walk implies. This is time-series (absolute) momentum rather than cross-sectional,
and its evidence base — Moskowitz, Ooi & Pedersen (2012) and the managed-futures
literature — is real but thinner in single-name equities than in indices and
futures.

**Why it is here.** Not because it is expected to be the better strategy, but
because it has a **genuinely different shape**: absolute rather than relative, with
a stop and a target and a bracket exit. If `Setup` and the scoring machinery cannot
express both shapes cleanly, the abstraction is wrong, and it is much better to
discover that in M2 than in M5.

**Allowed regimes:** per-symbol `{TREND_UP, TREND_DOWN}`, and market regime
`{RISK_ON, NEUTRAL}`.

### Rules

```python
class DonchianBreakout:
    strategy_id = "donchian_breakout_v1"
    allowed_regimes = frozenset({Regime.TREND_UP, Regime.TREND_DOWN})
    allowed_market_regimes = frozenset({MarketRegime.RISK_ON, MarketRegime.NEUTRAL})
    required_warmup_bars = 260

    entry_buffer_atr: float = 0.10      # breakout must clear the channel by this
    stop_atr: float = 3.0               # stop distance in ATR, and the sizing unit
    target_rr: float = 2.0              # target = target_rr * stop distance
    max_hold_bars: int = 40             # ~2 months
    min_ema_spread_atr: float = 0.20    # structural trend confirmation

    def detect(self, row: FeatureRow) -> Setup | None:
        if not row.is_warm:
            return None

        # LONG
        if row.regime is Regime.TREND_UP:
            if row.ema_spread_atr < self.min_ema_spread_atr:
                return None
            breakout_level = row.donchian_high_55 + self.entry_buffer_atr * row.atr_14
            if row.close <= breakout_level:
                return None
            stop = row.close - self.stop_atr * row.atr_14
            target = row.close + self.target_rr * self.stop_atr * row.atr_14
            return Setup(
                asset_id=row.asset_id, symbol=row.symbol, ts=row.ts,
                strategy_id=self.strategy_id,
                direction=Direction.LONG, regime=row.regime,
                reference_price=row.close, stop_price=stop, target_price=target,
                max_hold_bars=self.max_hold_bars,
                trigger_note="close>dc_high_55+buf",
            )

        # SHORT — exact mirror
        if row.regime is Regime.TREND_DOWN:
            if row.ema_spread_atr > -self.min_ema_spread_atr:
                return None
            breakdown_level = row.donchian_low_55 - self.entry_buffer_atr * row.atr_14
            if row.close >= breakdown_level:
                return None
            stop = row.close + self.stop_atr * row.atr_14
            target = row.close - self.target_rr * self.stop_atr * row.atr_14
            return Setup(..., direction=Direction.SHORT,
                         trigger_note="close<dc_low_55-buf")

        return None
```

### Parameter count: 5

`entry_buffer_atr`, `stop_atr`, `target_rr`, `max_hold_bars`,
`min_ema_spread_atr`. That is the budget. Adding a sixth requires deleting one or
justifying it against the trial budget in
[`12-RESEARCH_PROTOCOL.md`](12-RESEARCH_PROTOCOL.md).

### Design notes

- **55 sessions, not 20.** Twenty sessions is one month, which on daily equity
  bars is short enough that new extremes are common and mostly noise. Fifty-five
  is roughly a quarter and is the classic value from the managed-futures
  literature. Both `donchian_high_20` and `donchian_high_55` exist as features; the
  20-session channel is available for the M6 sensitivity run without a code change.
- `entry_buffer_atr` is 0.10 rather than crypto's 0.05, because overnight gaps mean
  a daily close just above a channel is a weaker signal — the "breakout" may be an
  entire gap that has already happened and is about to fill.
- `min_ema_spread_atr` is deliberately redundant with the regime gate. The regime
  label is a threshold on efficiency ratio and slope; this adds a *structural*
  requirement that the moving averages are actually separated. It is the one
  belt-and-braces filter in v1, and if the sensitivity analysis shows results are
  flat in it, delete it.
- **`target_rr` is 2.0, not crypto's 2.5.** With `stop_atr = 3.0` that is a 6 ATR
  target, about 11% at a 1.8% `atr_pct`. Round-trip cost of 12.5 bps is 1.1% of the
  gross target — an order of magnitude more comfortable than the crypto equivalent,
  and the reason a lower `target_rr` (which raises the hit rate) is affordable here.
- **No trailing stop in v1.** It adds a parameter and interacts with everything. M6
  experiment, own holdout budget.
- **This is not the Moskowitz–Ooi–Pedersen / CTA edge.** Time-series trend has a
  real literature, but the mechanism is *diversification across uncorrelated
  markets* (equity indices, rates, FX, commodities), not breakouts on 1,000 US
  common stocks. Donchian on AAPL, MSFT, and NVDA is three correlated bets on US
  equity beta. v1 still runs it on the same liquidity-ranked panel as momentum
  because that is the cheapest way to prove the `Setup` abstraction can express
  both shapes. It is a contrast, not a second independent premium. A 20–40 name
  liquid ETF sleeve that actually spans asset classes is an M6 candidate (§7),
  not a silent widening of v1.

---

## 4. What was dropped, and why

**`range_fade_v1` is not in v1.** It was Strategy B in the crypto design. Removed
for equities for three reasons, in order of weight:

1. **Daily-bar mean reversion in liquid equities is where the bid-ask-bounce
   illusion lives.** A close near the low of the day is disproportionately likely to
   be a print at the bid, and the next day's close is disproportionately likely to be
   at the mid. That manufactures apparent reversion that is entirely
   microstructural and entirely untradeable, and Corwin–Schultz spread estimates are
   not precise enough to net it out reliably.
2. **It is the most crowded retail strategy in existence.** Every published
   "RSI(2) mean reversion" backtest is this strategy. Whatever survived is thin.
3. Modelling it honestly needs intraday data to determine whether the reversion is
   capturable at all.

It remains a legitimate M6 experiment against its own trial budget, and if it is
run, it should be run on ETFs rather than single names — index mean reversion has a
better mechanism (liquidity provision to index rebalancing flow) and no
single-name gap risk.

---

## 5. Registry and construction

`src/scout/strategies/registry.py`

```python
STRATEGY_FACTORIES: dict[str, Callable[[Mapping[str, Any]], Strategy]] = {
    "xsec_momentum_v1": XSecMomentum.from_params,
    "donchian_breakout_v1": DonchianBreakout.from_params,
}


def build_strategies(cfg: Sequence[StrategyConfig]) -> tuple[Strategy, ...]:
    out = []
    for sc in cfg:
        if not sc.enabled:
            continue
        factory = STRATEGY_FACTORIES.get(sc.strategy_id)
        if factory is None:
            raise ScoutConfigError(
                f"unknown strategy_id {sc.strategy_id!r}; "
                f"known: {sorted(STRATEGY_FACTORIES)}"
            )
        out.append(factory(sc.params))
    return tuple(out)
```

`from_params` validates every param and raises `ScoutConfigError` on an unknown
key. Silently ignoring a misspelled parameter means you spend a day analysing a run
that used the defaults.

---

## 6. Versioning

`strategy_id` ends in `_vN`. Bump N when **any** rule or default changes.

The reason is mechanical, not stylistic: `strategy_id` is part of `BinKey`. Edge
statistics are pooled across all historical setups with the same key. If you change
`stop_atr` from 3.0 to 3.5 without bumping the version, the current decision uses
statistics computed from setups with a different stop distance, so `mean_r` no
longer describes the trade being considered and the ranking silently becomes wrong.
Bumping forces a relabel and an honest, empty starting sample.

`config_hash` (over the full resolved config) and `data_snapshot_id` are recorded
on every run and every edge table. A run whose either value does not match its edge
table's **fails at startup**. This is the tripwire that catches the mistake above
even if someone forgets to bump the version.

---

## 7. Adding a third strategy

Do this only at M6, and only after M3 has produced a holdout result for the first
two. Then:

1. Write the **hypothesis first**, in the PR description, in the form used in §2 and
   §3: what the mechanism is, and why it should persist. If you cannot write the
   second half, do not write the strategy.
2. Implement the class, ≤5 parameters, allowed regimes declared.
3. Register it, add `config/strategies/<id>.yaml`.
4. Unit test: at least one fixture where it fires and one where it must not.
5. Regenerate labels and the edge table
   (`scout label --config <cfg> --strategy <id>`, then `scout build-edge --config <cfg>`).
6. Run on the **development** period only. Record the trial in the registry.
7. If, and only if, the development result is materially positive, spend one holdout
   evaluation from the lockbox budget.

Candidates, in priority order:

| Strategy | Why it might work | Why it might not |
|---|---|---|
| **Low-volatility / quality tilt** | The most robust documented equity anomaly after momentum, and it is *negatively* correlated with momentum — genuine diversification rather than a second bet on the same thing | Fundamentally a slow, low-turnover holding; the bracket-and-bin machinery adds nothing, and it may be better implemented as a benchmark than a strategy |
| **Multi-asset ETF trend sleeve** | This is the actual TSMOM/CTA claim: 20–40 liquid ETFs spanning equity indices, rates, FX, and commodities, vol-targeted, daily or weekly. Diversification *is* the edge. Same Donchian (or 3–12m return-sign) rules, different universe. | A different data problem (ETF history, rolls, overseas calendars). Correlated with the primary sleeve whenever “risk-on” is the only regime that pays. Do not silently expand the v1 stock panel to fake this. |
| **Post-earnings-announcement drift** | Well documented, and a natural complement given that ADR-020 currently *excludes* the window | Requires earnings-surprise data, an entirely different cost profile, and it partly overlaps with momentum |
| Volatility contraction breakout | Squeeze-then-expand has a mechanical basis in option-hedging flows and stop clustering | Requires a squeeze definition — two more parameters |
| Pullback continuation | Better entry price than a breakout in the same trend; lower cost per unit of edge | Needs a "pullback depth" parameter that is easy to overfit |
| `range_fade_v1` on ETFs only | Index mean reversion has a real mechanism and no single-name gap risk | §4 |

The low-volatility tilt is the strongest *equity-factor* candidate because it is
negatively correlated with momentum. The multi-asset ETF trend sleeve is the
strongest *independent-premium* candidate, and it is the one thing a “run
trend-following as the core” analysis is actually pointing at. Neither is v1.
Adding a third momentum variant would raise the trial count without adding
independent information.

---

## 8. What is explicitly not a strategy concern

| Concern | Where it lives |
|---|---|
| "Is this asset liquid enough?" | `gates` |
| "Is the market regime right?" | `allowed_market_regimes`, checked by the engine before `detect` |
| "Is the per-symbol regime right?" | `allowed_regimes`, same |
| "Are earnings coming up?" | `gates`, `EARNINGS_IN_WINDOW` ([ADR-020](ADR/020-earnings-gate.md)) |
| "Can this be shorted?" | `gates`, `HARD_TO_BORROW` |
| "What is the probability of success?" | `scoring/edge.py` |
| "Is the expected value positive after costs?" | `scoring/rank.py` |
| "How large should the position be?" | `portfolio/sizing.py` |
| "Do we already hold something in this sector?" | `portfolio/selection.py` |
| "What does sentiment say?" | `portfolio/selection.py`, as a size multiplier |
| "When do we exit?" | The bracket derived from `Setup`, plus the time stop and the rank exit |

A strategy that needs to know any of the above is misplaced. The most common
violation is a strategy that wants to check "am I already in this position" — that
is `ALREADY_IN_POSITION` in the portfolio stage, and putting it in the strategy
makes the strategy stateful and destroys reproducibility.

---

## 9. Appendix: crypto (M7, optional)

The crypto v1 roster, retained for reference. Do not implement before an equity
holdout result exists.

| Strategy | Params |
|---|---|
| `donchian_breakout_v1_crypto` | `donchian_high_20`, `entry_buffer_atr` 0.05, `stop_atr` 2.0, `target_rr` 2.5, `max_hold_bars` 30 (4h bars), `min_ema_spread_atr` 0.20 |
| `range_fade_v1_crypto` | `stop_atr` 1.5, `target_rr` 1.2, `max_hold_bars` 18, `min_excursion_atr` 0.25, target clamped at `ema_slow` |

Note the different parameter values: tighter stops and larger `target_rr`, both
forced by crypto's 7× higher cost. The strategy ids carry a `_crypto` suffix
because the parameters differ and `strategy_id` is a bin key — sharing an id across
asset classes would pool statistics from setups that are not comparable.

Cross-sectional momentum is a plausible crypto strategy too, and the
`mom_*_xs_pct` features port directly. It would be the first thing to try in M7.
