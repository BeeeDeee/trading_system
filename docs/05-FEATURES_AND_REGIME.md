# Features and Regime

> Exact formulas. Implement these literally. Every formula here is causal:
> the value at session `t` uses only sessions with `close_time <= t`.

Module: `src/scout/features/`

**Scope: US-listed common stocks and ETFs on daily bars.** "Bars" means
**sessions** throughout, and all bar arithmetic uses `session_index`
([`04-DATA_AND_UNIVERSE.md §5`](04-DATA_AND_UNIVERSE.md#5-the-trading-calendar)).
Prices are the **adjusted** columns unless a formula says `close_raw`
([ADR-017](ADR/017-unadjusted-prices-pinned-snapshot.md)). Crypto differences:
[§12](#12-appendix-crypto-m7-optional).

---

## 1. Rules

1. **No third-party TA library.** No TA-Lib (a C build dependency that fails on
   Windows), no `pandas_ta` (unmaintained, silently changes formulas between
   versions). Every indicator here is 1–10 lines of pandas/numpy and is unit-tested
   against a hand-computed fixture. A silent formula change between library
   versions invalidates every stored edge statistic in the system.
2. **Vectorised per symbol** for per-symbol features; **vectorised per timestamp**
   for cross-sectional features (§8). Never row-by-row over the full panel.
3. **Forbidden operations**, checked by `tests/unit/test_no_lookahead.py` which
   greps the `features/` source: `.shift(-`, `.bfill`, `.fillna(method="bfill")`,
   `.interpolate`, `center=True`, `.iloc[::-1]`, `.rolling(...).apply` with a
   window that includes future rows.
4. **NaN is the correct answer during warm-up.** Never `fillna(0)`. A zero ATR
   produces a division by zero or an infinite position size; a NaN propagates and is
   caught by the `is_warm` gate.
5. **Three feature scopes**, and keeping them straight is the main structural
   discipline in this layer:

   | Scope | Computed over | Examples | Section |
   |---|---|---|---|
   | **Per-symbol** | one symbol's own history | `atr_14`, `regime`, `mom_252_skip21` | §2–§6 |
   | **Market** | the benchmark series only, then broadcast to every row at `t` | `market_regime`, `spy_dd_252`, `vix_close` | §7 |
   | **Cross-sectional** | all symbols eligible at `t` | `mom_252_xs_pct`, `vol_xs_pct` | §8 |

   There is **no context timeframe**. The crypto design used a 1d context panel
   joined onto 4h decision bars; on daily bars the context dimension is the *market*,
   which is more informative and costs no extra data
   ([ADR-016](ADR/016-daily-decision-bars.md)).

---

## 2. Base indicators

All per-symbol on daily bars. `c`, `h`, `l`, `o` are adjusted close, high, low,
open Series indexed by session close time, ascending.

### 2.1 True Range and Wilder ATR

```python
def true_range(h, l, c):
    prev_c = c.shift(1)
    return pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)

def atr_wilder(h, l, c, n=14):
    tr = true_range(h, l, c)
    # Wilder smoothing == EMA with alpha = 1/n. adjust=False is REQUIRED:
    # adjust=True produces a different (and non-recursive) series.
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
```

`atr_pct = atr_14 / c`.

ATR is the system's universal risk unit. Every price-space quantity that crosses a
symbol boundary is divided by it. That is what makes a $600 stock comparable to a
$12 stock and is a precondition for cross-sectional ranking.

**Note for equities specifically:** `true_range` includes the previous close, so it
already captures overnight gaps. That is correct and important — a stock whose
daily range is small but which gaps 3% every night is genuinely more volatile than
its intraday range suggests, and ATR reflects that. Using `h - l` alone would
understate risk on exactly the names where gap risk matters most.

Typical daily `atr_pct`: SPY ≈ 0.009, a large-cap ≈ 0.018, a mid-cap ≈ 0.030.

### 2.2 EMA and trend spread

```python
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

ema_fast = ema(c, cfg.ema_fast)          # default 20
ema_slow = ema(c, cfg.ema_slow)          # default 50
ema_spread_atr = (ema_fast - ema_slow) / atr_14
```

### 2.3 Normalised slope

```python
n = cfg.slope_lookback                   # default 20
slope_atr_20 = (c - c.shift(n)) / (atr_14 * np.sqrt(n))
```

Dividing by `atr * sqrt(n)` rather than `atr * n` is deliberate: under a random
walk, price displacement over `n` bars scales with `sqrt(n)`, so this normalisation
makes the statistic comparable across lookbacks and roughly interpretable as a
t-statistic of drift. Dividing by `n` would make longer lookbacks systematically
look flatter.

### 2.4 Donchian channels

```python
for n in (20, 55):
    donchian_high[n] = h.rolling(n, min_periods=n).max().shift(1)
    donchian_low[n]  = l.rolling(n, min_periods=n).min().shift(1)
```

**The `.shift(1)` is mandatory.** Without it, the channel at bar `t` includes bar
`t`'s own high, so `close > donchian_high` can never be true in a meaningful way
and a breakout rule becomes either impossible or trivially self-referential. This
is the single most frequent bug in breakout implementations.
`tests/unit/test_features.py::test_donchian_excludes_current_bar` asserts it.

Both lengths are computed. `donchian_breakout_v1` uses 55; the 20-session channel
exists so the M6 sensitivity run needs no code change
([`06-STRATEGIES.md §3`](06-STRATEGIES.md)).

```python
dist_to_high_atr = (donchian_high_55 - c) / atr_14
dist_to_low_atr  = (c - donchian_low_55) / atr_14
```

### 2.5 Keltner channel

```python
keltner_upper = ema_slow + cfg.keltner_k * atr_14      # default k = 2.0
keltner_lower = ema_slow - cfg.keltner_k * atr_14
```

Retained even though no v1 strategy reads it, because it is the natural envelope
for the M6 ETF mean-reversion experiment and it costs two lines.

### 2.6 Realised volatility

```python
r = np.log(c / c.shift(1))
vol_20 = r.rolling(20, min_periods=20).std() * np.sqrt(252)     # annualised
vol_60 = r.rolling(60, min_periods=60).std() * np.sqrt(252)
```

Annualised with 252, the session count, **not** 365.

### 2.7 Gap features

New for equities, and the reason several other decisions in the system exist.

```python
gap_atr = (o - c.shift(1)) / atr_14.shift(1)          # signed, in ATR units
gap_abs_mean_20 = gap_atr.abs().rolling(20, min_periods=20).mean()

# Fraction of total return variance that arrives overnight rather than intraday.
r_overnight = np.log(o / c.shift(1))
r_intraday  = np.log(c / o)
var_on  = r_overnight.rolling(60, min_periods=60).var()
var_id  = r_intraday.rolling(60, min_periods=60).var()
overnight_var_share_60 = var_on / (var_on + var_id)
```

`overnight_var_share_60` typically runs 0.4–0.7 for individual equities and 0.2–0.4
for broad ETFs. It quantifies the central problem of
[ADR-020](ADR/020-earnings-gate.md): for a stock at 0.65, two thirds of the risk
arrives when no stop order is armed.

**Recorded, not used in decisions in v1.** These three columns are the strongest
candidate for a fourth bin dimension at M6 — the hypothesis being that setups on
low-`overnight_var_share` names have tighter realised-R distributions and therefore
a higher `ev_r_lcb` for the same mean. Adding it now would take the bin count from
12 to 36 and starve the early walk-forward windows
([`07-EDGE_AND_SCORING.md`](07-EDGE_AND_SCORING.md)).

---

## 3. Momentum features

The primary strategy's inputs ([`06-STRATEGIES.md §2`](06-STRATEGIES.md)).

```python
n_mom  = cfg.mom_lookback        # 252, fixed. NOT a tunable
n_skip = cfg.mom_skip            # 21,  fixed. NOT a tunable

# 12-month total return skipping the most recent month.
mom_252_skip21 = (c.shift(n_skip) / c.shift(n_mom)) - 1.0

# Shorter horizons, recorded for the calibration diagnostic and as a candidate
# bin dimension. Not read by any v1 strategy.
mom_126_skip21 = (c.shift(n_skip) / c.shift(126)) - 1.0
mom_21         = (c / c.shift(21)) - 1.0
```

Three things here are **decisions, not parameters**, and they are documented as such
so that nobody sweeps them:

- **`c` is the adjusted close**, so this is a total return including dividends.
  Price-return momentum systematically penalises high-yield names, which is a
  spurious value tilt.
- **252 sessions.** The horizon at which cross-sectional momentum is documented most
  robustly across markets and decades.
- **Skip the most recent 21 sessions.** Short-horizon returns exhibit *reversal*, so
  including the last month contaminates the signal with the opposite effect.
  Standard in the literature. Baked in rather than exposed, because exposing it
  invites fitting.

`cfg.mom_lookback` and `cfg.mom_skip` exist in config only so that
`REQUIRED_WARMUP_BARS` can be derived from them (§10). Config validation rejects
any value other than the defaults unless `research.allow_momentum_horizon_sweep`
is explicitly set, which counts as a registered trial.

---

## 4. Kaufman Efficiency Ratio — the per-symbol regime backbone

```python
def efficiency_ratio(c, n):
    """Kaufman's Efficiency Ratio: net directional progress divided by total
    path length. Range [0, 1].

      1.0  -> every bar moved the same direction: a perfect trend
      0.0  -> price returned to its start after a long journey: pure chop
    """
    net = (c - c.shift(n)).abs()
    path = c.diff().abs().rolling(n, min_periods=n).sum()
    return (net / path).where(path > 0)
```

`efficiency_ratio_20` and `efficiency_ratio_60` on daily bars. The 60-session
version replaces the crypto design's context-timeframe ER: it provides the
"longer-horizon agreement" requirement from the *same* series, which removes the
timeframe join and the entire class of alignment bug that came with it.

### Why this and not ADX or an HMM

- **One parameter** (the lookback). ADX has three interacting smoothing parameters
  and a definition that varies between implementations.
- **Causal and non-repainting** by construction. Every term is a backward window. An
  HMM fitted on the full series repaints every historical label the moment a new bar
  arrives, which makes any backtest using it invalid, and fitting it walk-forward
  makes the labels unstable at exactly the boundaries that matter.
- **Measures the thing that matters.** A selective system cares whether directional
  movement is *efficient*, because path length is what stops out trend trades. ER is
  literally the ratio of the thing you want to the thing that hurts you.
- **Bounded and comparable** across assets and volatility regimes with no
  normalisation, unlike slope or ATR-based measures.

---

## 5. Volatility bucket

```python
def atr_percentile(atr_pct, window_bars):
    """Rank of the current atr_pct within its own trailing distribution.
    Trailing, with nothing forward. Result in [0, 1]."""
    return atr_pct.rolling(window_bars, min_periods=cfg.vol_min_periods).rank(pct=True)
```

`window_bars` = **504** sessions (≈2 years). `vol_min_periods` = **126** (≈6
months), so the percentile is available from 126 sessions rather than requiring two
full years. It is noisier early; that is acceptable and preferable to gating out
every symbol for its first two years, which would silently bias the universe toward
long-listed assets — a second-order survivorship effect.

```python
VolBucket.LOW  if atr_percentile <  0.33
VolBucket.MID  if 0.33 <= atr_percentile < 0.67
VolBucket.HIGH if atr_percentile >= 0.67
VolBucket.UNKNOWN if NaN
```

**Per-symbol percentiles, not cross-sectional.** "High volatility for KO" and "high
volatility for TSLA" are different absolute numbers and the relevant comparison is
each asset against its own history. The cross-sectional version exists separately
as `vol_xs_pct` (§8) and answers a different question.

---

## 6. The per-symbol regime classifier

```python
def classify_regime(er_20: float, er_60: float, slope_atr: float,
                    vol_pct: float, cfg: RegimeConfig) -> Regime:
    """Pure, stateless, four thresholds. No history, no smoothing, no state
    machine."""

    if any(math.isnan(x) for x in (er_20, er_60, slope_atr, vol_pct)):
        return Regime.UNKNOWN

    trending = (er_20 >= cfg.er_trend_min) and (er_60 >= cfg.er_long_trend_min)

    if trending and slope_atr >= cfg.slope_min:
        return Regime.TREND_UP
    if trending and slope_atr <= -cfg.slope_min:
        return Regime.TREND_DOWN
    if (er_20 <= cfg.er_range_max) and (vol_pct <= cfg.vol_range_max):
        return Regime.RANGE
    return Regime.CHOP
```

Defaults:

```yaml
regime:
  er_trend_min: 0.30        # 20-session ER floor for "trending"
  er_long_trend_min: 0.20   # 60-session ER must agree
  slope_min: 0.25           # |slope_atr_20| floor to assign a direction
  er_range_max: 0.15        # ER ceiling for "range"
  vol_range_max: 0.67       # ranges are not allowed in the top volatility third
```

Notes on the design:

- **CHOP is the default**, not a category you fall into by accident. Anything neither
  clearly trending nor clearly ranging is CHOP, and CHOP blocks
  `donchian_breakout_v1`. This directly implements the brief's "CHOP → usually NO
  TRADE" without a separate mechanism.
- **`xsec_momentum_v1` does not read `regime` at all.** It gates on *market* regime
  (§7) and cross-sectional rank. This is deliberate: a per-symbol trend label and a
  top-decile momentum rank measure nearly the same thing, and requiring both would
  narrow the universe without adding information.
- **`er_long_trend_min` is 0.20, lower than the crypto design's 0.25** context
  threshold, because a 60-session ER on a single equity is structurally lower than a
  20-bar ER on a daily crypto series. Mechanically the same requirement.
- **Five thresholds total** for the per-symbol regime. An HMM would have introduced a
  state count, a covariance structure, an initialisation, and a refit schedule, all
  fitted on the same data used to evaluate the strategy.
- **Label flapping is tolerable here** because the label is a strategy *gate*, not a
  signal. A flap costs a missed entry, never a wrong-way position. This is why the
  label must not be used for anything else.

### Regime label stability check (required in M3)

Report the transition matrix and the median run length per regime. If median
`TREND_UP` runs are shorter than `max_hold_bars`, the classifier is too twitchy and
`er_trend_min` needs raising. This is a diagnostic, not a parameter to optimise, and
it must be inspected once — not swept.

---

## 7. Market regime

New, and the single highest-value addition of the equity pivot. Computed **once per
timestamp** from the benchmark series in
`data/reference/benchmark_1d.parquet`, then broadcast to every symbol's row at that
timestamp.

The reason this matters more than any per-symbol feature: an equity index's
position relative to its long moving average is the most robustly documented
drawdown-reduction filter in the literature. It does not reliably improve returns;
it reliably improves the *shape* of returns, and it is the main structural defence
against momentum crashes ([ADR-019](ADR/019-cross-sectional-momentum-primary.md)).

```python
# All on SPY's adjusted close, causal.
spy_sma_200   = c_spy.rolling(200, min_periods=200).mean()
spy_above_ma  = c_spy > spy_sma_200
spy_dd_252    = 1.0 - c_spy / c_spy.rolling(252, min_periods=252).max()
spy_vol_20    = np.log(c_spy / c_spy.shift(1)).rolling(20, min_periods=20).std() * np.sqrt(252)
spy_vol_pct   = spy_vol_20.rolling(756, min_periods=252).rank(pct=True)   # own 3y history


def classify_market_regime(above_ma: bool, dd: float, vol_pct: float,
                           cfg: MarketRegimeConfig) -> MarketRegime:
    if any(x is None or (isinstance(x, float) and math.isnan(x))
           for x in (dd, vol_pct)):
        return MarketRegime.UNKNOWN

    stressed = (dd >= cfg.dd_stress) or (vol_pct >= cfg.vol_stress_pct)

    if above_ma and dd <= cfg.dd_warn and not stressed:
        return MarketRegime.RISK_ON
    if (not above_ma) and stressed:
        return MarketRegime.RISK_OFF
    return MarketRegime.NEUTRAL
```

```yaml
market_regime:
  benchmark_symbol: SPY
  sma_n: 200
  dd_warn: 0.10             # RISK_ON requires drawdown at or under this
  dd_stress: 0.15           # drawdown at or over this is stress
  vol_stress_pct: 0.80      # SPY vol above its own 80th percentile is stress
```

Design notes, each of which is load-bearing:

- **`UNKNOWN` blocks every strategy.** During the first 252 sessions of the data
  there is no market regime, so nothing trades. This is why the warm-up split runs
  to 2003 ([`04-DATA_AND_UNIVERSE.md §8`](04-DATA_AND_UNIVERSE.md#8-data-splits)).
- **Both v1 strategies allow `RISK_ON` and `NEUTRAL`, and neither allows
  `RISK_OFF`.** The regime is a gate, not a signal, and it is symmetric: it blocks
  shorts as well as longs. Allowing shorts in `RISK_OFF` would turn the regime label
  into a market-timing signal, which is a different and much weaker claim, and it
  would put the system maximally short at exactly the moments a momentum crash
  begins.
- **VIX is deliberately *not* in the regime definition**, even though it is
  available and would probably help. `^VIX` starts in 1990, `^VIX3M` in 2002, and
  `^VIX9D` in 2011. A regime label whose *definition* changes partway through the
  sample introduces a discontinuity that is nearly impossible to reason about and
  would make the pre-2011 and post-2011 results incomparable. SPY price data spans
  the whole sample. VIX columns are recorded on the feature panel for research and
  as sentiment inputs ([`10-SENTIMENT.md`](10-SENTIMENT.md)), not for the label.
- **The benchmark must not be gated.** `SPY` is loaded from
  `data/reference/`, not from the universe panel, so market regime is computed even
  in years when the universe is empty or `SPY` would fail a data-quality check.
- **Recorded on every `DecisionRecord`**, so `plots/regime_breakdown.png` can split
  performance by market regime. If the strategy's edge exists only in `RISK_ON`, that
  is worth knowing; if it exists only in `NEUTRAL`, the gate is miscalibrated.

### Required M3 diagnostic

Report the fraction of sessions in each market regime per year, and the strategy's
`ev_net_r` and realised R by regime. Two specific failure signatures:

- `RISK_ON` covering more than 85% of sessions means the gate is inert and is
  buying nothing.
- Realised R in `NEUTRAL` materially exceeding `RISK_ON` means the thresholds are
  mis-set and the gate is excluding the good periods.

---

## 8. Cross-sectional features

Computed **per timestamp, across symbols eligible at that timestamp**. This is the
one place the feature layer is not per-symbol, and it is what makes
`xsec_momentum_v1` expressible as a pure per-row strategy
([ADR-019](ADR/019-cross-sectional-momentum-primary.md)).

```python
def cross_sectional_ranks(features: pd.DataFrame,
                          snapshots: pd.DataFrame) -> pd.DataFrame:
    """Percentile ranks within the ELIGIBLE set at each ts.

    `snapshots` is the point-in-time universe table. Symbols not eligible at ts
    are excluded from the ranking population AND receive NaN, not a rank
    computed against a population they were not part of.
    """
    elig = eligible_mask(features, snapshots)          # backward-rounded lookup
    out = features.copy()
    for col, rank_col in (("mom_252_skip21", "mom_252_xs_pct"),
                          ("mom_126_skip21", "mom_126_xs_pct"),
                          ("vol_60",         "vol_xs_pct")):
        ranked = features[col].where(elig)
        out[rank_col] = ranked.groupby(level="ts").rank(pct=True, na_option="keep")
    return out
```

Four rules, all of which have a failure mode attached:

1. **Rank within the eligible set only.** Ranking against all candidates would let
   an illiquid, ineligible name occupy a decile slot and shift every other symbol's
   percentile. The strategy would then be reacting to the composition of a population
   it cannot trade.
2. **A symbol ineligible at `t` gets NaN, not a rank.** NaN fails `is_warm` and the
   gate rejects it. Assigning it a rank would make it tradeable through the back door.
3. **Ties get the average rank** (`rank`'s default). With 1,000 symbols and
   continuous returns, ties are essentially nonexistent; the behaviour is specified so
   it is deterministic rather than left to chance.
4. **The population size must be recorded.** `xs_population` goes on the
   `FeatureRow`. A percentile computed over 30 eligible symbols is not comparable to
   one computed over 1,000, and the early years of the sample will have smaller
   populations. `gates` rejects with `THIN_CROSS_SECTION` when
   `xs_population < cfg.gates.min_xs_population` (default 100), because the top
   decile of 40 names is four names and that is a concentration bet, not a
   cross-sectional one.

**Lookahead risk here is high and the test is specific.**
`tests/unit/test_no_lookahead.py::test_cross_sectional_truncation` computes the
panel over `[start, end]`, then over `[start, end - 100 sessions]`, and asserts
every cross-sectional column on the overlapping rows is **bit-identical**. A
cross-sectional statistic computed with `groupby(level="ts")` is naturally causal;
one computed with any global normalisation, standardisation, or z-score over the
full panel is not, and the difference is invisible in the output.

---

## 9. Beta to the benchmark

```python
window = cfg.beta_window          # default 90 sessions

r_sym = np.log(c / c.shift(1))
r_spy = np.log(c_spy / c_spy.shift(1))          # inner-joined on ts

cov = r_sym.rolling(window, min_periods=window // 2).cov(r_spy)
var = r_spy.rolling(window, min_periods=window // 2).var()
beta_bench_90 = cov / var
corr_bench_90 = r_sym.rolling(window, min_periods=window // 2).corr(r_spy)
```

`SPY`'s own row: `beta = 1.0`, `corr = 1.0`, set explicitly rather than computed.

Joining benchmark returns onto the symbol's timestamps must use an **inner join on
`ts`**. A reindex with forward-fill would pair a symbol's session with a stale
benchmark session and quietly bias beta toward zero — and for a halted stock, which
is exactly when you want beta to be right, it would be badly wrong.

These two columns are the entire correlation model
([ADR-007](ADR/007-cluster-caps-not-covariance.md)). `beta_bench_90` becomes the
portfolio's net-exposure measure; `corr_bench_90` is recorded for research and is
not used in decisions in v1.

**Beta matters more in equities than it did in crypto, and in the opposite
direction.** In crypto, BTC beta was a concentration measure. In equities, market
beta is what a long-biased equity system *accidentally becomes*: any long-only
equity strategy backtested over 2009–2021 shows a positive Sharpe from beta alone,
with no skill whatsoever. So `beta_bench_90` is both a risk cap and the central
diagnostic of [`12-RESEARCH_PROTOCOL.md`](12-RESEARCH_PROTOCOL.md).

---

## 10. Warm-up accounting

```python
REQUIRED_WARMUP_BARS = max(
    cfg.ema_slow,                     # 50
    cfg.atr_n,                        # 14
    cfg.donchian_n_long + 1,          # 56   (+1 for the mandatory shift)
    cfg.slope_lookback,               # 20
    cfg.er_n_long,                    # 60
    cfg.vol_lookback_long,            # 60
    cfg.beta_window // 2,             # 45
    cfg.vol_min_periods,              # 126
    cfg.mom_lookback + cfg.mom_skip,  # 273  <- binding constraint
)
# => 273 sessions, i.e. about 13 months

is_warm = bars_available >= REQUIRED_WARMUP_BARS
```

Compute this from config rather than hardcoding 273; a config change that silently
leaves warm-up too short produces features computed from three data points.

Note the binding constraint has moved from the volatility percentile (crypto: 360
4h bars) to the momentum lookback (273 sessions), which is why
`min_history_bars` in
[`04-DATA_AND_UNIVERSE.md §7.2`](04-DATA_AND_UNIVERSE.md#72-eligibility-rules)
is 400 rather than a smaller number. The margin above 273 is deliberate: it means a
symbol clearing the universe gate has warm features on the *first* session it is
eligible, so no candidate is ever silently dropped at the `is_warm` check.

**Market regime has its own, longer warm-up** — 252 sessions for `spy_dd_252` and
`spy_vol_pct`'s `min_periods`. This is not per-symbol, so it is not in the maximum
above; it gates *everything* simultaneously and is handled by the warm-up split in
the data document.

`is_warm == False` triggers the `INSUFFICIENT_HISTORY` gate. A strategy is never
called with a cold row.

---

## 11. Feature computation entry point

```python
def compute_features(
    panel: MarketPanel,
    benchmark: BenchmarkPanel,
    snapshots: pd.DataFrame,
    cfg: FeatureConfig,
) -> FeaturePanel:
    """Compute all FeatureRow columns for every (ts, asset_id) in `panel`.

    Called ONCE before the backtest loop, for the whole history. Not per bar.
    Per-bar recomputation is O(n^2) and turns a two-minute run into hours.

    Steps:
      1. Compute market-regime columns from `benchmark` (SS7). One row per ts.
      2. For each symbol: per-symbol features (SS2-SS6).
      3. Compute beta/corr against the benchmark (SS9).
      4. Compute cross-sectional ranks using `snapshots` (SS8).
      5. classify_regime row-wise (vectorise with np.select).
      6. Broadcast the market-regime columns onto every row by ts (a merge on ts,
         not merge_asof: the grids are identical by construction, and an
         exact-match merge that drops rows is a bug worth failing on).
      7. Assemble the FeaturePanel; validate the column set equals FeatureRow's
         field set EXACTLY.
    """
```

`snapshots` is passed in as data. The features module does **not** import the
universe module; that would couple two same-layer modules and break the layering
test in [`16-TESTING.md`](16-TESTING.md).

Step 6 is an exact merge on `ts` and asserts no row is lost. The crypto design used
`merge_asof` for the timeframe join and needed a `ctx_age_bars >= 0` production
assertion to guard it; that entire class of alignment bug is gone, because there is
only one timeframe. The replacement assertion is simpler and stronger:

```python
assert len(merged) == len(decision_features), "market regime broadcast lost rows"
assert merged["market_regime"].notna().all() | ~merged["is_warm"], \
       "warm row with no market regime"
```

Step 7's validation is a one-line `set` comparison against
`dataclasses.fields(FeatureRow)` and it prevents the most annoying class of bug in
this layer: a feature that a strategy reads, that was never computed, and that
therefore silently arrives as NaN or `KeyError` 4,000 bars into a run.

---

## 12. Adding a feature

1. Add the field to `FeatureRow` in `domain/features.py`, with a unit suffix.
2. Add the computation in `features/indicators.py` (per-symbol),
   `features/market.py` (benchmark), or `features/cross_sectional.py` (panel-wide)
   as a standalone pure function.
3. Add a unit test with a hand-computed expected value for a 10-bar fixture.
4. Add the column to `compute_features`.
5. Confirm `REQUIRED_WARMUP_BARS` still covers it.
6. Run `tests/unit/test_no_lookahead.py`. For a cross-sectional feature, confirm
   the truncation test in §8 covers the new column.

**Before adding it, answer in the PR description: which strategy or which bin
dimension consumes this, and why.** Unconsumed features are not free — they enlarge
the search space that later tempts you into trying "just one more variant", which is
how the trial budget in
[`12-RESEARCH_PROTOCOL.md`](12-RESEARCH_PROTOCOL.md) gets spent without a single
line of strategy code changing.

---

## 13. Appendix: crypto (M7, optional)

Differences for the optional crypto sleeve. Do not implement before an equity
holdout result exists.

- **Two timeframes**, decision 4h and context 1d
  ([ADR-010](ADR/010-4h-decision-timeframe.md)), joined with
  `merge_asof(direction="backward")` on the context bar's **close** timestamp, with
  a production assertion that `ctx_age_bars >= 0`. Joining on the context bar's
  *open* time exposes 24 hours of future information.
- **`efficiency_ratio_ctx`** on the 1d panel replaces `efficiency_ratio_60`, and
  `regime.er_ctx_trend_min` (0.25) replaces `er_long_trend_min`.
- **Market regime does not apply.** There is no crypto equivalent of the 200-day
  index filter with comparable evidence. BTC's own 200-bar MA is the obvious
  candidate and is a *much* weaker claim; if the crypto sleeve is built, it should be
  tested as a registered trial rather than assumed.
- **Beta reference is BTC**, `beta_bench_90` → `beta_btc_90`,
  `beta_window` 90 4h bars (15 days).
- **Volatility percentile window** 2190 4h bars (≈1 year), `vol_min_periods` 360.
- **No gap features.** Price is continuous, `overnight_var_share` is undefined, and
  `true_range`'s previous-close terms rarely bind.
- **Donchian 20 only**, not 55.
- **Warm-up binding constraint** is `vol_min_periods` at 360 bars, unless the
  cross-sectional momentum features are ported, in which case it is
  `mom_lookback + mom_skip` as here.
