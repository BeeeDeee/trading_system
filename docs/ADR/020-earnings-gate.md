# ADR-020: Exclude positions whose holding window contains an earnings date

**Status:** Accepted

## Context

This is the largest genuinely-new risk in the equity pivot, and it directly
attacks the R-based risk framework.

In crypto, price is continuous. A 1 R stop loses approximately 1 R, and a loss of
3 R is a rare liquidity event. In equities, a single-name earnings report produces
overnight gaps of 5–20% routinely. A stop does not help; the stop is jumped. With
a 3 ATR stop on a stock with a 1.8% daily ATR — a 5.4% stop distance — a 15%
earnings gap is a **2.8 R loss on one position**, and it is not a tail event. It is
a scheduled quarterly occurrence with an approximately known date.

Three separate problems:

1. **It destroys the risk model's calibration.** Portfolio heat, position sizing,
   and the daily-loss circuit breaker are all denominated in R, and all assume the
   loss distribution is bounded near 1 R on the downside. Earnings gaps break that
   assumption in a way that no amount of position sizing repairs.
2. **It is not the edge being tested.** Holding through earnings is a near-symmetric
   bet on a binary information event. Whatever a momentum or breakout signal
   predicts, it does not predict earnings surprises. Including these events adds
   large, roughly zero-mean variance, which makes the *real* effect harder to detect,
   not easier.
3. **It contaminates the bin statistics.** A bin whose realised R distribution is a
   mixture of ordinary outcomes and earnings-gap outcomes has a wide dispersion, so
   `ev_r_lcb` is penalised heavily and the strategy is rejected for a reason that has
   nothing to do with its signal.

## Decision

**No new position may be opened if a confirmed or estimated earnings date falls
within the expected holding window.**

```
earnings_in_window(...) -> bool   # internal helper of evaluate_gates; not a public API
```

True if an earnings date for `symbol` falls in `(t, t + max_hold_bars]`, using
`max_hold_bars` from the strategy under consideration, not the realised hold
and not a separate config knob. There is no `earnings_blackout_sessions`.

Implemented as gate `EARNINGS_IN_WINDOW` inside `evaluate_gates`, in the fixed
gate order ([`14-CONFIG.md` §5](../14-CONFIG.md#5-gate-order)).

**An open position whose earnings date arrives is closed at the next open before
the report**, with `exit_reason="EARNINGS"`. This can only happen when the earnings
date is revised into the window after entry, which is common — companies confirm
dates roughly three weeks ahead.

Applies to individual equities. **ETFs are exempt**, since a diversified fund has
no single earnings event.

### Point-in-time discipline

Earnings dates are the hardest point-in-time problem in the project, because
**announcement dates are themselves revised**. A vendor's current table says
"AAPL reported 2019-07-30". What matters for a decision on 2019-07-10 is the date
*expected* on 2019-07-10, which may have been an estimate.

The rule:

- Store both `date_confirmed` and `date_estimated`, plus `as_of` (when the row
  became known) where the vendor provides it.
- If `as_of` / `available_ts` is available, use only rows with `available_ts <= t`.
- If it is not, use the **conservative fallback** (project rule 4): treat the
  earnings date as a window of `± EARNINGS_UNCERTAINTY_SESSIONS` (constant **4**,
  sessions on the XNYS grid — not a YAML knob) around the known date, and gate
  if the *window* intersects the holding period. A missing earnings date is
  treated as **"earnings unknown, therefore blocked"** for individual equities —
  never as "no earnings, therefore allowed."

The conservative fallback means a symbol with no earnings data is untradable. That
is intentional and it is the correct direction of error.

## Consequences

- **Roughly 8–15% of candidate-days are gated out** for individual equities. With
  21-session momentum holds and quarterly earnings, a 21-day window covers about
  33% of a 63-day earnings cycle, so the gate is more binding than it first
  appears — it may exclude a third of momentum candidate-days. This is a real cost
  in opportunity count and it must be measured, not assumed: `funnel.csv` reports
  the `EARNINGS_IN_WINDOW` rejection count and M3 reports it explicitly.
- **The gate may remove real edge.** Post-earnings-announcement drift is a
  documented anomaly, and momentum partly *is* PEAD. Excluding earnings windows may
  therefore discard part of the effect being measured. This is a genuine trade-off,
  not a free win.
  - M3 runs a **paired sensitivity**: gate on and gate off, same config otherwise,
    reported side by side. If the gate-off variant has materially higher `ev_net_r`
    *and* a comparable worst-case single-trade loss, the decision is revisited with
    the evidence in hand.
  - The gate stays on by default because the failure mode it prevents — an
    uncontrolled 3 R single-position loss inside a system whose every risk limit
    assumes 1 R — is the kind that ends accounts.
- Earnings data becomes a required dependency. Sharadar `ACTIONS`/`SF1` and
  Norgate both supply it. A free approximation from yfinance exists and is
  unreliable for history.
- ETF exemption means the ETF sleeve has systematically more available
  candidate-days than the equity sleeve, which slightly tilts the portfolio toward
  ETFs. Bounded by the ETF cluster cap.

## Alternatives rejected

**Hold through earnings and let position sizing absorb it.** Requires sizing every
position for a 3 R worst case, which cuts size by two thirds and destroys expectancy
against a 0.023 R cost base. Also leaves the daily circuit breaker mis-calibrated.

**Reduce size rather than block, e.g. one third size into earnings.** Keeps some
PEAD exposure with bounded damage, and is a defensible middle path. Rejected for v1
only because it adds a size-modifier parameter and a second code path before any
baseline exists. It is the first thing to try if the paired sensitivity shows the
gate is expensive.

**Trade earnings deliberately as a third strategy.** A legitimate and
well-documented strategy family, and entirely out of scope. It needs options data,
implied-move data, and a completely different cost model.

**Ignore earnings entirely.** What most retail backtests do. It produces a system
whose realised drawdowns are two to three times the backtested ones, discovered
live.

## Revisit trigger

M3, with the paired sensitivity result in hand. If the gate costs more than about
0.02 R of mean `ev_net_r`, evaluate the reduced-size variant.
