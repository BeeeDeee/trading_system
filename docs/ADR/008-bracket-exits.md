# ADR-008: Broker-native bracket exits; no pipeline-driven stops

**Status:** Accepted — carries forward the previous project's ADR-008. Amended by
§"Equity instantiation" below.

## Context

The decision pipeline runs once per bar close — daily for equities, 4h for crypto.
If exits were also pipeline-driven, a position could sit unprotected for a full
session or longer, and a great deal of ground is covered in that time.

Separately, if the process dies, any protection that lives in the process dies
with it.

## Decision

Every position is opened as a **bracket**: entry, protective stop, and (optionally)
take-profit submitted together as one logical unit. The stop and target are
**broker-native conditional orders**, both reduce-only or equivalent.

- `Broker.submit_bracket(entry, stop, target)` is the only entry path.
- If the stop cannot be placed, the entry is **cancelled or closed immediately**.
  A naked position is never acceptable, and the Protocol contract states this.
- The only pipeline-driven exits are the **time stop** at `max_hold_bars` and the
  **rank exit** for `xsec_momentum_v1`, both evaluated at the start of each cycle
  before new entries are considered.
- `target` may be `None` ([ADR-019](019-cross-sectional-momentum-primary.md)); the
  stop may not.
- No trailing stops, no partial exits, no stop-to-breakeven, no pyramiding in v1.

## Equity instantiation

The design carries over, with three differences that matter:

1. **`target=None` is now a normal case.** `xsec_momentum_v1` submits an entry plus
   a 5 ATR disaster stop and no target. A one-legged bracket. The
   "stop must be placed or the entry is cancelled" rule is unchanged and is the
   only part that was ever load-bearing.
2. **The stop is a GTC stop order held at the broker across sessions**, so it is
   armed during regular hours the next day. It is **not** armed overnight or
   pre-market, which is the whole reason [ADR-020](020-earnings-gate.md) exists.
   The honest statement is: a broker-native stop protects against intraday
   deterioration and provides *no* protection against gaps. Everything about the
   equity risk model follows from accepting that rather than pretending otherwise.
3. **Gap-through fills are the common case, not the tail.** In crypto a stop
   gapped through was a liquidity event; in equities every holding period contains
   overnight gaps, so `SimBroker` must model gap-through fills correctly or the
   backtest is materially optimistic. See
   [`11-BACKTEST_ENGINE.md`](../11-BACKTEST_ENGINE.md).

## Consequences

- A daily decision clock is safe against intraday moves, because protection is
  continuous during the session. It is *not* safe against gaps, and no
  bracket design would be.
- **If the process dies, protection survives.** This is the single most important
  operational property of the design.
- Exits do not consume a pipeline cycle, so the loop stays simple.
- The `Setup` fully determines the exit, which is what makes the labeling pass
  match live behaviour exactly. That correspondence is what makes the edge
  statistics meaningful.
- Cost: the exit rule is fixed at entry. No adaptation to what happens next.
  Accepted for v1 — each adaptive exit variant adds parameters *and* invalidates
  the stored edge statistics, requiring a full relabel and a strategy version
  bump.
- Cost: stop orders slip badly in a fast market, and gap through entirely
  overnight. Modelled explicitly in `SimBroker`: a bar that opens beyond the stop
  fills at the open, producing a loss greater than 1 R. Simulations that cap every
  loss at exactly 1 R systematically understate tail risk and overstate Sharpe —
  and for daily equity bars the understatement is large, not marginal.

## Alternatives rejected

**Pipeline-driven stops evaluated each bar.** A full session of unprotected
exposure, and no protection at all if the process is down. Non-starter.

**A separate fast loop (1m) purely for exit monitoring.** Adds a second clock, a
second data feed, and a class of race condition between the loops, to replicate
something the broker already does correctly and for free. For equities it would
also do nothing about the gap risk that actually dominates.

**Trailing stops in v1.** Adds at least one parameter, interacts with
`max_hold_bars` and `target_rr`, and changes the label distribution so the edge
statistics must be rebuilt. It is a legitimate M4 experiment with its own trial
budget, and it is plausibly a real improvement — but not before there is a
baseline to improve on.

**Time stop as a broker order.** Brokers do not offer "close after 21 sessions".
It must be pipeline-driven, which is acceptable because a time stop is not
protection — it is capital recycling, and being one session late costs almost
nothing. The same applies to the momentum rank exit.

## Revisit trigger

M4, for one exit-rule variant, evaluated against the trial budget.
