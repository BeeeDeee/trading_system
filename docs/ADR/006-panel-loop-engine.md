# ADR-006: Single time-ordered panel loop

**Status:** Accepted

## Context

The system is cross-sectional: at each bar it compares all eligible assets, ranks
them, and selects a few subject to portfolio constraints.

The instinctive backtest implementation is a per-symbol loop — it is how nearly
every tutorial, and most retail backtesting libraries, are structured. A
per-symbol loop **structurally cannot** express this system: it cannot compare
assets at an instant, cannot enforce a portfolio cap, and cannot know what else
was available at the moment it decided.

Discovering this after writing the engine means rewriting the engine, and it is
usually discovered when portfolio limits are added — that is, late.

## Decision

The outer loop iterates **timestamps**. All per-symbol work happens inside a
timestamp.

```python
for ts in panel.timestamps:      # OUTER
    for symbol in eligible_at(ts):
        ...
```

Enforced by `tests/unit/test_engine_structure.py`, which parses the AST of
`BacktestEngine.run` and asserts the outermost `for` iterates timestamps.

Supporting requirements:

- Data is one long-format panel (`symbol`, `ts`, OHLCV), loaded once before the
  loop.
- `MarketPanel.as_of(ts)` is the single causality primitive, backed by a
  precomputed `searchsorted` boundary index — O(log n) plus a view, not a boolean
  mask per bar.
- Features are computed **once** for the entire history before the loop, not per
  bar.
- Exits are applied before entries within each timestamp.

## Consequences

- Cross-sectional ranking and portfolio caps are natural rather than bolted on.
- Determinism is straightforward: one ordered loop, no concurrency.
- 13,000 bars × 120 symbols in under 90 seconds, with features precomputed.
- The same loop shape serves live: only the driver changes, from
  `for ts in timestamps` to a scheduler waking after each bar close.
- Memory: the full panel is roughly 83 MB for 120 symbols × 6 years of 4h bars.
  Fine.
- Cost: cannot use off-the-shelf per-symbol vectorised backtesters. Not a loss —
  none of them model portfolio-level constraints correctly anyway.

## Alternatives rejected

**Per-symbol loop with post-hoc portfolio filtering.** Requires knowing which
trades were "available" at each instant, which requires the panel loop. It is the
panel loop with extra steps and a subtle bug where a filtered-out trade still
consumed a slot.

**`vectorbt` or `backtrader`.** Both are per-symbol-first. Expressing portfolio
heat and cluster caps in either means fighting the framework, and neither models
the conservative intrabar tie rule the way this system requires.

**Event-driven architecture with a queue.** Adds nondeterminism and complexity for
a single-threaded, bar-driven system. The bar close *is* the only event.

**Boolean-mask filtering per bar.** O(n) per bar; turns a 30-second run into
20 minutes at 13,000 bars.

## Revisit trigger

None foreseen. This is a foundational constraint.
