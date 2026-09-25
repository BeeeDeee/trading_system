# ADR-018: Universe by causal liquidity rank, not index membership

**Status:** Accepted — refines [ADR-005](005-point-in-time-universe.md) for equities

## Context

The obvious equity universe is an index: S&P 500, Russell 1000, Nasdaq 100. It is
also a trap.

**Index membership is a forward-looking signal.** A stock is added to the S&P 500
*after* it has performed well and grown large. A backtest run on today's S&P 500
constituents buys those stocks in 2010 *because* they were destined to be added.
This is not ordinary survivorship bias — it is a direct leak of future
performance into the selection criterion, and it inflates a momentum backtest more
than almost anything else you could do wrong.

The correct fix is historical point-in-time constituent lists. The S&P 500 turns
over roughly 20–25 names a year, so 25 years of history requires about 550 add and
delete events with exact effective dates. That data exists (Norgate includes it;
Wikipedia's change log is scrapeable and imperfect) but it is an extra dependency,
an extra correctness surface, and an extra thing to get subtly wrong.

There is also a second issue: **index membership is not what we care about**. The
system needs symbols that are liquid enough to trade at modelled cost. Index
membership is a noisy proxy for that.

## Decision

**Define the universe by a causally-computed liquidity rank. Do not use index
membership anywhere.**

At each rebalance date `t`, a symbol is eligible if all of:

| Criterion | Default | Why |
|---|---|---|
| Median dollar volume over the trailing 60 sessions ranks in the top `universe_size` | `1000` | The actual thing we care about |
| Median dollar volume ≥ `min_adv_usd` | `$5,000,000` | Absolute floor, so a thin decade does not admit thin names |
| Unadjusted close ≥ `min_price_usd` | `$5.00` | Excludes penny stocks, sub-penny tick behaviour, and reverse-split artifacts. Uses `close_raw` per ADR-017 |
| Has at least `min_history_bars` of history | `400` | Feature warm-up plus margin |
| Not flagged as an ADR/OTC/preferred/warrant/unit | — | Different microstructure and cost behaviour |
| Listed on NYSE, NASDAQ, or ARCA | — | Cost tiers are calibrated for these |

ETFs are admitted under the same rules, flagged `is_etf=True`, and are subject to
a separate cluster cap (a sector-ETF sleeve and 40 constituent stocks are one
bet, not 41).

Snapshots are recomputed **monthly**, on the first session of the month, and
written to `data/universe/equity/snapshot_<YYYY-MM-DD>.parquet`. Between
rebalances the snapshot is held fixed — a symbol dropping below the threshold
mid-month does not force an exit, but a new position cannot be opened in it.

**Delisted symbols are included** for every date on which they qualified. The
candidate file `config/universe_candidates.txt` is append-only (project rule 6),
seeded from the vendor's *full* ticker list including delisted, not from a current
index.

## Consequences

- No point-in-time constituent data needed. One entire data dependency and its
  correctness surface removed.
- The universe is a rolling computation from data already required, so it is
  reproducible from the pinned snapshot (ADR-017) with no extra vendor coupling.
- The universe is **larger and more diverse** than an index: about 1,000 names
  spanning large and mid cap. Cross-sectional breadth is a first-order input to
  momentum, so this is an advantage.
- The universe drifts with the market. In 2000 the top 1,000 by dollar volume was
  dominated by technology; in 2008 by financials. That drift is real and is
  precisely what a point-in-time universe should reproduce.
- **Liquidity rank is itself mildly momentum-correlated** — a stock's dollar volume
  rises with its price and its attention. A momentum strategy on a
  liquidity-ranked universe therefore has a slight selection tilt toward recent
  winners. This is honest, because the tilt is computed only from past data, but it
  must be measured: M3 includes a sensitivity run at `universe_size` 500 and 2000.
  If the result depends strongly on `universe_size`, that is a red flag.
- Delisting handling: a position in a symbol that delists is closed at the last
  available close with `exit_reason="DELISTED"`, and the trade is recorded. Not
  dropped, not filled forward.

## Alternatives rejected

**Current S&P 500 constituents.** The trap described above. Would produce a
beautiful, meaningless backtest.

**Point-in-time S&P 500 constituents.** Correct, and a reasonable robustness check
in M6, but an unnecessary dependency when liquidity rank both avoids the problem
and better expresses the actual requirement.

**All listed symbols with no size filter.** Admits thousands of illiquid names
where the cost model is not calibrated and where the apparent edge is
mostly bid-ask bounce. The dominant source of fake mean-reversion edge in
published retail backtests.

**Market-capitalisation rank.** Requires point-in-time share counts, which require
their own corporate-action handling, to select for something we do not want
(size) instead of what we do want (tradability).

## Revisit trigger

M6: run the S&P 500 point-in-time universe as a robustness check, if the vendor
provides constituent history. Agreement between the two is reassuring; a large
divergence localises the liquidity-tilt concern above.
