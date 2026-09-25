# ADR-015: US equities and ETFs first; crypto last and optional

**Status:** Accepted — supersedes the crypto-first scope in ADR-001 through ADR-014

## Context

The original scope was crypto USDT perpetuals with equities as a later extension.
Reassessment on the merits favours the reverse order, and by a wide margin.

### Cost, which dominates everything else

| | Crypto perps | US equities |
|---|---|---|
| Commission | 5 bps taker × 2 | $0, or ~$0.0035/share at IBKR Pro |
| Spread | 1–8 bps | 0.5–3 bps on liquid names |
| Funding | ~10 bps per 10-day hold, persistently against longs | none |
| Borrow | n/a (funding subsumes) | ~30 bps/yr ETB ≈ 1 bp per 10-day hold |
| Slippage | 4h drift between decision and fill | opening auction: single clearing price, no spread crossed |
| **`cost_r` (worked examples)** | **0.159 R** | **0.023 R** |

Roughly a **7× cost reduction**. Required gross edge falls from about 0.20 R to
about 0.05 R. Full derivations: [`08-COSTS.md §3`](../08-COSTS.md#3-worked-examples).

### Evidence base

Cross-sectional equity momentum has thirty years of out-of-sample literature
across markets, periods, and asset classes, and it survived publication. Crypto
trend-following has roughly eight noisy years, most of it inside one enormous
bull market — the sample cannot distinguish a risk premium from a bull market.

You are no longer hunting for something unknown; you are attempting to capture
something documented, net of costs, at retail scale. That is a materially easier
problem and a much better-posed question.

### History and regime variety

25+ years of daily equity data spanning the dot-com bust, 2008, 2011, 2015–16,
2018, COVID, 2022, and 2023–25, versus six years of crypto. More independent
regimes is worth more than more bars.

### Capacity

A retail account in liquid US equities has effectively unlimited capacity. Nothing
about the result depends on order size, which removes an entire class of doubt.

## Decision

**Primary and only scope through M6: US-listed common stocks and ETFs.**

- Decision timeframe: **daily bars** (ADR-016).
- Universe: liquidity-ranked, point-in-time, delisted included (ADR-018).
- Prices: unadjusted plus a corporate-actions table, pinned snapshot (ADR-017).
- Primary strategy: cross-sectional momentum (ADR-019).
- Market regime from a broad index, per-symbol regime as a feature.

**Crypto becomes M7 and is optional.** It reuses the same engine, the same
scoring, the same portfolio layer, and the same research protocol. The crypto
work already documented — perpetual funding, the 4h clock, the crypto cluster map,
funding-skew sentiment — is retained in the docs, marked as M7, and is *not*
deleted. If the equity result is positive, crypto becomes a diversification
question. If it is negative, crypto will not rescue it.

## Consequences

New complexity the crypto version did not have, and none of it is optional:

| New concern | Where handled |
|---|---|
| Corporate actions: splits, dividends, spinoffs, ticker changes | ADR-017, [`04-DATA_AND_UNIVERSE.md §4`](../04-DATA_AND_UNIVERSE.md) |
| Trading calendar: holidays, half days, DST | `exchange_calendars` dependency, XNYS |
| Overnight gaps dominate daily variance | ADR-020 (earnings gate), wider stops, gap-through-stop modelling |
| Earnings events | ADR-020 |
| Short borrow availability and cost | [`08-COSTS.md §2.5`](../08-COSTS.md), hard-to-borrow gate |
| Point-in-time index membership | Avoided entirely by ADR-018 |
| Paid data | ~$60–70/month. Genuinely required; see [`04-DATA_AND_UNIVERSE.md §1`](../04-DATA_AND_UNIVERSE.md#1-data-source-this-decision-is-load-bearing) |

Complexity removed: perpetual funding mechanics, exchange-native bracket order
quirks, 24/7 operation, liquidation cascades, and the crypto delisting tail.

The hardest genuinely-new problem is that **overnight gaps degrade the R-based
risk model**. In crypto, price is continuous and a 1 R stop loses about 1 R. In
equities an earnings gap loses 3–5 R, routinely, not as a tail event. The
mitigations are the earnings gate, wider stops (3 ATR rather than 2), and honest
gap modelling in `SimBroker`. It remains the weakest joint in the design and is
called out as such.

## Alternatives rejected

**Crypto first, equities later** — the original plan. Rejected on the cost
arithmetic above. Spending six weeks proving that a 0.20 R edge does not exist,
before testing a market that needs 0.05 R, is the wrong order.

**Both simultaneously.** Doubles the data engineering and the trial budget before
either has an answer.

**Equities via a free data source (yfinance).** Survivorship-biased, retroactively
revised adjusted prices, no reliable delisting record. Acceptable for a
prototype, invalid for a result. See
[`04-DATA_AND_UNIVERSE.md §1`](../04-DATA_AND_UNIVERSE.md#1-data-source-this-decision-is-load-bearing).

**Futures instead.** Excellent cost structure and a genuinely documented
time-series momentum premium, but roughly 30 instruments means no
cross-sectional breadth, and continuous-contract roll construction is its own
data project.

## Revisit trigger

M7, for crypto as a diversifying sleeve — and only after an equity holdout result
exists.
