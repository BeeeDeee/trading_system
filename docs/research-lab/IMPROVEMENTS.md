# Research Lab – what limits hypothesis generation, and possible improvements

Status 2026-10-05, after the first 4 hypotheses (all rejected at G1). Ordered by expected value per cost.
Nothing here is decided; each item becomes a row in the PLAN decision log when the owner picks it.

## 1. Where hypotheses come from today

- **Source:** the Scout (Opus 5.5, headless) proposes from its own knowledge of the literature and markets
  (pretraining), plus WebSearch to check references. One card per run.
- **What it sees:** the catalog (which datasets exist and which can be loaded), descriptive fact sheets of the
  dev period (instruments, volatility, liquidity, correlations; no returns), the 15 earlier studies and their
  lessons, the registry (every hypothesis with status and reason code of death), the gate thresholds.
- **What it does not see (by design):** data, returns, and the gate metrics of earlier hypotheses. It knows
  *that* H-0003 died at G1 (`g1_sharpe_excess`), not by how much.

## 2. What constrains it now (honest list)

| Constraint | Kind | Effect seen so far |
|---|---|---|
| **Only two loadable datasets** (Sharadar SFP ETFs, Binance spot daily) | data / framework code | 3 of 4 hypotheses used SPY/IEF; ideas needing stocks, perps, funding, macro, options, hourly data are blocked |
| Single stocks (Sharadar SEP) have data on disk but **no loader**; no forward data (Sharadar not renewed) | framework + data | cross-sectional equity ideas cannot be tested at all; even with a loader they could not reach G5 |
| Daily bars only, fills at the next open, gross ≤ 1, no leverage, no options | engine | intraday, market-making, option-premium, levered carry ideas are out of scope |
| Data-blind Scout (decision Q5) | deliberate | it cannot look for patterns in data (which is the point: no untracked trials), but it also learns nothing from near misses |
| Contaminated history: all data up to 2026 is in the model's pretraining and in 15 earlier studies | unfixable | only paper forward data (G5) is clean evidence; famous anomalies get a +20 trial literature prior |
| Strict gates (benchmark + 0.10 Sharpe, costs ×2, neighbors, random entry, DSR) | deliberate | most ideas die at G1; this is the intended base rate, but it makes the funnel slow |
| Budget: Claude Pro, plan 8 runs/day; a Scout run ≈ 3 min / ~0.3 M input tokens, Builder ≈ 5 min / ~0.5 M | money | ~2 new hypotheses per day under the plan |
| Machine: 3 cores, 3.9 GB RAM, ~8 GB free disk | money | gate runs serialized; no large panels (hourly, single-stock features); Archivist ingest capped |
| Scout diversity: no explicit novelty pressure beyond a 0.9 text-similarity duplicate check within a family | framework | convergence on the cheapest instruments (SPY/IEF) |
| Warm-up: the data view starts at the latest `period` start of a card | framework | an instrument cannot have a longer history for indicators than the traded pair |
| No Librarian yet: lessons are not distilled back into the Scout's context | framework (step 3) | each Scout run starts from the same context |

Nothing *prevents* new hypotheses: there is no cap other than the run budget and the duplicate check.

## 3. Combining hypotheses

Not supported yet, on purpose, because it is the easiest way to mine the dev period:
choosing which dead or half-dead components to combine after seeing their gate results is selection on dev
data that no trial counter sees. A safe version:

- a **combination card** type that references parent hypotheses and fixes weights or a rule *ex ante*
  (equal weight, inverse volatility, fixed risk budget; no optimization on dev returns),
- its family is the union of the parents' families and its trial count includes all parent trials,
- components must each have passed at least G2 (robust on their own), so a combination cannot resurrect noise,
- G2 for a combination adds a leave-one-component-out check (no single component carries it).

Worth building once at least two hypotheses survive G2; with today's base rate (0 of 4 past G1) it would
have nothing to combine.

## 4. Improvements without money (framework work)

1. **Loaders for data already on disk:** Sharadar SEP single stocks (LIQ-N, S&P 500 PIT universes; G1–G4 only,
   no G5), Binance perp + funding + mark (carry, basis, funding-momentum ideas), Binance 1h (intraday features
   aggregated to daily decisions), FRED macro via the generic loader of the Archivist.
2. **Scout diversity:** rotating task briefs per run (asset class, mechanism class: risk premium, behavioral,
   flow/structural, cross-asset), a novelty requirement against the registry (different instruments *or*
   mechanism), and a cap of one open hypothesis per instrument set.
3. **Librarian lessons:** after each death, a short structured lesson (mechanism, data, stage, which check, in
   words, not numbers) appended to the Scout's context. Trade-off: feeding gate *metrics* back is adaptive
   search on dev data; it must be counted in the family's trials, so lessons stay qualitative.
4. **Warm-up per dataset:** allow indicator history before the trading start (the view starts at the earliest
   period, trading is masked until the latest).
5. **Combination cards** as in §3, when there is something to combine.
6. **Intraday engine** (hourly decisions and fills) for crypto: the hourly data is on disk, but an hourly
   panel for 100+ pairs over 9 years needs ~1 GB per float matrix and a cost model for hourly taker fills;
   realistic only with more RAM (see §5). Until then the hourly timeframe enters as daily features.
7. **Faster funnel:** run G1 for several hypotheses in one tick, cache panels across hypotheses.

## 5. Improvements that cost money (owner decisions)

| Item | What it unlocks | Notes |
|---|---|---|
| **More LLM capacity** (Claude Max plan, or an API key with a monthly cap) | 3-10× more Scout/Builder/Skeptic runs per day | the strongest lever on throughput; quality is limited by data, not by run count |
| **VPS upgrade** (8-16 GB RAM, 4-8 cores, 100+ GB disk) | hourly panels, single-stock feature panels, parallel gate runs, bigger Archivist datasets | today's 3.9 GB forces one gate run at a time and blocks hourly/single-stock work |
| **Equity data with forward updates** (Sharadar renewal, or a cheaper EOD source such as Tiingo/EODHD/Polygon for prices + SEC EDGAR for fundamentals) | single-stock hypotheses can reach G5 (paper) | Q3 decided no renewal; free sources lack point-in-time delisting guarantees |
| **Intraday / microstructure data** (Binance 1m/trades free; paid: Databento, Polygon, Kaiko) | intraday mechanisms, realistic fill/cost models | needs the engine to support intraday decisions |
| **Options data** (Deribit public API free for crypto; US equity options are expensive: OptionMetrics, ORATS, CBOE DataShop) | volatility-risk-premium, skew, put-call ideas | options also need an engine that prices option positions |
| **On-chain / alternative data** (Coin Metrics community free, Glassnode paid; Google Trends free; news sentiment paid) | flow and attention mechanisms in crypto | point-in-time and revision risk must be audited at ingest |
| **Futures data** (CME continuous contracts, e.g. via Norgate/CSI) | trend and carry across commodities, rates, FX: the best-documented risk premia | the ETF proxies we have are a weak substitute |

Prices change and are not quoted here; ask before buying anything.

## 6. Found by the morning checks (2026-10-10)

- **Disk**: 49 GB, 89 % used (5.5 GB free); the sibling projects' data is 8.4 GB of derived panels. The lab's own runtime
  (transcripts, caches, forward data, releases) grows by a few hundred MB per week; `lab usage` and the dashboard tile
  show it. Candidate cleanups: prune transcripts older than 30 days, release directories (already limited to 5).
- **Synthetic market realism for `lab try`/G0**: no dividends, no insider events, only 80 names (book always flat),
  so the pre-check can pass vacuously (Librarian, #133). The real G0 on real dev data is not vacuous, but the Builder
  gets weaker feedback than it could. Add event-like features and a larger synthetic universe.
- **Diagnostic output channel**: a Builder's own sanity statistics (share of regular payers, hit rate of predicted
  earnings days) have nowhere to go, so a death cannot be split into "failed proxy" and "absent premium". A small
  `diagnostics` dict returned by `diagnostic.py` and stored in the gate result would do.
- **Index-membership strategies**: the engine forbids weights on non-members of the card's universe, so an index
  deletion leg (long a stock after it left the S&P 500) cannot be expressed (H-0018, parked). Needs a universe kind that
  includes recent leavers, with the membership known point-in-time.
