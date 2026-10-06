# Research Lab – plan (step 0)

Status: **draft for approval** · 2026-10-04 · branch `research-lab` (worktree `/home/kapo/ccode/research_lab`,
forked from `strategy_backtester_2026_sep` @ `7fa5ba0`). No code has been written yet.

The lab is a multi-agent pipeline: LLM agents propose and build trading hypotheses, and deterministic code
decides whether they survive. Most hypotheses are expected to die, and every death is recorded with its reason.

---

## 1. What exists and what can be reused

All four directories under `/home/kapo/ccode` are worktrees of one GitHub repo (`BeeeDeee/trading_system`),
one branch each.

| Asset | Where | Reuse in the lab |
|---|---|---|
| **`qlab` framework** (5.7k LOC, 124 tests) | `strategy_backtester_2026_sep/src/qlab` | **Core of the judge.** Imported, never edited. |
| Vector engine (long-only, no leverage, fill at open t+1, delistings, `tradable` mask) | `qlab.engine.vector` | Long-only sims, and the reference for a parity test of the signed engine. |
| Ledger engine (USD, per-order fees), parity with vector engine to 1e-15 | `qlab.engine.ledger` | Paper-trading accounting (Steward) and engine cross-check. |
| Signed engine (shorts, short carry/funding, gross ≤ 1) | `qlab.research10.sim.simulate_signed` | **Base of the unified multi-asset engine** (long/short, cross-asset). |
| Cost model: liquidity-tier half-spread, pre-decimalization ×2.5, multiplier for ×2/×3 runs | `qlab.engine.costs` | US equities/ETF costs. Crypto tiers come from the cpb config (5–50 bps by volume). |
| Cash returns from FRED DTB3 | `qlab.engine.costs.cash_returns` | Benchmarks and cash leg. |
| Look-ahead test (truncation + perturbation of the future) | `qlab.validation.leakage` | **Mandatory integrity check (G0)** and the look-ahead canary. |
| PSR / DSR / PBO (CSCV) / stationary bootstrap / SPA | `qlab.validation.stats` | G3 and the confidence intervals in G1/G4. |
| Append-only trial registry (SQLite triggers) | `qlab.validation.registry` | Pattern reused; the lab needs a `family` column, so it gets its own append-only table in `lab.db`. |
| OOS vault (freeze hash, one-time open, logged) | `qlab.validation.vault` | Logic reused by G4; the lab adds OS-level separation on top. |
| Point-in-time universes (LIQ-N, S&P 500 by date), synthetic market with known truth | `qlab.universe`, `qlab.data.synthetic` | Universes for G1/G2; the synthetic market is used for agent smoke tests and the **positive control**. |
| Rebalance schedules without look-ahead | `qlab.schedule` | Strategy helpers. |
| Binance daily loaders (spot, perp, funding, mark; ms/µs timestamps; gap = delisting) | `qlab.research8/9` | Crypto datasets in the catalog. |
| Cross-asset observation clocks (US close, crypto 00:00 UTC, FX noon, VIX 16:15, day offsets) | `market_relations_2026_oct/src/mrel/clocks.py` | Timestamp alignment for cross-asset strategies (copied with attribution, or imported). |
| Paper bot crypto (`cpb`): fetch with fallbacks, data checks, hash chain, pinning, replay, static dashboard, Caddy + basic auth + fail2ban, systemd timers | `paper_trading_crypto_*` (running as user `cpb`, served from `/srv/cpb`) | **Patterns and modules** for the lab paper runner and dashboard. The running bots themselves are not touched (see §8.4). |
| Paper bot stocks | branch `paper-trading-bot`; runs as a claude.ai scheduled task with an artifact DB, **not on this VPS** | Not reusable as a host. Only the evaluation tools (`random_null`, `prereg_eval`). |

**Data on disk** (`strategy_backtester_2026_sep/data`, 16 GB, git-ignored):
- Sharadar snapshot `sharadar_2026-09-25`: SEP/SFP prices, delistings (with acquisition consideration), SF1
  fundamentals, insiders, 13F holdings, actions, S&P 500 membership, events, tickers. Derived panels:
  LIQ1000, ETF, research 4/5/11.
- Binance `binance_2026-10-03`: spot 1d (all USDT pairs), perp 1d, funding, mark; FRED DTB3.
- `market_relations`: FRED macro (VIX, yields, curve, breakevens, oil, dollar), Binance 1h.
- cpb bots: forward-only hourly order book, OI, long/short ratio since 2026-09.

**Lessons from 13 studies that shape the gates** (README of the backtester):
1. Research 1: the low-vol ensemble passed DSR 0.999 and still failed the holdout. N counted only formal trials,
   not the choice of a well-known family. → G3 must count the family and a literature prior.
2. Research 5: the chosen config was an isolated peak, and 53 % of the profit came from 1999–2002.
   → G2 needs parameter neighborhoods and a concentration limit across sub-periods.
3. Research 6: a real-looking effect with 9 % time invested was not a strategy. → minimum exposure and trade count.
4. Short-horizon strategies lost everything to costs. → costs ×2 is a hard gate, not just a report.
5. Research 8: funding carry decayed after Ethena. → a post-publication / regime sub-period check.
6. Research 13: worse than all 500 random-entry runs. → a random-entry null at matched turnover and exposure.

---

## 2. Gaps against the brief

| Brief | State | Gap |
|---|---|---|
| Unified interface: weights over any instruments | Engines take one `Panel` (one asset class, one calendar) | Multi-asset panel on a union calendar, per-instrument cost/calendar/clock, signed weights. |
| Holdout physically separated, agents cannot read it | Vault is "local code, can be bypassed deliberately" | OS users + permissions + sandboxed execution (§5). |
| **Holdout data actually unseen** | **Everything up to 2026-09-25 (US) and 2026-10-03 (crypto) has already been used by research 1–13 and mrel, and is inside the LLM's training data (cutoff 2026-06)** | Cannot be fixed. See §6.1: the in-sample holdout is a quasi-holdout; only paper forward data is clean. |
| New equity data | Sharadar subscription ended 2026-09-27 | Paper trading for single stocks needs a renewal or a free source (open question Q3). |
| Blackboard, typed messages, state machine, roles | none | New. |
| Gate runner, `gates.yaml`, canaries | Per-study scripts only (`rN_final.py`) | New generic gate runner over qlab. |
| Data catalog with biases | Scattered across `docs/DATA_FINDINGS.md`, audits, research docs | New `catalog.yaml`, seeded from these docs. |
| Orchestrator, budget, agent definitions | cpb shows a single headless `claude -p` call with restricted tools | New. |
| Generic paper runner for arbitrary lab strategies | cpb runs fixed, pre-registered variants | New runner reusing cpb modules. |
| Lab dashboard | cpb static dashboard behind Caddy | New page, same hosting pattern. |

**Resource limits:** 3 cores, 3.9 GB RAM, **8.7 GB disk free (83 % used)**. The lab must not duplicate panels:
dev/holdout separation is done by access control and slicing, not by copying 1–3 GB panels.

---

## 3. Architecture overview

```
                 ┌───────────── user labagent (no data access) ─────────────┐
 cron/systemd →  │ orchestrator renders workspace → claude -p --agent X →  │
 orchestrator    │ agent reads inbox/, writes outbox.json + files            │
 (user labcore)  └───────────────────────────────────────────────────────────┘
       │ validates outbox (schema, role, allowed transition), applies atomically
       ▼
   lab.db (blackboard, owned by labcore)  ←→  gate runner / Sentinel (deterministic, labcore)
       │                                        │ runs strategy code in bwrap sandbox
       ▼                                        ▼ (no network, read-only data, time + RAM limits)
   static dashboard → /srv/lab (Caddy)     data: dev slice | holdout slice (labcore only)
```

Key decisions:

1. **Agents never touch real market data, not even dev data.** Looking at dev data is an untracked trial
   (forking paths), so all data access goes through the gate runner, which counts every run. Agents get the
   catalog (metadata), the registry, gate results (metrics only) and a synthetic market for unit tests. This is
   stronger than the brief asks, and it removes the need for copies of dev data. See Q5 if Scout should be
   allowed logged exploration.
2. **Agents do not write to the blackboard or the repo directly.** The `lab` CLI inside an agent run works
   in *staged* mode: it validates and appends to `outbox.json` in the run's workspace. After the agent exits,
   the orchestrator (a different OS user) re-validates every action against the role table and applies it in
   one transaction, then copies accepted files into the repo and commits them. This gives idempotency for free
   (outbox keyed by invocation id) and makes role spoofing impossible: the agent can set any env var it likes,
   it still cannot write `lab.db`.
3. **Two layers of separation** as required: `.claude/settings.json` deny rules (soft, catches mistakes) and
   Unix users + file permissions + bwrap (hard, catches intent). The deny rules alone are not enough: any agent
   with Bash or the ability to run Python can read any file its user can read.
4. **The judge is code.** No state after IMPLEMENTED can be reached by an agent's action. The Skeptic can only
   block. SKEPTIC_REVIEW → HOLDOUT happens when the deterministic runner sees a completed Skeptic review with no
   open objection (§7).

---

## 4. Directory structure

```
research_lab/                      # worktree, branch research-lab (qlab untouched in src/qlab)
  lab/
    framework/                     # judge: agents read-only (deny rules + file owner labcore)
      cli.py                       # `lab` entry point (staged mode for agents, direct mode for labcore)
      db.py                        # schema + migrations of lab.db
      states.py                    # state machine: allowed transitions × roles
      schemas/                     # JSON Schemas: hypothesis card, each message payload, outbox, strategy meta
      engine.py                    # multi-asset panel + signed engine adapter over qlab (+ parity tests)
      costs.py                     # per-asset-class cost models
      gates.py                     # gate runner (G0–G5), reads gates.yaml
      sandbox.py                   # bwrap runner for untrusted strategy code
      benchmarks.py                # benchmark choice is the framework's, never the agent's
      sentinel.py                  # hard risk limits, kill switch
      paper.py                     # deterministic paper runner (no LLM)
    gates.yaml                     # thresholds; agents read-only
    orchestrator.py                # scheduling, budget, locks, invocation log
    data/
      catalog.yaml                 # datasets, biases, holdout boundaries (written via ingest only)
      sources/<dataset>.py         # fetchers written by Archivist, executed by the ingest runner
    hypotheses/H-0001.yaml         # cards (content); state lives in lab.db
    strategies/H-0001/strategy.py  # Builder code + test_strategy.py
    canaries/                      # look-ahead, noise, overfit, positive control
    dashboard/                     # template + generator
    tests/                         # pytest for state machine, gates, engine, canaries
  .claude/
    settings.json                  # deny rules
    agents/{chair,scout,archivist,builder,skeptic,steward,librarian}.md
  docs/research-lab/               # PLAN, DECISIONS (log), RUNBOOK, CATALOG notes

/srv/research-lab/                 # runtime, outside git (owner labcore, 0750)
  lab.db                           # blackboard (WAL); nightly JSON export committed to git (see Q2)
  workspaces/<invocation_id>/      # per-run agent workspace (labagent rw), kept 30 days
  data/                            # new datasets from Archivist (dev + holdout slices)
  holdout_log/                     # G4 opening log (append-only)
```

Existing Sharadar/Binance data stay where they are; labcore gets read access with an ACL on
`strategy_backtester_2026_sep/data` (no copy, no move, the sibling projects keep working).

---

## 5. Separation of powers (enforcement)

| Rule | Soft layer (`.claude/settings.json`) | Hard layer |
|---|---|---|
| Agents cannot change `lab/framework/`, `lab/gates.yaml`, `src/qlab/` | `deny: Edit(...)`, `Write(...)` on these paths | Agents write only in their workspace; the orchestrator rejects outbox files outside the role's allowed paths (e.g. Builder: `lab/strategies/<own id>/**`). |
| Agents cannot read holdout (or any real data) | `deny: Read(...)` on data paths | labagent has no read permission on `/srv/research-lab/{data,lab.db}` or `/home/kapo` (already 0750). |
| Strategy code cannot leak holdout | – | Strategy code runs only inside bwrap as labcore: no network, read-only bind of the needed slice, tmpfs, CPU/RAM/time limits. Only gate metrics leave the sandbox, never logs or arrays. On G4 only pass/fail + a fixed metric set is stored. |
| Agents cannot approve | No CLI verb exists for it | Transitions past IMPLEMENTED are only executed by labcore code. |
| Bash for agents | `allow` only `uv run lab *`, `uv run pytest lab/strategies/<id>` (Builder) | Even if bypassed, labagent sees only code, catalog and synthetic data. |

**Remaining leak: the LLM's memory of history.** An agent can hard-code what it remembers ("hold NVDA in 2023",
"go to cash in March 2020"). No permission system stops that. Mitigations:
- G0 static scan: no date literals, no ticker/instrument literals beyond the card's declared instruments;
- **ticker-blind mode** for cross-sectional strategies: the strategy sees opaque asset ids, not tickers;
- the Skeptic's checklist includes hindsight mechanisms ("works because of the 2022 rate hikes");
- G4 is treated as a quasi-holdout and only G5 (paper) counts as clean evidence (§6.1).

---

## 6. Data

### 6.1 Holdout boundaries – honest version

| Class | Dev (agents' gate runs) | Holdout (G4, one shot) | Already consumed by |
|---|---|---|---|
| US equities / ETF (Sharadar) | 1998-01-02 → 2020-12-31 | 2021-01-01 → 2026-09-25 (5.7 y) | research 1–7, 10–11 (2020+), mrel holdout 2024+ |
| Crypto (Binance) | 2017-08-17 → 2022-12-31 | 2023-01-01 → 2026-10-03 (3.8 y) | research 8, 9, 10, 13 |
| New datasets (Archivist) | first 70 % | last 30 %, min 2 years, boundary fixed at ingest | – |
| Forward data (after the snapshot) | – | – | **G5 paper only. The only clean test.** |

Both holdouts are contaminated twice: by the owner's earlier studies (we know which families failed on them) and by the
LLM's pretraining, which covers all of it. G4 is therefore a filter against overfitting the dev period, **not
evidence of an edge**. Promotion to LIVE_CANDIDATE requires G5. Why 2021 / 2023 and not later: with ~2.5 years
the standard error of an annual Sharpe is ~0.65, so the test would be close to useless. With 5.7 years it is ~0.42.
Still weak, so G4 checks for a clear failure, not for proof.

### 6.2 Catalog entry (`lab/data/catalog.yaml`)

```yaml
- id: sharadar_sep_2026-09-25
  asset_class: us_equity            # us_equity | us_etf | crypto_spot | crypto_perp | fx | macro | sentiment | alt
  instruments: "all US common stocks incl. delisted (permaticker)"
  frequency: 1d
  range: [1998-01-02, 2026-09-25]
  clock: us_close                   # when an observation becomes known (mrel clocks)
  calendar: XNYS
  source: Sharadar SEP via Nasdaq Data Link (paid, expired 2026-09-27)
  quality: audited (docs/audit/sharadar_2026-09-25.md)
  holdout_from: 2021-01-01          # set at ingest, immutable
  cost_model: us_equity_tiers
  known_biases:
    - "survivorship: none for prices (delistings included); terminal return policy is an assumption (-30 % performance delist)"
    - "SF1 fundamentals: datekey-based PIT, but historical restatements may be partly backfilled"
    - "13F: 45-day reporting lag must be applied by the loader"
  forward_source: null              # how G5 gets new data; null = no paper possible
```

Initial entries: Sharadar SEP/SFP/SF1/insiders/13F/S&P 500, Binance spot/perp/funding/mark 1d, Binance 1h, FRED
DTB3 + mrel macro series, cpb hourly microstructure (forward-only, from 2026-09). Known crypto biases to document:
data.binance.vision lists delisted pairs only partially (survivorship), exchange-specific prices, wash volume in
small caps, the ms→µs timestamp change in 2025.

### 6.3 Ingest of new data

The Archivist does not download data itself. It finds a source (WebSearch/WebFetch), writes
`lab/data/sources/<id>.py` and a draft catalog entry, and emits `DATA_READY`-candidate. The deterministic
`lab data ingest <id>` (labcore, sandbox **with** network only for this step) runs the fetcher, validates
(schema, gaps, duplicates, outliers, timestamps vs clock), sets the holdout boundary, stores dev and holdout
slices separately and returns a validation report. Only then is the dataset in the catalog and blocked
hypotheses move to DATA_READY. Free/public sources are preferred; anything paid becomes a `QUESTION` to the owner.

---

## 7. State machine

`GATE_n` means "passed gate n". Each failure goes to `REJECTED` with `stage` and `reason_code`.

| From → To | Who | Condition (checked by code) |
|---|---|---|
| ∅ → IDEA | Scout | card passes schema |
| IDEA → SPECIFIED | Scout | `mechanism`, `signal`, `falsification_criteria` non-empty; dedup check against registry (Librarian index + embedding-free text similarity); card hash frozen |
| SPECIFIED → DATA_READY / BLOCKED_DATA | system | every `data_requirements` item resolves in the catalog / otherwise `DATA_REQUEST` |
| BLOCKED_DATA → DATA_READY | system after ingest | catalog now resolves |
| BLOCKED_DATA → PARKED | Archivist | infeasible, with reason (paid, does not exist, no PIT history) |
| DATA_READY → IMPLEMENTED | Gatekeeper, on the Builder's `IMPL_DONE` | **G0 integrity** passes (a G0 failure leaves it in DATA_READY, not a trial) |
| IMPLEMENTED → GATE_1 → GATE_2 → GATE_3 | Gatekeeper | gates.yaml |
| GATE_3 → SKEPTIC_REVIEW | system | – |
| SKEPTIC_REVIEW → DATA_READY / IDEA | Skeptic `OBJECTION` (to Builder / Scout) | round ≤ 2; the Builder re-implements (G0 again), the Scout sends a REVISION (new version); every re-run of G1–G3 is a new trial |
| SKEPTIC_REVIEW → REJECTED | Skeptic, or system after round 3 | – |
| SKEPTIC_REVIEW → HOLDOUT | system | completed Skeptic checklist, every item with evidence, no open objection |
| HOLDOUT (passed G4) → PAPER | Sentinel | capacity and risk limits |
| PAPER → LIVE_CANDIDATE | Gatekeeper (G5) + Sentinel | G5 |
| PAPER/LIVE_CANDIDATE → RETIRED | Sentinel | kill rule, or the owner |
| any non-terminal → PARKED | Chair | backlog hygiene, with reason |

A revision of the card after any gate result increments the family trial count and keeps the same id with a
new `version`. The falsification criteria cannot change after SPECIFIED.

### 7.1 Hypothesis card (extensions to the brief are marked +)

```yaml
id: H-0001
version: 1                                  # +
title: ...
author_agent: scout
family: short-term-reversal                 # Chair/Librarian can merge families, never split them
asset_classes: [us_equity]
universe: {kind: liq_n, n: 1000}            # or explicit instruments for allocation strategies
mechanism: {why: ..., counterparty: ..., why_not_arbitraged: ...}
signal: {description: ..., params: {lookback: {value: 20, grid: [10, 15, 20, 30, 40]}}}  # + grid for G2
data_requirements: [{dataset: sharadar_sep, fields: [ret_co, ret_oc], frequency: 1d, period: [1998, 2020]}]
holding_period: {typical_days: 5, rebalance: daily}
market_exposure: long_only | long_short | market_neutral   # + chooses the benchmark
references: [{cite: "Lehmann 1990", year: 1990}]           # + publication decay check
falsification_criteria: ["net Sharpe below SPY on dev", "edge disappears after 1990 publication"]
status: IDEA
history: [{ts, from, to, agent, invocation_id, reason}]
```

---

## 8. Gates (`lab/gates.yaml`) – proposed thresholds

All metrics are net of costs on the class's cost model. Sharpe is annualized (252 for US, 365 for crypto).
The benchmark is chosen by the framework from `asset_classes` + `market_exposure`, never by the agent:

| Strategy | Benchmark |
|---|---|
| US equity long-only | SPY TR |
| Multi-asset ETF long-only | 60/40 SPY/IEF, monthly rebalance |
| Crypto long-only | BTC buy & hold (and EW top-20 as a second benchmark) |
| Long/short, market-neutral | T-bill; Sharpe must also be ≥ 0.5 |
| Cross-asset long-only | 60/40 + 5 % BTC (or the card's asset mix, set by the framework) |

**G0 – integrity** (precondition of IMPLEMENTED): `check_point_in_time` truncation + perturbation on the dev
panel, identical output on two runs, weight schema (finite, gross ≤ 1), no date/ticker literals, runtime < 20 min,
RAM < 1.5 GB. Any failure → back to Builder (not a trial, no data result is shown).

**Configurations.** The card fixes one *primary* configuration and a parameter grid **before** any gate runs.
G1 evaluates only the primary configuration. G2 evaluates the grid neighbors to test stability, never to pick a
winner: the primary configuration stays the one that goes on. Every evaluated configuration is a trial in the
family (G3). Changing the primary value after seeing results is a REVISION (new version, more trials). A
hypothesis may instead declare a selection procedure (e.g. walk-forward choice of the parameter); then the
procedure is the strategy and it is evaluated as a whole.

**G1 – basic** (dev period, full universe):
- Sharpe ≥ benchmark Sharpe + 0.10,
- one-sided 90 % stationary-bootstrap CI of the active return (vs benchmark) > 0 – a weak bar on purpose,
- ≥ 100 position entries and ≥ 36 decision dates with non-zero turnover,
- mean exposure ≥ 20 % (lesson 3),
- max drawdown ≤ 1.5 × benchmark max drawdown.
Why so lenient: G1 is a cheap filter. Strictness belongs to G2–G4, where it does not depend on one point estimate.

**G2 – robustness:**
- parameters: every declared parameter on its grid neighbors (one step each way, all combinations up to 3
  params). ≥ 75 % of neighbors beat the benchmark Sharpe and the median neighbor Sharpe ≥ 0.7 × the point
  Sharpe (lesson 2: no isolated peaks),
- sub-periods: dev split into 3 equal blocks. Active return > 0 in ≥ 2 of 3, and no block brings > 60 % of the
  total active P&L (lesson 2),
- alternative universe chosen by the framework (LIQ1000 → S&P 500 PIT, top-20 crypto → top-50, ETF set →
  substitute ETFs): Sharpe ≥ benchmark and ≥ 0.5 × the main universe,
- costs ×2: still Sharpe ≥ benchmark Sharpe (hard). Costs ×3 reported only,
- random-entry null: 200 runs with the same turnover, exposure and holding period. The strategy must beat the
  95th percentile (lesson 6),
- publication decay: if `references` has a year inside dev, the post-publication part keeps ≥ 50 % of the
  pre-publication active Sharpe (McLean–Pontiff found ~1/3 decay on average).

**G3 – multiple testing:**
- `N = trials in the family + literature prior` (prior = 20 when `references` is non-empty, else 0; lesson 1),
  with a floor N ≥ 10. A trial = every gate run with a distinct config, including Skeptic-driven re-runs,
- DSR ≥ 0.95, with the Sharpe variance taken from the family's trials, floored at the variance implied by the
  sample length,
- PBO (CSCV, 16 blocks) ≤ 0.30 over the G2 parameter grid,
- lab-wide report (not a gate yet): number of G3 attempts in the lab and the expected number of false passes.

**SKEPTIC_REVIEW:** fixed checklist (look-ahead, leakage through universe/features, survivorship, costs and
capacity, data snooping/family history, mechanism and counterparty, hindsight, regime dependence). Max 2
objection rounds, then REJECTED.

**G4 – holdout** (one attempt per hypothesis, logged; family cap 3 holdout attempts, then the family needs the owner's
approval):
- Sharpe ≥ benchmark Sharpe,
- Sharpe ≥ 0.5 × dev Sharpe,
- costs ×2 active return ≥ 0,
- max drawdown ≤ 1.5 × dev max drawdown.
The thresholds tighten with family holdout attempts (Bonferroni over the attempts: the active-return CI level
goes from 80 % to 90 % to 93 %). Power note: on 5.7 y this only catches clear failures.

**G5 – paper** (forward data only):
- ≥ 6 months and ≥ 30 position entries (crypto daily: ≥ 4 months),
- tracking: the paper P&L vs a backtest replay over the same days, daily correlation ≥ 0.9 and cumulative gap
  ≤ max(2 %, 2 × tracking error × √years),
- realized costs ≤ 1.5 × modeled costs,
- forward Sharpe not below the 5th percentile of the backtest-implied distribution for that length.
LIVE_CANDIDATE is a label for the owner's decision. No automated live trading.

**Sentinel (deterministic):** max 5 strategies in PAPER, max 25 % weight per instrument, gross ≤ 1, no leverage;
kill on drawdown > 1.5 × backtest max drawdown or on failing the G5 tracking test for 20 consecutive days;
global kill switch file `/srv/research-lab/KILL`.

### 8.1 Canaries (run on every change of `lab/framework/` or `gates.yaml`, and in CI pytest)

| Canary | Must |
|---|---|
| look-ahead: a sound signal shifted one day into the future | fail G0 (and, with G0 disabled, show an implausible G1 Sharpe) |
| noise: 20 seeded random-weight strategies | all fail G1–G3 |
| overfit: best of 2 000 random parameter sets of a nonsense signal, picked on dev | fail G2 or G3 (and G4) |
| memorized hindsight: hard-coded 2020 crash timing | fail G0 static scan |
| **positive control**: a planted edge on the synthetic market (`qlab.data.synthetic`) | **pass** G0–G3 |

The positive control matters as much as the canaries: a judge that rejects everything is also broken.

### 8.2 Steward and the existing paper bots

The cpb bots are pre-registered experiments (evaluation from 2026-12-04). Adding lab strategies to them would
break their pre-registration. So the lab gets **its own deterministic paper runner** (`lab/framework/paper.py`,
systemd timer, no LLM) that reuses cpb modules (fetch with exchange fallbacks, data checks, hash chain) and the
qlab ledger engine. The Steward agent only reads the runner's results and writes reports. Forward data: crypto
from public Binance/OKX APIs (as cpb), ETFs from a free daily source, single stocks blocked by Q3.

---

## 9. Blackboard (`lab.db`, SQLite WAL, owner labcore)

```sql
hypotheses(id PK, version, family, title, status, stage_reason, card_path, card_sha256,
           objection_rounds, priority, created_at, updated_at)
transitions(id PK, hypothesis_id, from_status, to_status, actor, invocation_id, reason, ts)   -- append-only
messages(id PK, type, from_agent, to_agent, hypothesis_id, payload_json, schema_version,
         idempotency_key UNIQUE, created_at, handled_at, handled_by_invocation)
runs(id PK, hypothesis_id, kind, config_json, config_sha256, code_commit, data_ids, started_at, ended_at,
     status, error)                                                                          -- gate executions
gate_results(id PK, run_id, hypothesis_id, gate, passed, metrics_json, thresholds_sha256, ts)  -- append-only
trials(id PK, family, hypothesis_id, run_id, config_sha256, sharpe, n_obs, ts)                 -- append-only
agent_invocations(id PK, agent, model, hypothesis_id, input_sha256, workspace, started_at, ended_at,
                  exit_code, outcome, n_turns, tokens_in, tokens_out, transcript_path, outbox_applied)
budget(day, agent, invocations_planned, invocations_used)
locks(hypothesis_id PK, invocation_id, lease_until)
```

Append-only tables get the same UPDATE/DELETE triggers as `TrialRegistry`. Message types: `NEW_HYPOTHESIS`,
`DATA_REQUEST`, `DATA_READY`, `IMPL_DONE`, `GATE_RESULT`, `OBJECTION`, `REVISION`, `VERDICT`, `QUESTION`,
`ALERT`, each with a JSON Schema in `lab/framework/schemas/messages/`. `QUESTION` to `human` shows on the
dashboard.

## 10. Orchestrator and budget

- systemd timer every 30 min (consistent with the cpb bots, with `Persistent` and journal logs). Cron also works
  if you prefer it.
- One cycle: run deterministic work first (ingest, gates, Sentinel, paper), then pick at most one agent task by
  Chair's priority list, take the lease on the hypothesis, render the workspace (inbox, card, catalog, registry
  excerpt, gate metrics), run
  `claude -p --agent <name> --model <per agent> --output-format stream-json --max-turns <n>` as labagent with a
  timeout, store the transcript, validate and apply the outbox, release the lease.
- Daily budget in `lab/budget.yaml` (proposal: 12 invocations/day: Scout 3, Builder 3, Skeptic 2, Archivist 2,
  Librarian 1, Chair 1, Steward 0–1). Chair can reallocate within the total. On a rate-limit error everything
  stops until the next window.
- Quiet window 00:00–02:00 UTC so the lab never competes with the cpb daily run for the same subscription.
- Auth for labagent via `claude setup-token` (long-lived token in an env file readable only by labcore).

## 11. Dashboard

A static page generated after every cycle into `/srv/lab/current` and served by the existing Caddy with basic
auth (same template and CSP as cpb): funnel by stage, rejection reasons by stage and code, per-hypothesis trace
(messages + transitions + invocations in time order), budget use per agent, paper vs backtest replay vs benchmark,
canary status, open questions for the owner.

## 12. Implementation steps (after approval)

1. **Walking skeleton:** schemas, `lab.db`, state machine, `lab` CLI (direct + staged mode), stub agents (Python
   functions that write a valid outbox), one hand-written hypothesis through all states. Single-user dev mode.
2. Multi-asset engine adapter + parity tests, gate runner G0–G3 over qlab, canaries + positive control, pytest for
   the state machine and gates.
3. Agents one by one (Scout → Builder → Skeptic → Archivist → Librarian → Chair → Steward), each first run by hand.
4. OS users, bwrap sandbox, orchestrator, systemd timer, G4/G5, Sentinel, paper runner.
5. Dashboard.

---

## 13. Open questions (owner decisions)

- **Q1 – OS users (needs sudo).** Hard separation needs users `labcore` and `labagent`, `/srv/research-lab`, an ACL
  on the existing data dir, a Caddy site and a systemd timer. I will write the install script, but you have to
  run it (sudo needs a password). Without it principle 2 holds only by convention. OK?
- **Q2 – Public repo.** The cpb README says `BeeeDeee/trading_system` is public. Should the lab branch be pushed,
  and should a nightly `lab.db` JSON export (hypotheses, reasons for death) be committed? Proposal: push code and
  cards, keep `lab.db` local with an encrypted backup.
- **Q3 – Equity forward data.** Without Sharadar, single-stock strategies cannot reach G5. Renew Sharadar, accept a
  free source without PIT/delisting guarantees for paper only, or limit G5 to ETF and crypto for now?
- **Q4 – Budget.** How many headless runs per day can the lab use alongside the cpb bot? Proposal: 12, and which
  model per agent (proposal: Opus for Scout, Builder, Skeptic; Sonnet for Archivist, Librarian, Chair, Steward).
- **Q5 – Scout exploration.** Should the Scout get a logged exploration tool on dev data (each query counted as a
  trial in its family), or stay data-blind (proposal for v1)?
- **Q6 – Holdout boundaries** 2021-01-01 (US) and 2023-01-01 (crypto) as in §6.1, knowing they are contaminated?
- **Q7 – PARKED** is terminal in the brief. Should Chair be able to reopen a parked hypothesis (as a new version,
  counted in its family)? Proposal: yes.
- **Q8 – Leverage and shorts** in v1: gross ≤ 1 (shorts allowed, as research 10), no leverage. OK?

## 14. Risks

1. **Contaminated history (biggest).** Everything up to now is in the LLM's training data and in the earlier studies.
   The pipeline can still kill bad ideas cheaply, but only G5 forward data can confirm a good one. Expect
   months between an idea and LIVE_CANDIDATE.
2. **Family gaming.** A new family name resets N. Mitigation: the Librarian proposes merges, Chair merges, and the
   literature prior applies regardless of family.
3. **Base rate.** 0 of 13 earlier studies survived. If the lab produces a LIVE_CANDIDATE quickly, suspect the judge
   first (canaries, positive control, a manual audit).
4. **Disk (8.7 GB free).** Archivist ingest has a per-dataset size cap (proposal 500 MB) and a lab-wide quota.
5. **Small machine.** Gate runs are serialized (one at a time, `ulimit` + bwrap memory cap), panels via mmap.
6. **Subscription limits** are not visible via an API. The budget is counted in invocations and tokens from the
   transcript, with back-off on rate-limit errors.
7. **Agent output quality.** Strategy code from an LLM may be subtly wrong in ways G0 misses (e.g. universe
   filters on future data inside a loader). Mitigation: strategies receive data only through the framework's
   `DataView`, which is itself point-in-time tested.
8. **Complexity vs value.** The secondary goal (learning multi-agent design) is well served. The primary goal
   (finding an edge) is limited by data, not by architecture. The cheapest win may be a stricter, faster
   way to kill ideas, which this design gives.

## 15. Decision log

| Date | Question | Decision |
|---|---|---|
| 2026-10-04 | Q1 OS users | Yes: `labcore` / `labagent`, install script run by the owner with sudo. |
| 2026-10-04 | Q2 Public repo | Yes: push the lab branch and commit the nightly `lab.db` JSON export. |
| 2026-10-04 | Q3 Equity forward data | Sharadar will **not** be renewed. G5 starts with ETF and crypto only. Single-stock hypotheses that pass G4 go to PARKED (`no_forward_data`) until the Archivist validates a free source (stockanalysis/Stooq prices cross-checked, SEC EDGAR XBRL for fundamentals); then Chair reopens them. |
| 2026-10-04 | Q4 Budget | Claude Pro. Start with 8 runs/day: Scout 2, Builder 2, Skeptic 1 (Opus 5.5); Archivist 1, Librarian 1, Chair 1 (Sonnet 5.5); Steward 0 until something is in PAPER. Quiet window 00:00–02:00 UTC, stop on rate limit. Owner monitors and adjusts `lab/budget.yaml`. |
| 2026-10-04 | Q5 Scout exploration | Data-blind Scout. It gets descriptive per-dataset fact sheets from the catalog (coverage, instrument count, typical volatility/liquidity, cross-class correlations), never signal-conditional returns. Logged exploration may be added later as an explicit decision. |
| 2026-10-04 | Q6 Holdout boundaries | Confirmed: holdout starts 2021-01-01 (US) and 2023-01-01 (crypto); dev ends the day before. |
| 2026-10-04 | Q7 Reopen PARKED | Yes: Chair can reopen as a new version, counted in its family. |
| 2026-10-04 | Q8 Shorts / leverage | Yes: shorts allowed, gross ≤ 1, no leverage. |
| 2026-10-04 | Step 1 design | Agents only send messages; the framework performs every transition in reaction to a message or a gate result (`lab/framework/tick.py`). The Skeptic returns work to DATA_READY (Builder) or IDEA (Scout), not to IMPLEMENTED/SPECIFIED, because the returned stage is the one that owns the fix. |
| 2026-10-04 | Agent permission rules | Agents run with their workspace as cwd, so the repo's `.claude/settings.json` would not apply to them and would block the owner's own sessions on the framework. Their deny rules go to `lab/agents/settings.json`, passed by the orchestrator with `--settings` (step 3). |
| 2026-10-04 | Step 2: risk-adjusted comparison | All Sharpe ratios are of returns in excess of cash. G1/G4 test the Sharpe difference to the benchmark (bootstrap CI of SR − SR_bench), not the raw active return: a long-only strategy that is partly in cash would otherwise have to beat the benchmark's return, not its risk-adjusted return. Sub-periods: Sharpe ≥ benchmark in ≥ 2 of 3 blocks, no block > 60 % of the log return. G3: P(true SR > SR_bench + expected max SR of N unskilled trials) ≥ 0.95. |
| 2026-10-04 | Step 2: PBO report-only | With a pre-registered primary configuration there is no selection over the grid, and a robust plateau gives PBO ≈ 0.5 by construction (the positive control failed it). PBO is stored in the G3 metrics, the gate is the neighbor test in G2. |
| 2026-10-04 | Step 2: canaries gate the judge | `lab canaries` records a pass/fail per hash of lab/framework + lab/canaries + gates.yaml. The real evaluator refuses to run any gate unless the current hash has a passing run. pytest runs them too (quick mode). |
| 2026-10-04 | Step 2: loaders | The gate runner loads `sharadar_sfp` (any ETF set, built on demand), `binance_spot_1d` (research 9 panel, to 2026-08-31) and the synthetic market. Catalog entries carry `loader: true/false`; a hypothesis on a dataset without a loader stays BLOCKED_DATA and the owner gets a QUESTION. Loaders are framework code, so the Archivist cannot add them. |
| 2026-10-04 | Step 2: sandbox later | Strategy code runs in-process with the static scan (imports allowlist, no I/O, no date/instrument literals) and a wall-clock limit. The bwrap sandbox with RAM limits and the separate OS user come in step 4. Until then G0 is the only protection against hostile code. |
| 2026-10-05 | Step 3: headless runner | `claude -p` per `lab/agents/agents.yaml`: the agent's prompt file is the whole system prompt (`--system-prompt`), workspace as cwd, `--restricted --permission-mode dontAsk --strict-mcp-config --no-session-persistence`, only the listed tools, Bash only as `lab <verb>`, deny rules from `lab/agents/settings.json`. The transcript goes to `LAB_HOME/transcripts/` (outside the workspace) and is audited after the run: any attempted call outside the workspace, any other shell command or tool discards the whole outbox (`policy_violation`) and sends an ALERT to the owner. Attempts count, not only successes. |
| 2026-10-05 | Step 3: Scout context | Workspace gets `factsheets.md` (dev-only, descriptive: fund/pair counts, top instruments by liquidity with first price date and volatility, cross-class weekly correlations; no returns or Sharpe), `prior_studies.md` (the 13 studies + mrel, with lessons), the card schema, an example card and `gates.yaml`. WebSearch is allowed for literature, WebFetch is not (it could fetch prices). `lab check <card>` tells the Scout before sending what the framework will refuse. |
| 2026-10-05 | Step 3: parameter grids | A card reaches SPECIFIED only if every primary value is in its grid, numeric grids are strictly increasing and there is at least one neighbor (G2 tests neighbors; a value outside the grid used to crash the evaluator). A neighbor on each side is advice (`lab check` warning), because edge values such as weight 1.0 under gross ≤ 1 are legitimate. |
| 2026-10-05 | Data finding | Sharadar SFP `closeadj` has 191 one-day spikes that reverse next day (67 funds, e.g. SSO 2014-06-24 ×3.95). Owner decision (same day): the loader drops these days (`data.drop_spikes`, |log move| > 0.4 reversed within 0.1 the next day): the day has no bar, the next return runs from the last good close. Loader version in the cache key, so old cached panels are not reused. Canaries re-run. |
| 2026-10-05 | Step 3b: Builder | Data-blind like the Scout. `lab try` (staged) runs the same G0 `integrity()` as the gate runner on a synthetic market with the card's instruments and dev calendar (late listing, missing opens, crypto delisting), runs every G2 grid neighbor once and the Builder's pytest file, and prints only trading statistics (decision dates, exposure, turnover, entries), never returns. G0 was extracted from the evaluator into `evaluator.integrity()` for this; canaries re-run. |
| 2026-10-05 | Units of the Sharpe-difference CI | The bootstrap lower bound in G1/G4 was stored per period while the Sharpe values are annualized (H-0001: point −0.34 annualized, bound −0.04 per period). Now annualized; the verdict (sign) is unchanged. Gate results before this change carry the per-period bound. |
| 2026-10-05 | First full pass | H-0001 (Scout, turn-of-month SPY/IEF) → Builder implemented it (rule-based NYSE holiday calendar, no date literals) → G0 passed on real dev data → **G1 rejected**: Sharpe 0.38 vs 60/40 0.72, CAGR 4.9 % vs 8.6 %, 441 switches, exposure 1.0. |
| 2026-10-05 | Evaluator: requirements per dataset | Several `data_requirements` on the same dataset (H-0003: SPY and IEF as two sharadar_sfp entries) overwrote each other and dropped SPY, so G0 crashed on real data while `lab try` passed. `evaluator.strategy_instruments()` now unions them and `lab try` uses the same function. H-0003's G0 was re-run with `lab rerun-g0` (owner, logged, not a trial). Known limitation: the view starts at the latest `period` start, so a card cannot ask for a longer warm-up of one instrument. |
| 2026-10-05 | Policy audit stays strict | Builder run for H-0004 ran `ls` on its own workspace and was discarded. Kept strict (attempts count); prompts now say "Glob/Read, never ls/cat/find". |
| 2026-10-05 | Scout round 2 + Skeptic | Three more Scout runs: H-0002 month-end 60/40 rebalancing front-running (L/S SPY/IEF, G1: Sharpe 0.06 vs T-bill bar 0.5), H-0003 volatility-timed SPY/IEF (G1: 0.65 vs 0.72), H-0004 crypto new-listing short vs seasoned long (G1: Sharpe 0.44, CI lower −0.20, bar 0.5). All four lab hypotheses died at G1; none reached the Skeptic. Skeptic built (history, family, prior studies, `lab try`, WebSearch) and rehearsed in a sandbox on the positive control: valid no_objection with evidence for all 8 items, G4 passed, PAPER. It found that `lab try` lacked the generator's series for synthetic_market cards (fixed). |
| 2026-10-05 | Step 3d: Archivist + real ingest | The Archivist (Sonnet 5.5, WebSearch + WebFetch for API docs) writes `lab/data/sources/<id>.py` (`fetch(get) -> [(date, key, value)]`, statically scanned, network only through the framework's https client with request/byte/time caps) and a draft entry. `lab fetch` lets it test the fetcher and shows counts, ranges and gaps, never values. On DATA_READY the judge (`evaluator.ingest`, not an agent) fetches again, validates, sets the holdout boundary (class boundary 2021-01-01 / 2023-01-01 if the history allows, else 70 %), writes `LAB_HOME/data/<id>/{dev,holdout}.parquet` and adds the entry with `loader: generic`. Ingest moved into the evaluator so that `tick` without an evaluator never touches the network. Only daily data so far. |
| 2026-10-05 | Generic datasets in strategies | A `loader: generic` dataset is a signal, not tradable: every key arrives as `series["<id>.<key>"]`, aligned by the clock (us_close/crypto: usable on day d; next_morning: from d+1). All keys are attached; a card's `fields` are the Scout's guess from before the ingest. |
| 2026-10-05 | Owner void of a framework-bug rejection | New owner-only transition REJECTED → DATA_READY (`lab void`, only for gate rejections, reason logged, earlier trials stay counted). First use: H-0005 was rejected at G1 with zero exposure because the evaluator attached only the card's guessed field name; after the fix and `lab rerun-g0` it was rejected at G1 for a real reason. Builder prompt: fail loudly on missing inputs (the H-0005 strategy silently held nothing). |
| 2026-10-05 | H-0005 (first new dataset) | Scout: crypto top-10 inverse-vol weekly, risk-on while USD stablecoin supply grew over 28 days. Archivist: DefiLlama stablecoin supply (free, backfilled/recomputed history: flagged as look-ahead risk). G1: Sharpe 0.73 vs BTC 0.37, CAGR 31 % vs 0.5 % (dev 2018-2022), but the 90 % bootstrap bound of the Sharpe difference is −0.06 → rejected. |
| 2026-10-05 | Improvements list | `docs/research-lab/IMPROVEMENTS.md`: what limits hypothesis generation, combination cards, free and paid improvements. |
| 2026-10-05 | Step 3e: Librarian | Sonnet 5.5, no web. Gets `cases.json` (each decided hypothesis without a lesson: card, full gate metrics, Builder summaries, Skeptic messages) and sends one LESSON per case (new message type, librarian → system). Lessons must be qualitative: the framework refuses decimals, percentages, multiples and signed numbers, because the Scout reads them (`lab/knowledge/lessons.md`, rendered from accepted lessons, append-only `lessons` table) and metric feedback would be an untracked adaptive search on dev data. Family merges go as QUESTION to the Chair. First run: 5 lessons; it flagged H-0001's per-period CI bound (fixed earlier the same day), H-0002's exit-at-close deviation (the engine trades at opens only), H-0004's spot shorts without borrow cost, and proposed merging `rebalancing-flow-pressure` into `turn-of-month-flows`. |
| 2026-10-05 | Open: crypto spot shorts | Spot shorts are simulated without borrow cost (H-0004). Until perps + funding are loadable, a market-neutral crypto card is flattered; decision pending (charge a borrow rate, or allow shorts only on perps). |
| 2026-10-05 | Loaders for data on disk | `sharadar_sep`: qlab LIQ1000 panel, universe kinds `liq_n` (n ≤ 500) and `sp500`, ticker-blind ids `E<permaticker>`, columns = union of LIQ-n and S&P 500 members (G2 alternative universe: liq_n ↔ sp500), float32 matrices, equity cost tiers by PIT liquidity rank. Measured: an S&P 500 card through G0–G2 peaks at 1.7 GB RSS (LIQ-1000 would not fit in 3.9 GB). Variant runs (neighbors, cost stress, alt universe) keep only returns. `binance_perp_1d`: research 8 panel (389 perps with a spot pair, 2019-12..2026-09), ids `<SYMBOL>.P`, funding charged by the engine (new `funding` argument: longs pay, shorts receive), funding/basis as extras. `fred_macro` (29 series) and `fred_dtb3`: imported by the owner command `lab import-local` into the generic format (holdout boundary of the existing entries kept). Not done: Binance 1h (needs intraday features or an intraday engine), standalone funding (part of perps). Strategies must not read `alt_universe`. |
| 2026-10-05 | Timeframes | Owner: explore different timeframes and multi-timeframe confirmation. Decisions stay daily (engine), but (1) Scout `propose` runs get a deterministic **brief**: the least-used (holding horizon: short 1-5 d / swing 1-4 w / medium 1-6 m / long 6-24 m) × (asset group: crypto, US single stocks, US ETFs, cross-asset) cell of the registry, plus a multi-timeframe requirement while fewer than half of the cards combine timeframes; (2) cards may declare `signal.timeframes` (1h..1y); (3) PIT higher-timeframe helpers in `lab.framework.api` (`completed_period_value`, `period_return`, `last_completed_row`, `ema`; a period counts once it is complete); (4) `binance_1h_features` (imported from the hourly klines on disk: realized vol, Asia/Europe/US session returns, US-session volume share for 105 pairs) brings the hourly timeframe into daily decisions. Intraday holding needs an intraday engine (see IMPROVEMENTS). |
| 2026-10-05 | Benchmark of mixed spot/perp cards | A long-only card trading Binance spot and perps (H-0006) got the cross-asset benchmark (57/38/5 SPY/IEF/BTC) because two asset classes were present. Crypto spot + perps now map to BTC buy & hold, US stocks + ETFs to 60/40. H-0006's rejection stands without a void: it also failed the mean-exposure minimum (in the market about an eighth of the time), which does not depend on the benchmark. |
| 2026-10-05 | Scout round 3 (briefs) | H-0006 crypto liquidation-flush rebound gated by a weekly BTC trend (short horizon, 1w+1d): G1 rejected (exposure, bootstrap bound). H-0007 LIQ-500 weekly long/short, intraday-return reversal confirmed by 12-month intraday momentum (1d/1w/1mo): **first G1 pass** (Sharpe 0.53 vs T-bill, bound > 0, random entry top), **G2 rejected**: S&P 500 universe Sharpe ≈ 0, one sub-period holds two thirds of the log return, neighbor median about two thirds of the primary. H-0008 earnings-announcement premium via volume seasonality (1d..1y): G1 rejected (negative Sharpe). The G2 run on single stocks peaked at 2.4 GB RSS (Builder-written strategy), at the limit of this machine. Note: one unreproducible test failure in a full run (145 passed on two reruns). |
| 2026-10-05 | Random-entry null for universe cards | Found by the Librarian on H-0007: random runs drew columns from every listed stock (2 364 in the LIQ-500 ∪ S&P 500 panel), mostly illiquid names with several times the cost, so the null was far too low (median Sharpe −1.85) and the check passed trivially. Random assets now come from the card's universe (≥ 50 % of the window). H-0007's verdict is unaffected (it failed other G2 checks). |
| 2026-10-05 | G0 look-ahead on sparse strategies | A flaky test showed that a peeking momentum strategy with monthly decisions was caught only on some synthetic markets: the random cuts rarely hit decision rows. G0 now adds as many cuts on rows where the weights change. `lab try` seeds its synthetic market from the card content only (it varied with the card's history). Canaries re-run. |
| 2026-10-05 | First Chair run | Merged `rebalancing-flow-pressure` (H-0002) into `turn-of-month-flows` (H-0001) on the Librarian's proposal; declined to merge `stablecoin-flow-pressure`. Steward waits for the paper runner (step 4): nothing is in PAPER. |
| 2026-10-06 | Knowledge base | Owner: every learning goes into a knowledge database. `lab/knowledge/KNOWLEDGE.md`, rendered from an append-only `knowledge` table (K-ids; corrections supersede, nothing is edited). Kinds: market, data, method, framework, process. Written by the Librarian and the Chair (KNOWLEDGE message) and by the owner (`lab knowledge add`); read by every agent. `market` entries are qualitative like lessons. Seeded with K-0001..K-0012 from steps 1-3. |
| 2026-10-06 | Step 4a: orchestrator | `lab cycle` (systemd timer every 30 min): file lock, tick, then at most `max_agent_runs_per_cycle` agent runs inside `lab/budget.yaml` (per agent and total per UTC day counted from agent_invocations, quiet window 00-02 UTC, PAUSE file, 5 h pause after a rate-limit error with an ALERT). The Scout proposes only while fewer than 2 hypotheses wait in IDEA/DATA_READY, with the rotating brief. Every cycle is logged to LAB_HOME/cycles.jsonl. |
| 2026-10-06 | Step 4b: separation | Tick split into `ingest` (Archivist fetchers, network, in the orchestrator) and `judge` (strategy code) which runs in bubblewrap when `LAB_SANDBOX=1`: read-only root, no network, own PID namespace, writable only LAB_HOME, lab/hypotheses, lab/knowledge (tested). Agents run as `labagent` via a sudoers rule limited to the claude binary; workspaces live in /srv/research-lab/workspaces (group labwork, 2770), LAB_HOME (labcore 0700) is invisible to agents; labcore alone gets read ACLs on the data. `deploy/research-lab/install.sh` (owner runs it, step by step), units with MemoryMax 2.8 GB, nightly `lab export` + commit + push of cards/strategies/knowledge/export only. |
| 2026-10-06 | Step 4c: paper trading, G5, Sentinel, Steward | Forward data: framework code fetches closed UTC days after the snapshot from the Binance public API (spot klines; perp klines + funding) in the ingest phase, stored append-only per instrument in LAB_HOME/forward (a day is never rewritten). Paper runner (judge phase, sandbox): the strategy runs over history + forward days with the lab engine, new forward days are appended to the append-only `paper_days` table and never recomputed. Sentinel retires on paper drawdown > 1.5 x the dev drawdown. G5 after the minimum period (crypto 4 months, else 6) and >= 30 entries: forward Sharpe >= 5th percentile of the dev-implied distribution; fill tracking and realized costs are N/A without real orders. Forward cash return = last known T-bill day. Steward (Sonnet) writes a weekly ALERT report while something is in PAPER and may add KNOWLEDGE. Only crypto can paper trade (no forward source for ETFs/stocks, decision Q3). |
| 2026-10-06 | Step 5: dashboard | `lab dashboard <web>`: one static HTML page without JavaScript (CSP-friendly), rebuilt after every production cycle (`LAB_WEB`), published as a release directory with an atomic `current` symlink. Sections: status tiles (canaries, budget, tokens, pauses), the owner's inbox, funnel and causes of death, horizon × asset-group coverage of the Scout briefs, hypothesis table with gates and full trace, paper book with cumulative growth vs benchmark, agent runs, budget, recent cycles, knowledge base, lessons. Served by the existing Caddy site at /lab/ behind the cpb basic auth (install step `web`). Owner operations on production: `lab-prod` (sudo rule for kapo → root-owned wrapper → `lab` as labcore with the production environment), `lab ack` marks owner messages handled and can reply to the agent. |
| 2026-10-06 | First production cycles | The cycle unit's hardening (ProtectKernelTunables/Modules, LockPersonality) implies NoNewPrivileges, so `sudo -u labagent` refused and 3 agent runs failed before the model answered. Removed those options (strategy code is sandboxed by bwrap anyway). Runs that fail before any model turn no longer count against the budget, and 3 such failures in a row create the PAUSE file with a critical ALERT instead of retrying every 30 minutes. Also fixed `lab status`/`inbox`/`invocations` (a local import shadowed `report`; smoke tests now cover every read-only command). |
| 2026-10-06 | Agents blind in production | After the sudo fix the first production Scouts ran as labagent but every file tool was denied: the agents' deny rule `Read(//srv/**)` (meant for the lab state) covered /srv/research-lab/workspaces. Narrowed to /srv/research-lab/home, /srv/research-lab/web, /srv/cpb, /etc/research-lab; a test checks that no deny rule covers a workspace. The Scouts reported it as QUESTIONs to the owner instead of guessing. The dashboard now makes its web directories world-readable (the cycle runs with umask 0007, Caddy could not read the releases). `lab-prod-update` lets the owner deploy pushed code (pull, sync, canaries) without root; units and sudoers still go through `install.sh update`. |
| 2026-10-06 | Judge calibration on real data | Owner asked for it (improvement 1). `lab calibrate <name>` runs known strategies through the real gates in a throwaway lab (fixtures `lab/calibration/<name>/`, results `docs/research-lab/calibration/`): low-vol stocks and funding carry (good in dev, failed holdout in research 1 / 8), momentum (no firm expectation), 5-day reversal and BTC trend (failed in research 5/6/9). Findings: (1) the engine reproduces research 8's funding-carry Sharpe (8.4); (2) `min_position_entries` rejected the carry (4 entries, 156 weekly rebalances): for books invested at least half the time the rebalancing dates now count as bets (`continuous_min_exposure`); (3) the carry then died at G2 on one sub-period holding 60.4 % of the log return (limit 60 %), consistent with research 8's regime dependence; (4) the BTC trend filter died at G1 on the Sharpe-difference bound, as in research 9; (5) the catalog said spot data runs to 2026-10-03 but the loadable panel ends 2026-08-31: range corrected. |
| 2026-10-06 | Mechanism test (improvement 2) | Cards may declare `mechanism_test` {event, primary_horizon_days, horizons}; the Builder writes `diagnostic.py: events(data, params) -> (mask, side)`. The judge runs, inside G1 and before the strategy is simulated, an event study on dev rows: forward return from the next open to the close of day t+h minus the same-date mean of the other eligible instruments, side-adjusted, against 200 placebos (the whole event set shifted in time by a random number of rows, which keeps clustering and counts). Passes if: >= 200 events on >= 30 dates, positive mean abnormal return, placebo percentile >= 0.95, positive in >= 2 of 3 sub-periods, gross abnormal return >= round-trip cost. Failure = REJECTED at G1 with `g1_mechanism_*` and the strategy is not run. One extra trial of the family (other horizons descriptive only). G0 checks diagnostic.py like the strategy (static scan, determinism, truncation/perturbation); `lab try` shows the event counts, never returns. Canaries: planted effect passes, absent mechanism dies at G1, 20 random event sets pass at most 3. |
| 2026-10-06 | Calibration results | Five fixtures through the real gates: low-vol stocks reach G2 (G1 Sharpe 0.54 vs SPY 0.36, bound > 0, random-entry percentile 1.0) and die on publication decay (post/pre-2007 excess Sharpe ratio 0.18); funding carry Sharpe 8.4 (= research 8), G2 dies on one sub-period at 60.4 % of the log return (limit 60 %); momentum 12-1, 5-day reversal and BTC trend die at G1 (Sharpe excess about 0 or bound below 0). No false pass, no unexplained death. What this cannot show: that the judge passes a genuinely good real strategy, because none is known. Bug found: the random-entry null drew from stocks listed in more than half of the window (survivorship, null Sharpe 0.62 > SPY 0.36); replaced by `nulls.random_book` (date-by-date eligibility, retention-matched turnover). |
