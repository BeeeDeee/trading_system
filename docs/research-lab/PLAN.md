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
| 2026-10-05 | Data finding | Sharadar SFP `closeadj` has 191 one-day spikes that reverse next day (67 funds, e.g. SSO 2014-06-24 ×3.95). Documented as a known bias in the catalog; the loader does not clean them yet. A strategy trading an affected fund gets a fake +x/−x day pair. Cleaning is a loader (framework) change and needs a canary run. |
