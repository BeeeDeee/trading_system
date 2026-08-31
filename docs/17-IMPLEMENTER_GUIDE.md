# Implementer Guide

> **Read this before writing any code.** It is written for whoever implements
> this system, including AI models working from these documents.

---

## 1. How to work on this project

1. Open [`15-ROADMAP.md`](15-ROADMAP.md). Find the lowest-numbered task that is
   not done.
2. Read the document referenced by that task, in full.
3. Implement **only that task**. Do not implement the next one because it is
   small. Do not refactor an earlier one because you would have done it
   differently.
4. Write the tests listed for that task in
   [`16-TESTING.md`](16-TESTING.md).
5. Run `ruff check .`, `mypy src/scout`, `pytest`.
6. Stop. Report what you did, which acceptance criteria are met, and which are
   not.

If a document is ambiguous or contradicts another document, **stop and ask**. Do
not resolve it by guessing. An ambiguity resolved silently in the scoring or
labeling code produces a system that looks correct and is wrong, and the error
will not be found until it costs money.

---

## 2. The five rules that matter most

Every rule below has cost someone real money in a real trading system.

### Rule 1 — Never use future data

A value computed for timestamp `t` may only use data with `close_time <= t`.

Forbidden anywhere in `features/`, `scoring/`, or `strategies/`:

```python
df["x"].shift(-1)          # explicit future
df.bfill()                 # backward fill = future
df.interpolate()           # fills from both sides
df.rolling(20, center=True)  # window includes the future
df["close"].max()          # whole-series aggregate on a per-bar column
```

And the subtle ones, which are the ones that actually happen:

```python
# Donchian channel that includes the current bar
h.rolling(20).max()                    # WRONG
h.rolling(20).max().shift(1)           # correct

# Filling a gap so the code does not crash
df["atr"] = df["atr"].fillna(0)        # WRONG: creates a fake observation
                                       # correct: leave NaN, let is_warm gate it

# Fitting a scaler or a model on the whole series
scaler.fit(all_data)                   # WRONG: leaks the future into the past

# Using vendor "adjusted close" that was rewritten after a later split
# WRONG: a 2024 split silently changes 2012 prices and every return computed
# from them. Pin a snapshot_id; adjust yourself from unadjusted + actions.
# See ADR-017.

# Treating an earnings gap as a normal stop-out of -1 R
# WRONG: the open is through the stop; realised R is worse than -1.
# Gate the window (ADR-020) and fill at the open.
```

### Rule 2 — Never round in your own favour

```python
qty = round(qty_raw / step) * step               # WRONG: may round up
qty = (qty_raw / step).to_integral(ROUND_DOWN) * step   # correct

# A stop rounded closer to entry silently tightens risk below the mandate
# and increases stop-out frequency. Round stops AWAY from entry.
```

### Rule 3 — Never assume the favourable intrabar path

If a bar's low touches your stop and its high touches your target, OHLC data
cannot tell you which came first. **Assume the stop.** Always. This rule alone
separates a believable backtest from a fantasy, and its effect grows with
volatility, so it inflates exactly the periods that look most exciting.

### Rule 4 — Never let a missing input make a trade more likely

Every fallback for missing data must be the *conservative* one:

| Missing | Correct fallback | Wrong fallback |
|---|---|---|
| Bars | symbol ineligible | forward-fill the last bar |
| Features | symbol ineligible | fill with 0 or the mean |
| Bin statistics | no trade | use pooled or optimistic statistics |
| Spread estimate | tier floor (pessimistic) | assume 0 |
| Funding rate | config default | assume 0 |
| Sentiment | neutral, multiplier 1.0 | assume favourable |
| Universe snapshot | no entries this cycle | use the current symbol list |
| Earnings date (equity, not ETF) | blocked (`EARNINGS_IN_WINDOW`) | assume none, allow |

### Rule 5 — Never use wall-clock time

Only `src/scout/utils/clock.py` may call `datetime.now`. Everywhere else, time
comes from the bar. A single `datetime.now()` in a backtest is a lookahead bug
with a variable, irreproducible effect, which is the worst kind.

---

## 3. Hard prohibitions

Do not do any of the following without an ADR and explicit approval.

| Prohibited | Why |
|---|---|
| Adding a dependency | The list in [`13-PROJECT_LAYOUT.md §3`](13-PROJECT_LAYOUT.md#3-pyprojecttoml) is final for v1 |
| TA-Lib, `pandas_ta`, `vectorbt`, `backtrader` | Build pain on Windows; silent formula changes between versions invalidate stored edge statistics |
| Loosening a version pin | A pandas minor release has changed `resample` boundaries, shifting every bar by one period |
| A weighted score of any kind | [`ADR/002`](ADR/002-single-ranking-statistic.md). This is the design's core decision. |
| Machine learning before M6 | [`15-ROADMAP.md`](15-ROADMAP.md) |
| A database, web framework, or Docker before M5 | Nothing to serve yet |
| `async` anywhere in the decision path | The loop is sequential and CPU-bound |
| Threads or multiprocessing in the decision path | Destroys determinism |
| A base class with inherited behaviour | Protocols only. No `AbstractStrategy`. |
| Import-time registration or plugin discovery | Makes availability depend on import order |
| A global settings object | Config is passed as a parameter |
| Deleting a symbol from `universe_candidates.txt` | Survivorship bias, applied by hand |
| Running the holdout to "check progress" | [`ADR/011`](ADR/011-holdout-lockbox.md) |
| `cost_multiplier < 1.0` | Rejected by config validation, and rightly |
| Catching `ScoutLookaheadError` | It must crash |
| A per-symbol outer loop in the engine | [`11-BACKTEST_ENGINE.md §1`](11-BACKTEST_ENGINE.md#1-the-hard-constraint) |

---

## 4. Code style

### Types

Full annotations on every function in `src/scout/`, except `research/` and
`cli/`. `mypy --strict` must pass.

```python
def estimate_cost(
    setup: Setup,
    universe_entry: UniverseEntry,
    notional_usd: float,
    risk_capital_usd: float,
    expected_bars_held: float,
    cfg: CostConfig,
) -> CostEstimate: ...
```

### Naming

| Convention | Meaning |
|---|---|
| `*_r` | units of R |
| `*_bps` | basis points |
| `*_pct` | fraction, `[0, 1]` — not 0–100 |
| `*_usd` | US dollars |
| `*_bars` | bar counts |
| `*_ts` | timezone-aware UTC datetime |
| `*_atr` | normalised by ATR |
| `is_*`, `has_*` | booleans |
| `n_*` | counts |

`_pct` meaning a fraction rather than a percentage is a deliberate, project-wide
choice. Mixing 0.15 and 15 for the same quantity is how a position ends up 100×
too large.

### Comments

Only for constraints the code cannot express.

```python
# Good: states a non-obvious constraint
# .shift(1) is mandatory: the channel must exclude the current bar or the
# breakout condition becomes self-referential.
donchian_high = h.rolling(n).max().shift(1)

# Good: states why the conservative branch exists
# OHLC cannot order intrabar events, so a bar touching both barriers is
# resolved as a stop. Assuming otherwise inflates results materially.
if hit_stop:
    return STOP

# Bad: narrates the code
# Compute the ATR
atr = atr_wilder(h, l, c, 14)

# Bad: explains your change to a reviewer
# Changed this from rolling(20) to fix the lookahead bug
```

### Pandas

```python
# No inplace. Ever. It is deprecated in spirit and impossible to reason about.
df = df.assign(atr=atr_wilder(...))          # good
df["atr"] = ...                              # acceptable on a fresh copy
df.fillna(0, inplace=True)                   # forbidden

# No chained assignment.
df[df.x > 0]["y"] = 1                        # forbidden, silently no-ops

# Always specify how= on merges, and assert the row count afterwards.
before = len(left)
out = left.merge(right, on="symbol", how="left", validate="many_to_one")
assert len(out) == before, "merge duplicated rows"
```

That last assertion catches the duplicate-row bug that silently doubles your trade
count.

### Decimal

```python
d = Decimal(str(f))            # ALWAYS via str. Decimal(0.1) is not 0.1.
f = float(d)                   # only when writing an analytics record
```

Where `Decimal` is required and where `float` is required:
[`09-PORTFOLIO_AND_RISK.md §7`](09-PORTFOLIO_AND_RISK.md#7-the-floatdecimal-boundary).
Do not Decimal-ify a DataFrame. Do not float a ledger.

### Errors

```python
# Specific and actionable
raise ScoutConfigError(
    f"unknown strategy_id {sid!r}; known: {sorted(STRATEGY_FACTORIES)}"
)

# Useless
raise ValueError("bad config")

# Never silently swallow
try:
    ...
except Exception:
    pass                       # forbidden
```

---

## 5. Testing habits

1. **Tests before implementation** for anything with a formula. Write the
   hand-computed expected value first; if you cannot, you do not yet understand
   the formula well enough to implement it.
2. **Golden values, not self-consistency.** `assert atr[5] == pytest.approx(1.234)`
   with the number computed by hand, not `assert atr[5] == atr_wilder(...)[5]`.
3. **Fixtures are deterministic.** Seeded, or hand-written. No wall-clock, no
   network, no unseeded randomness.
4. **Test the conservative branch.** For every "assume the worse case" rule in
   these docs, there is a test asserting the worse case is what happens.
5. **Every `RejectionReason` must be produced by some test.** An unreachable
   reason is dead code or a gate that never fires.

---

## 6. What to do when something looks wrong

### Results look too good

Assume a bug. In order of likelihood:

1. Missing `.shift(1)` on the Donchian channel.
2. Fill at the decision bar's close instead of the next session's open.
3. Tie rule resolving in your favour.
4. Stop/target not re-anchored on the actual entry price.
5. Universe built from currently-listed symbols (survivorship). Yahoo data.
6. Adjusted close rewritten by a later split (no pinned snapshot).
7. Earnings window not gated; gap-through losses capped at −1 R.
8. Bin statistics including setups that had not yet resolved.
9. A scaler or model fitted on the full series.
10. Cross-sectional ranks computed over the full history instead of the
    eligible set at `t`.

Run `tests/integration/test_synthetic_no_edge.py` with 20 seeds. If the system
finds an edge on a random walk, you have a leak. That test exists precisely for
this moment.

### Results look bad

That is the more likely outcome, and it is information. Before concluding there
is no edge, check the funnel: a gate rejecting 99.9% of candidates produces a flat
equity curve that looks like discipline. Then check trade count: fewer than 100
trades supports no conclusion in either direction.

Then accept the result. Do not tune. See
[`12-RESEARCH_PROTOCOL.md §8`](12-RESEARCH_PROTOCOL.md#8-anti-patterns-and-what-each-one-actually-does).

### A test is failing and you cannot see why

Do not delete the test. Do not add a tolerance to make it pass. Do not mark it
`xfail`. Several tests in [`16-TESTING.md`](16-TESTING.md) are the only mechanism
preventing a specific expensive error, and every one of them is failing for a
reason.

---

## 7. Commit and reporting discipline

**Commits.** One task per commit. Message states the task id and what changed.
Never commit `data/`, `results/`, `logs/`, or `.env`. Always commit
`experiments/registry.csv` and `experiments/holdout_lockbox.json` — they are the
project's integrity record.

**Reporting.** When a task is finished, report:

1. Which task.
2. Which files changed.
3. Which acceptance criteria are met, and which are not, individually.
4. Which tests were added and their result.
5. Anything ambiguous in the docs that you had to interpret, and how.

Point 5 is the most valuable thing you can report. A silent interpretation in the
scoring or labeling code becomes a bug nobody knows to look for.

---

## 8. Starting a new chat

Do **not** paste ADRs, scoring math, or this whole guide into every chat.
`AGENTS.md` is already injected. The agent is required to read this file and
[`15-ROADMAP.md`](15-ROADMAP.md) before writing code, then do only the lowest
unfinished task.

**One chat per task** (`M0.1`, `M0.2`, …), not one chat per whole milestone.
M1 and M3 will blow the context if you keep them in a single thread. Commit
when a task's acceptance criteria pass, then start a new chat. Git is the
memory between chats.

Paste this and fill in the two blanks:

```text
Continue Scout implementation.

Read docs/17-IMPLEMENTER_GUIDE.md, then docs/15-ROADMAP.md.
Do only the lowest-numbered unfinished task (currently: <TASK ID, e.g. M0.1>).
If a doc is ambiguous, stop and ask. Do not guess.

Last task done: <none | M0.1 — one sentence + commit hash>
```

That is enough. Do not attach `02`, `07`, or the ADR folder unless the agent
asks. It will open the document the task points at.

If you do not know the task id, omit the parenthetical — the agent must pick
the lowest unfinished item from the roadmap and confirm it before coding.

---

## 9. Ten-second orientation

| Question | Answer |
|---|---|
| What decides a trade? | `ev_net_r = ev_r_lcb - cost_r`. One number, units of R. |
| Where does it come from? | Historical outcomes of similar setups, resolved before now, with a lower confidence bound. |
| What are the weights? | There are none. That is the design. |
| What does regime do? | Gates which strategies may fire. Nothing else. |
| What does sentiment do? | Reduces size or vetoes. Never increases, never creates. |
| Where does money exist? | `portfolio/`, `backtest/ledger.py`, `execution/`. Nowhere else. |
| What is the outer loop? | Timestamps. Never symbols. |
| When do exits happen? | Exchange-native brackets, intrabar, plus a time stop. |
| What is the most important file? | `docs/07-EDGE_AND_SCORING.md`. |
| What is the most important test? | `test_synthetic_no_edge.py`. |
| What is the most important plot? | `calibration.png`. |
| What is the most likely outcome? | No edge. Building the harness that can tell you that honestly is the deliverable. |
