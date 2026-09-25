# ADR-011: Trial registry and holdout lockbox

**Status:** Accepted — the highest return-on-effort mechanism in the repository

## Context

The brief asks for train / validation / out-of-sample / walk-forward methodology
and warns against reusing the test period. Every quantitative developer intends
this and most fail, because the failure is gradual and invisible:

1. Run the holdout. It looks mediocre.
2. "Just one tweak" on development. Run the holdout again.
3. Repeat six times.
4. The holdout is now training data, and there is no record of how it happened.

Separately, a reported Sharpe ratio is uninterpretable without the number of
trials behind it. "Sharpe 1.4" and "the best of 200 attempts at Sharpe 1.4" are
entirely different claims, and only one of them is evidence. Nobody remembers
their trial count honestly, and everybody underestimates it.

## Decision

Two committed artifacts, both enforced in code.

### 1. Trial registry — `experiments/registry.csv`

Every backtest run appends one row **automatically**, from `write_run_outputs`.
No manual step, no opt-out. Records `trial_id`, `run_id`, `git_sha` (with a
`--dirty` marker), `config_hash`, `split`, period, strategies, and headline
results.

`research/metrics.py` reads the trial count from this file to compute the
**deflated Sharpe ratio** (Bailey & López de Prado). `sharpe` and
`deflated_sharpe` are always reported together.

### 2. Holdout lockbox — `experiments/holdout_lockbox.json`

```json
{"budget": 3, "used": 0, "evaluations": []}
```

`run_backtest` calls `request_holdout_evaluation(reason, git_sha)` whenever the
configured period overlaps the holdout. It raises `ScoutConfigError` when the
budget is exhausted, and it refuses any holdout run from a dirty git tree.

The file is **committed**, so every evaluation is a visible commit with a stated
reason and a diff. Spending a peek silently is not possible.

Bypass requires `--force-holdout`, which prints a large warning, records
`forced: true`, and consumes budget anyway.

Budget of 3, allocated in advance:

1. M3 baseline — the answer to "does this work at all".
2. M4 best development candidate.
3. Reserve — sentiment promotion, or a final pre-go-live check.

## Consequences

- The trial count becomes a fact rather than a guess, which makes the deflated
  Sharpe computable and therefore makes the headline number honest.
- Holdout reuse becomes visible in git history. Self-deception requires a
  deliberate, auditable act.
- A dirty tree cannot produce a holdout result, so every holdout result is
  reproducible.
- Cost: about 40 lines of code and a small amount of friction at exactly the
  moment friction is valuable.
- Cost: three evaluations may feel restrictive. That is the mechanism working. If
  a fourth seems necessary, the honest options are to accept the development
  result as a heavily discounted estimate, or to wait for real forward data.

## Alternatives rejected

**Discipline without tooling.** Everyone intends this. Very few achieve it, and
nobody can prove afterwards that they did.

**Rolling walk-forward with no fixed holdout.** The estimator is already
walk-forward (ADR-003), so the development period is an honest walk-forward of
*it*. But thresholds, parameters, gates, and bin definitions are chosen by a human
looking at development results, and only a never-touched period catches that.

**A larger budget, say 10.** Ten evaluations with 30% of the sample is enough to
overfit the holdout meaningfully. Three is small enough to force the discipline of
finishing a candidate before testing it.

**Automatic parameter optimisation against the holdout.** The failure mode this
ADR exists to prevent, fully automated.

## Revisit trigger

None. If the budget is exhausted, get more data — do not raise the budget.
