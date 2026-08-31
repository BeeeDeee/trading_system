# M2.2 Labeling sanity

Recorded after `tests/unit/test_labeling.py::test_no_edge_series_gives_near_zero_mean_r`
passed. Seed `20260827` throughout (config default / `bootstrap_ci` default).

## Random-walk check (mandatory)

A driftless geometric random walk, 4,000 sessions, overnight σ = 0.008 and
intraday σ = 0.008. Long setups placed every 8th bar from index 20, stop at 2% of
the decision close, target at 2R (`rr = 2.0`), `max_hold_bars = 40`. OPEN rows
excluded.

| Statistic | Value | Expectation |
|---|---|---|
| n resolved | 493 | — |
| `mean(realised_r_gross)` | −0.0082 | near 0 |
| bootstrap 90% CI of the mean | [−0.1215, 0.0998] | contains 0 |
| `win_rate` (`realised_r_gross > 0`) | 0.341 | `1/(1+2) ≈ 0.333` |
| `min(realised_r_gross)` | −1.885 | below −1.0 (gap-through) |
| stop / target / time rates | 0.657 / 0.337 / 0.006 | stop ≈ `rr/(1+rr) = 0.667` on a martingale; time-stops rare at hold 40 |

The mean sits inside the bootstrap CI of zero. Win rate is within 0.01 of
`1/(1+rr)`. Stop rate is the gambler's-ruin complement of a 2:1 target, not the
0.4–0.6 band quoted in `07-EDGE_AND_SCORING.md` §3 for *Donchian-detected* setups
(those have a time stop that eats some of the stop mass). This pass is labeling
geometry on a no-edge series, not strategy detection.

`min_r < −1` is the gap-through-stop rule firing. If it were clipped at −1.0 the
rule would be missing.

## Real data — **OPEN (M2.2a, Blocked)**

This is the half of the M2.2 sanity that the random walk cannot substitute.
Tracked as [M2.2a](../15-ROADMAP.md#m22a-real-data-labeling-sanity--blocked).
Do not mark it done until the table below is filled from
`scout label --config config/development.yaml`.

Raw Sharadar ingest is on disk (`data/raw/equity/`, snapshot `20260830-sharadar`,
30 781 symbols, 1998–2026). Labeling still cannot run:

| Trigger file | Present? |
|---|---|
| `data/processed/panel/1d/*.parquet` | no — run `scout adjust` |
| `data/universe/snapshots.parquet` | no — run `scout build-universe` |
| `data/reference/benchmark_1d.parquet` | no — no M1 task writes this; schema in `04` §6.3 |

When all three exist, run label and paste per-strategy pooled stats here
(`donchian_breakout_v1` and `xsec_momentum_v1`):

| Check | `donchian_breakout_v1` | `xsec_momentum_v1` |
|---|---|---|
| n resolved | | |
| `mean(realised_r_gross)` | | |
| win rate | | |
| stop_rate | | |
| time_rate | | |
| `min(realised_r_gross)` | | |
| `mean(entry_gap_atr)` | | |

**If pooled `mean_r` exceeds about 0.4 R, that is a bug, not an edge.** Stop.
Do not start M3.6 and do not trust a real `EdgeTable` until this section is
filled and `mean_r` is not ~0.4 R.

## Interpretations (docs were incomplete, not contradictory)

These are the places the documents did not specify a unique answer. They are
called out so they can be reversed if wrong, rather than becoming silent scoring
bugs.

1. **`vol_bucket` on `resolve_setup`.** `ResolvedSetup` requires it; `Setup` does
   not carry it; the `03-INTERFACES.md` signature does not pass it. Implemented as
   a keyword-only argument defaulting to `UNKNOWN`. The CLI passes
   `FeatureRow.vol_bucket`.
2. **OPEN with no next bar.** `entry_ts` is required and not nullable. With an
   empty forward window it is set to `setup.ts` and `entry_price` is NaN.
   `realised_r_gross` is NaN. OPEN after an actual entry (data ran out mid-trade)
   keeps the real `entry_ts` / `entry_price`.
3. **Parquet `stop_price` / `target_price`.** Written as the *re-anchored* levels
   used for resolution, not the decision-bar levels on `Setup`. Distances match
   the intended R/R; the decision-bar numbers remain on `reference_price`.
4. **`adv_usd_60`.** Not a `FeatureRow` field. Joined from the latest universe
   snapshot with `snapshot.ts <= setup_ts` (backward as-of, by `asset_id`).
5. **One file per strategy.** `data/labels/setups_<strategy_id>.parquet` is a
   single parquet file. Year partitioning is not implemented; 07's volume note
   is a performance hint, and the acceptance criterion names a `.parquet` file.
6. **`LabelConfig` in the interface is `LabelingConfig`** from
   `config/schema.py`. No second type.
7. **`had_earnings_in_window`** is `gates.eligibility.earnings_in_window` — i.e.
   whether the earnings gate *would* have fired, including the conservative
   “no earnings row ⇒ blocked” fallback for non-ETFs.
