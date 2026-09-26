"""Run the candidate grid on the development period -> matrix R[day x candidate] (spec §7.2, §9).

Usage: python scripts/run_grid.py sharadar_YYYY-MM-DD [--limit N] [--families a,b] [--universe sp500] [--delisting optimistic|pessimistic]
Output: data/derived/<snapshot>/candidates/<grid_hash>/{R,turnover,costs,exposure,n_pos}.npy,
        candidates.parquet, done.npy, meta.json. Resumable: finished columns are skipped.
Only data up to the vault boundary is used. The run is recorded in the trial registry.
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from qlab.pipeline import dev_setup
from qlab.strategies.grid import build_grid
from qlab.strategies.run import run_candidate
from qlab.validation.registry import TrialRegistry, config_hash

ap = argparse.ArgumentParser()
ap.add_argument("snapshot")
ap.add_argument("--limit", type=int, default=0, help="only the first N candidates (timing)")
ap.add_argument("--families", default="", help="comma-separated subset")
ap.add_argument("--universe", default="liq1000", choices=["liq1000", "sp500"])
ap.add_argument("--delisting", default="base", choices=["base", "optimistic", "pessimistic"])
ap.add_argument("--final", action="store_true", help="whole history incl. holdout (vault session)")
args = ap.parse_args()

setup = dev_setup(args.snapshot, universe=args.universe, delisting=args.delisting,
                  final=args.final)
robustness = args.universe != "liq1000" or args.delisting != "base" or args.final
cfg, ctx, panel, start, end = setup.cfg, setup.ctx, setup.panel, setup.start, setup.end
der = Path("data/derived") / args.snapshot
dev_start, dev_end = cfg["periods"]["development"]
if args.final:
    dev_end = panel.dates[-1]
costs_cfg = cfg["costs"]

grid = build_grid(cfg["grid"])
if args.families:
    grid = [c for c in grid if c.family in args.families.split(",")]
grid.sort(key=lambda c: c.signal_key)  # one signal in memory at a time
if args.limit:
    grid = grid[: args.limit]
run_cfg = {"snapshot": args.snapshot, "grid": cfg["grid"], "costs": costs_cfg,
           "universe": cfg["universe"], "period": [str(dev_start), str(dev_end)],
           "candidates": [c.candidate_id for c in grid]}
if robustness:  # keeps the base run's hash unchanged
    run_cfg["robustness"] = {"universe": args.universe, "delisting": args.delisting}
if args.final:
    run_cfg["final"] = True
run_hash = config_hash(run_cfg)
out = der / "candidates" / run_hash
out.mkdir(parents=True, exist_ok=True)
n_days, n_cand = end - start, len(grid)


def matrix(name: str, dtype) -> np.ndarray:
    path = out / f"{name}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r+")
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n_days, n_cand))


R, TURN, COST, EXPO, NPOS = (matrix(n, d) for n, d in (
    ("R", np.float32), ("turnover", np.float32), ("costs", np.float32),
    ("exposure", np.float32), ("n_pos", np.int16)))
done_path = out / "done.npy"
done = np.load(done_path) if done_path.exists() else np.zeros(n_cand, bool)
if not (out / "candidates.parquet").exists():
    pl.DataFrame([{"col": k, "candidate_id": c.candidate_id, "family": c.family,
                   "signal_key": c.signal_key, **{f"p_{a}": b for a, b in c.params},
                   "top_n": c.top_n, "weighting": c.weighting, "rebalance": c.rebalance,
                   "hysteresis": c.hysteresis, "overlay": c.overlay}
                  for k, c in enumerate(grid)]).write_parquet(out / "candidates.parquet")
    (out / "meta.json").write_text(json.dumps(
        {"dates": [str(d) for d in panel.dates[[start, end - 1]]], "n_days": n_days,
         "n_candidates": n_cand, "created": str(date.today())}, indent=2))
    if not args.limit and not args.families:
        TrialRegistry(Path("runs/registry/trials.sqlite")).record(
            "other" if robustness else "candidates", run_cfg, n_configs=n_cand,
            note=f"{'final ' if args.final else 'robustness ' if robustness else ''}grid run {run_hash}")

t0, ran = time.time(), 0
for k, c in enumerate(grid):
    if done[k]:
        continue
    r = run_candidate(c, ctx)
    s = slice(start, end)
    R[:, k], TURN[:, k], COST[:, k] = r.returns[s], r.turnover[s], r.costs[s]
    EXPO[:, k], NPOS[:, k] = r.exposure[s], r.n_positions[s]
    done[k] = True
    ran += 1
    if ran % 25 == 0 or k == n_cand - 1:
        for m in (R, TURN, COST, EXPO, NPOS):
            m.flush()
        np.save(done_path, done)
        rate = (time.time() - t0) / ran
        print(f"{done.sum()}/{n_cand} done, {rate:.2f}s/candidate, "
              f"eta {rate * (n_cand - done.sum()) / 60:.0f} min", flush=True)
for m in (R, TURN, COST, EXPO, NPOS):
    m.flush()
np.save(done_path, done)
print(f"finished {out} ({done.sum()}/{n_cand})")
