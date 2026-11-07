"""Research 22 search driver (dev stage only): GA, random search (N2) and permutation-null GA (N1).

    uv run python scripts/r22_search.py ga     --run A --seed 1
    uv run python scripts/r22_search.py null   --run A --index 0
    uv run python scripts/r22_search.py random --run A --seed 1 --n 300000
    ... --smoke   small budgets, writes to runs/scratch/r22/ and needs no run lock

A full-budget run refuses to start unless docs/research22/run_lock.json exists (written when the
pre-registration is frozen). Output: runs/r22/<run>/<mode>_<id>.json (summary) and .ledger.jsonl.gz.
"""

import argparse
import resource
import sys
import time
from pathlib import Path

from qlab.research22 import setup
from qlab.research22.grammar import Evaluator
from qlab.research22.nulls import run_null_ga
from qlab.research22.search import Searcher, run_ga, run_random
from qlab.research22.store import pairs_to_rows, save_result as _dump
from qlab.validation.registry import TrialRegistry


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ga", "null", "random"])
    ap.add_argument("--run", choices=["A", "B"], default="A")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--pop", type=int, default=setup.POP_SIZE)
    ap.add_argument("--gens", type=int, default=setup.N_GEN)
    ap.add_argument("--n", type=int, default=setup.N_RANDOM)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if not args.smoke and not setup.LOCK.exists():
        sys.exit(f"refusing a full-budget run: {setup.LOCK} is missing (freeze the pre-registration first)")

    root = Path("runs/scratch/r22" if args.smoke else "runs/r22") / args.run
    budget = args.n if args.mode == "random" else args.pop * (args.gens + 1)
    TrialRegistry(setup.SCRATCH_REGISTRY if args.smoke else setup.REGISTRY).record(
        "candidates", {"mode": args.mode, "run": args.run, "seed": args.seed, "index": args.index, "pop": args.pop,
                       "gens": args.gens, "n": args.n}, n_configs=budget,
        note=f"r22 {args.mode} run {args.run} (registered before computing; n_configs = evaluation budget)")
    t0 = time.time()
    compact = setup.load_compact(args.run, "dev")
    print(f"[{time.time() - t0:6.1f}s] dev {args.run}: {compact.shape[0]} weeks, K={compact.shape[1]}", flush=True)
    ev = Evaluator(compact)

    def progress(h: dict) -> None:
        print(f"[{time.time() - t0:6.1f}s] gen {h['gen']:3d} best_q25={h['best_q25']} feasible={h['feasible_share']:.2f} "
              f"ledger={h['ledger']} backtests={h['backtests']} hits={h['cache_hits']}", flush=True)

    meta = {"mode": args.mode, "run": args.run, "pop": args.pop, "gens": args.gens, "smoke": args.smoke}
    if args.mode == "ga":
        searcher = Searcher(ev)
        res = run_ga(searcher, args.seed, args.pop, args.gens, progress=progress)
        _dump(root / f"ga_{args.seed}", {**meta, "seed": args.seed, "history": res.history,
                                         "population": pairs_to_rows(res.population),
                                         "front": pairs_to_rows(res.front), "best_q25": res.best_q25(),
                                         "n_backtests": res.n_backtests}, res.ledger)
    elif args.mode == "null":
        res = run_null_ga(ev, compact, args.index, args.pop, args.gens)
        _dump(root / f"null_{args.index}", {**meta, "index": args.index, "history": res.history,
                                            "best_q25": res.best_q25(), "n_backtests": res.n_backtests}, res.ledger)
    else:
        searcher = Searcher(ev)
        run_random(searcher, args.seed, args.n)
        feas = [f.q25 for f in searcher.ledger.rows.values() if f.feasible]
        _dump(root / f"random_{args.seed}", {**meta, "seed": args.seed, "n": len(searcher.ledger),
                                             "best_q25": max(feas) if feas else None,
                                             "n_backtests": searcher.n_backtests}, searcher.ledger)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time() - t0:6.1f}s] done, peak RSS {peak_gb:.2f} GB, evaluator cache {ev.used / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
