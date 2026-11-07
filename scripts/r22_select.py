"""Research 22 selection and diagnostics: gates G1-G4 + ensemble E (`select`) and the DNA gate G5 (`dna`).

    uv run python scripts/r22_select.py select --run A
    uv run python scripts/r22_select.py dna    --run A --n-perm 5000
    ... --smoke   reads/writes runs/scratch/r22 and uses the DEV stage in place of validation: it checks the
                  plumbing only and never touches validation data

Inputs are the outputs of scripts/r22_search.py (ga_*, null_*, random_*). A full run refuses to start unless
docs/research22/run_lock.json exists. The late stage is never read here.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from qlab.research22 import setup
from qlab.research22.grammar import Evaluator
from qlab.research22.search import Ledger
from qlab.research22.selection import gate_g5, select_run
from qlab.research22.store import load_ga_result, load_ledger


def _jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, tuple):
        return repr(x)
    raise TypeError(f"not serialisable: {type(x)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "dna"])
    ap.add_argument("--run", choices=["A", "B"], default="A")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--pbo-sample", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    if not args.smoke and not setup.LOCK.exists():
        sys.exit(f"refusing a full run: {setup.LOCK} is missing (freeze the pre-registration first)")

    root = Path("runs/scratch/r22" if args.smoke else "runs/r22") / args.run
    ga = [load_ga_result(p) for p in sorted(root.glob("ga_*.json"))]
    if not ga:
        sys.exit(f"no GA results in {root}")
    null_best = [json.loads(p.read_text())["best_q25"] for p in sorted(root.glob("null_*.json"))]
    random_ledger = Ledger()
    for p in sorted(root.glob("random_*.ledger.jsonl.gz")):
        part = load_ledger(p)
        random_ledger.rows.update(part.rows)
        random_ledger.first_gen.update(part.first_gen)
    print(f"{len(ga)} GA seeds, {len(null_best)} null runs, {len(random_ledger)} random trees", flush=True)

    dev = setup.load_compact(args.run, "dev")
    val = dev if args.smoke else setup.load_compact(args.run, "validation")
    ev_dev = Evaluator(dev)
    ev_val = ev_dev if args.smoke else Evaluator(val)

    if args.mode == "select":
        out = select_run(ga, random_ledger, null_best, ev_dev, dev, ev_val, val, n_boot=args.n_boot,
                         pbo_sample=args.pbo_sample, seed=args.seed)
        for key in ("ensemble_trees", "random_ensemble_trees"):
            out[key] = [repr(t) for t in out[key]]
        out["champion_tree"] = repr(out["champion_tree"]) if out["champion_tree"] is not None else None
        name = "selection.json"
    else:
        out = gate_g5(ga, random_ledger, ev_dev, dev, ev_val, val, n_perm=args.n_perm, seed=args.seed)
        name = "dna.json"
    out["smoke_validation_is_dev"] = bool(args.smoke)
    (root / name).write_text(json.dumps(out, indent=1, default=_jsonable))
    gates = {k: v.get("passed") if isinstance(v, dict) else v for k, v in out.items() if k.startswith("G") or k == "passed"}
    print("gates:", gates, "->", root / name, flush=True)


if __name__ == "__main__":
    main()
