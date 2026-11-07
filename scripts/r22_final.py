"""Research 22 late stage (gate G6): prepare -> freeze -> run. The late stage is read ONCE per run through the vault.

    uv run python scripts/r22_final.py prepare --run A [--d "<tree repr>"]   # dev/validation results only
    uv run python scripts/r22_final.py freeze  --run A                       # hash lock + registry entry
    QLAB_R22_METHODOLOGY=<hash> uv run python scripts/r22_final.py run --run A

`run` needs both runs frozen (A and B are opened together), a clean working tree for the research 22 files, and
the frozen hash in the environment. `--smoke` skips the vault, reads runs/scratch/r22 and evaluates on DEV in place
of the late stage (plumbing only). The late stage is "late validation", not a clean holdout.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from qlab.research22 import setup
from qlab.research22.grammar import Evaluator
from qlab.research22.late import frozen_from_json, frozen_to_json, hash_files, late_report, prepare_frozen
from qlab.research22.store import load_ledger, tree_from_repr
from qlab.validation.registry import TrialRegistry, current_commit

GUARDED = ["src/qlab/research22", "scripts/r22_search.py", "scripts/r22_select.py", "scripts/r22_final.py",
           "tests/test_research22.py", "docs/research22"]


def methodology_hash(run: str) -> str:
    files = ["docs/research22/PREREGISTRATION.md", str(setup.RESULTS / f"frozen_{run}.json"),
             *sorted(str(f) for f in Path("src/qlab/research22").glob("*.py")),
             "scripts/r22_search.py", "scripts/r22_select.py", "scripts/r22_final.py"]
    return hash_files(files)


def ensure_clean() -> None:
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *GUARDED], capture_output=True, text=True).stdout
    if dirty.strip():
        sys.exit("uncommitted changes in research 22 files (commit before the late-stage run):\n" + dirty)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "freeze", "run"])
    ap.add_argument("--run", choices=["A", "B"], default="A")
    ap.add_argument("--d", default=None, help="repr of the hand-written simple strategy D (frozen with E)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    root = Path("runs/scratch/r22" if args.smoke else "runs/r22") / args.run
    frozen_path = (root if args.smoke else setup.RESULTS) / f"frozen_{args.run}.json"

    if args.action == "prepare":
        selection = json.loads((root / "selection.json").read_text())
        sel = {**selection, "ensemble_trees": [tree_from_repr(t) for t in selection["ensemble_trees"]],
               "random_ensemble_trees": [tree_from_repr(t) for t in selection.get("random_ensemble_trees", [])],
               "champion_tree": tree_from_repr(selection["champion_tree"]) if selection.get("champion_tree") else None}
        ledger = load_ledger(next(iter(sorted(root.glob("random_*.ledger.jsonl.gz")))))
        for p in sorted(root.glob("random_*.ledger.jsonl.gz"))[1:]:
            part = load_ledger(p)
            ledger.rows.update(part.rows)
        dev = setup.load_compact(args.run, "dev")
        frozen = prepare_frozen(sel, ledger, Evaluator(dev), dev, args.run,
                                tree_from_repr(args.d) if args.d else None)
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        frozen_path.write_text(json.dumps(frozen_to_json(frozen), indent=1, default=float))
        print(f"wrote {frozen_path}: E has {len(frozen['ensemble_trees'])} members")
        return

    if args.smoke:
        late_c = setup.load_compact(args.run, "dev")
        frozen = frozen_from_json(json.loads(frozen_path.read_text()))
        res = late_report(frozen, Evaluator(late_c), late_c, n_boot=args.n_boot)
        res["smoke_late_is_dev"] = True
        (root / f"final_{args.run}.json").write_text(json.dumps(res, indent=1, default=float))
        print("G6 (smoke, meaningless):", res.get("passed"))
        return

    vault = setup.vault(args.run)
    if args.action == "freeze":
        mh = methodology_hash(args.run)
        vault.freeze(mh)
        TrialRegistry(setup.REGISTRY).record("methodology_eval", {"run": args.run, "methodology_hash": mh}, 1,
                                             note=f"r22 freeze run {args.run}")
        print(f"frozen: {mh}")
        return

    if not all(setup.vault(r).lock_path.exists() for r in ("A", "B")):
        sys.exit("both runs (A and B) must be frozen before either late stage is opened")
    mh = os.environ.get(setup.FINAL_ENV, "")
    if mh != methodology_hash(args.run):
        sys.exit("methodology changed since freeze (or environment variable missing)")
    ensure_clean()
    vault.open_final(mh, setup.SHARADAR, current_commit())
    late_c = setup.load_compact(args.run, "late")
    frozen = frozen_from_json(json.loads(frozen_path.read_text()))
    res = late_report(frozen, Evaluator(late_c), late_c, n_boot=args.n_boot)
    res.update({"run": args.run, "snapshot": setup.SHARADAR, "commit": current_commit(), "methodology_hash": mh})
    out = setup.RESULTS / f"final_{args.run}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"G6 run {args.run}: passed={res.get('passed')} ir_1x={res.get('ir_1x')} -> {out}")


if __name__ == "__main__":
    main()
