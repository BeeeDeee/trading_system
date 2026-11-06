"""Research 21 harness (prereg §8 F5).

Usage:
  python scripts/r21_run.py smoke                  # real data, S0 only and partition SHARES, scratch registry, nothing in docs/
  python scripts/r21_run.py freeze                 # register every configuration, write docs/research21/run_lock.json
  python scripts/r21_run.py full [--rerun "<reason>"]   # the single evaluation on the real data
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from qlab.research17.forward import rule_hash
from qlab.research21 import pipeline as PL
from qlab.research21 import setup as S
from qlab.research21 import strategy as ST
from qlab.validation.registry import TrialRegistry, current_commit

RULE_FILES = sorted(str(p) for p in Path("src/qlab/research21").glob("*.py")) + [
    "src/qlab/research5/sim.py", "src/qlab/research5/costs.py", "src/qlab/research19/signals.py",
    "docs/research21/PREREGISTRATION.md", "scripts/r21_run.py", "scripts/r21_build_features.py"]


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json, allow_nan=True))


def registered(entry: dict) -> dict:
    return {"id": entry["id"], "role": entry["role"], **entry["config"]}


def register_all(reg: TrialRegistry) -> int:
    for e in PL.plan():
        reg.record("candidates" if e["counted"] else "other", registered(e), 1, note=f"research21 {e['id']} ({e['role']})")
    return len(PL.plan())


def clean_rule_files() -> bool:
    return not subprocess.run(["git", "status", "--porcelain", "--", *RULE_FILES], capture_output=True, text=True,
                              check=True).stdout.strip()


action = sys.argv[1] if len(sys.argv) > 1 else ""
if action == "smoke":
    S.SCRATCH_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    if S.SCRATCH_REGISTRY.exists():
        S.SCRATCH_REGISTRY.unlink()
    n = register_all(TrialRegistry(S.SCRATCH_REGISTRY))
    t0 = time.time()
    out = PL.smoke(ST.load_context())
    dump(out, Path("runs/scratch/research21_smoke.json"))
    print(json.dumps(out, indent=1, default=_json))
    print(f"smoke ok: {n} configs registered (scratch), {time.time() - t0:.0f}s")
    sys.exit(0)

if action == "freeze":
    if S.LOCK.exists():
        raise SystemExit(f"{S.LOCK} already exists")
    if not clean_rule_files():
        raise SystemExit("rule files have uncommitted changes: commit them before the freeze")
    reg = TrialRegistry(S.REGISTRY)
    if reg.total_configs():
        raise SystemExit(f"{S.REGISTRY} is not empty")
    n = register_all(reg)
    dump({"rule_hash": rule_hash(RULE_FILES), "files": RULE_FILES, "commit": current_commit(), "n_configs": n,
          "n_counted": sum(e["counted"] for e in PL.plan()), "frozen_at": datetime.now(timezone.utc).isoformat()}, S.LOCK)
    print(f"frozen {n} configurations, rule_hash {json.loads(S.LOCK.read_text())['rule_hash']}")
    sys.exit(0)

if action == "full":
    rerun = sys.argv[sys.argv.index("--rerun") + 1] if "--rerun" in sys.argv else None
    if not S.LOCK.exists():
        raise SystemExit("not frozen: run `freeze` first")
    lock = json.loads(S.LOCK.read_text())
    if lock["rule_hash"] != rule_hash(lock["files"]) or lock["files"] != RULE_FILES:
        raise SystemExit("rule files changed since the freeze")
    if not clean_rule_files():
        raise SystemExit("rule files have uncommitted changes")
    reg = TrialRegistry(S.REGISTRY)
    missing = [e["id"] for e in PL.plan() if not reg.has(registered(e))]
    if missing:
        raise SystemExit(f"configurations missing in the registry: {missing}")
    dest = S.RESULTS / "full.json"
    if dest.exists() and not rerun:
        raise SystemExit(f"{dest} exists: the evaluation is run once (use --rerun '<reason>' after a crash)")
    if rerun:
        reg.record("other", {"study": "research21", "rerun": rerun, "lock": lock["rule_hash"]}, 0, note=f"research21 rerun: {rerun}")
        print("rerun logged in the registry; add a row to PREREGISTRATION.md §11 (decision log)")
    t0 = time.time()
    out = PL.evaluate(ST.load_context())
    out["provenance"] = {"commit": current_commit(), "rule_hash": rule_hash(RULE_FILES), "snapshot": S.SHARADAR,
                         "created_at": datetime.now(timezone.utc).isoformat(), "seconds": round(time.time() - t0, 1),
                         "lock_commit": lock["commit"]}
    dump(out, dest)
    print(json.dumps(out["verdict"], indent=1))
    print(f"done in {time.time() - t0:.0f}s")
    sys.exit(0)

raise SystemExit(__doc__)
