"""Research 19 harness (PLAN §6, §8 F6).

Usage:
  python scripts/r19_run.py smoke                  # synthetic data, scratch registry, nothing is written to docs/
  python scripts/r19_run.py freeze                 # register every configuration, write docs/research19/run_lock.json
  python scripts/r19_run.py full [--rerun "<reason>"]   # the single evaluation on the real data
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from qlab.research17.forward import rule_hash
from qlab.research19 import pipeline as PL
from qlab.research19 import setup as S
from qlab.validation.registry import TrialRegistry, config_hash, current_commit

RULE_FILES = sorted(str(p) for p in Path("src/qlab/research19").glob("*.py")) + [
    "src/qlab/engine/vector.py", "src/qlab/research5/sim.py", "docs/research19/PREREGISTRATION.md",
    "scripts/r19_run.py"]
OUTPUTS = ("h1a", "h1b", "h2", "h3a", "h3b", "h3c", "h4")


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json, allow_nan=True))


def registered(entry: dict) -> dict:
    return {"id": entry["id"], "role": entry["role"], **entry["config"]}


def register_all(reg: TrialRegistry) -> int:
    for e in PL.plan():
        reg.record("candidates" if e["counted"] else "other", registered(e), 1, note=f"research19 {e['id']} ({e['role']})")
    return len(PL.plan())


def strip(out: dict) -> dict:
    """Drop big arrays (daily return series) from the committed JSON; keep metrics, criteria, controls."""
    res = {}
    for k, v in out.items():
        res[k] = {kk: vv for kk, vv in v.items() if kk != "returns"} if isinstance(v, dict) else v
    return res


def clean_rule_files() -> bool:
    return not subprocess.run(["git", "status", "--porcelain", "--", *RULE_FILES], capture_output=True, text=True,
                              check=True).stdout.strip()


def provenance(lock: dict | None, data: PL.Data, seconds: float) -> dict:
    return {"commit": current_commit(), "rule_hash": rule_hash(RULE_FILES), "snapshots": data.snapshots,
            "created_at": datetime.now(timezone.utc).isoformat(), "seconds": round(seconds, 1),
            "lock_commit": None if lock is None else lock["commit"]}


action = sys.argv[1] if len(sys.argv) > 1 else ""
if action == "smoke":
    scratch = Path("runs/scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    if S.SCRATCH_REGISTRY.exists():
        S.SCRATCH_REGISTRY.unlink()
    n = register_all(TrialRegistry(S.SCRATCH_REGISTRY))
    t0 = time.time()
    out = PL.evaluate(PL.synthetic_data())
    dump(strip(out), scratch / "research19_smoke.json")
    print(f"smoke ok: {n} configs registered (scratch), evaluated in {time.time() - t0:.1f}s; "
          f"output runs/scratch/research19_smoke.json (synthetic numbers are meaningless)")
    print("shapes:", {k: sorted(v) if isinstance(v, dict) else type(v).__name__ for k, v in out.items() if k in OUTPUTS})
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
    S.LOCK.parent.mkdir(parents=True, exist_ok=True)
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
        reg.record("other", {"study": "research19", "rerun": rerun, "lock": lock["rule_hash"]}, 0,
                   note=f"research19 rerun: {rerun}")
        print("rerun logged in the registry; add a row to PREREGISTRATION.md §9 (decision log)")
    t0 = time.time()
    data = PL.load_real()
    out = PL.evaluate(data)
    prov = provenance(lock, data, time.time() - t0)
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    nulls = {"h1a_shift_sharpes": out["h1a"]["null_sharpes"], "h1b_shift_sharpes": out["h1b"]["null_sharpes"],
             "h2_c2": out["h2"]["controls"]}
    for k in OUTPUTS:
        dump({**strip({k: out[k]})[k], "provenance": prov}, S.RESULTS / f"{k}.json")
    dump(nulls, S.RESULTS / "nulls.json")
    dump({**strip(out), "verdict": out["verdict"], "provenance": prov}, dest)
    print(json.dumps(out["verdict"], indent=1))
    print(f"done in {time.time() - t0:.0f}s")
    sys.exit(0)

raise SystemExit(__doc__)
