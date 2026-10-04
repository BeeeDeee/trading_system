"""Canaries: strategies that test the judge itself. Run them after every change of lab/framework or gates.yaml
(`uv run lab canaries`, also part of pytest).

Each canary goes through the real pipeline (card -> Builder invocation -> IMPL_DONE -> tick with the real
evaluator) in a throwaway lab on the synthetic market. Expectations:
    lookahead   G0 fails with g0_lookahead
    hindsight   G0 fails with g0_static
    noise       20 seeds, none reaches SKEPTIC_REVIEW
    overfit     best of 2000 random timing seeds picked on dev; it must fool G1 and die at G2 or G3, with and
                without the 2000 trials recorded
    positive    passes G0-G3, then G4 after a clean Skeptic review (a judge that rejects everything is broken too)
    replica_*   the benchmark itself on real data (60/40, BTC) fails G1
"""

import copy
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from lab.framework import catalog, invocations
from lab.framework.api import period_starts
from lab.framework.blackboard import Lab
from lab.framework.db import now
from lab.framework.data import SYNTHETIC_DATASET, synthetic
from lab.framework.evaluator import QlabEvaluator
from lab.framework.invocations import stage
from lab.framework.paths import sandbox_paths
from lab.framework.states import S
from lab.framework.tick import tick

CANARIES = Path(__file__).resolve().parents[1] / "canaries"
QUICK = {"G1": {"bootstrap_samples": 300}, "G2": {"random_entry": {"runs": 40}}}


@dataclass
class Result:
    name: str
    ok: bool
    detail: str


def _lab(root: Path) -> Lab:
    lab = Lab.open(sandbox_paths(root))
    catalog.add(lab.paths.catalog, SYNTHETIC_DATASET)
    return lab


def submit_and_build(lab: Lab, card: dict, source: Path, evaluator, params: dict | None = None) -> str:
    """Card in as human, strategy in through a Builder invocation, then the deterministic phase."""
    lab.send("NEW_HYPOTHESIS", "human", "system", None, {"card": card})
    tick(lab, evaluator)
    hid = lab.hypotheses()[-1]["id"]
    inv = invocations.start(lab, "builder", hid)
    shutil.copy(source, inv.workspace / "strategy" / "strategy.py")
    stage(inv.workspace, "IMPL_DONE", "gatekeeper", hid, {"files": ["strategy/strategy.py"], "summary": "canary"})
    invocations.finish(lab, inv.id)
    tick(lab, evaluator)
    return hid


def last_result(lab: Lab, hid: str, gate: str):
    return lab.con.execute("SELECT * FROM gate_results WHERE hypothesis_id = ? AND gate = ? ORDER BY id DESC LIMIT 1",
                           (hid, gate)).fetchone()


def _card(name: str) -> dict:
    return yaml.safe_load((CANARIES / name / "card.yaml").read_text())


def clean_review(lab: Lab, hid: str, evaluator) -> None:
    inv = invocations.start(lab, "skeptic", hid)
    checklist = yaml.safe_load(lab.paths.gates.read_text())["skeptic"]["checklist"]
    stage(inv.workspace, "VERDICT", "system", hid, {"decision": "no_objection", "reason": "canary review",
          "checklist": {c: {"status": "ok", "evidence": "canary, nothing to review"} for c in checklist}})
    invocations.finish(lab, inv.id)
    tick(lab, evaluator)


def run_integrity(name: str, root: Path, expected_code: str, overrides=QUICK) -> Result:
    lab = _lab(root)
    ev = QlabEvaluator(overrides, require_canaries=False)
    hid = submit_and_build(lab, _card(name), CANARIES / name / "strategy.py", ev)
    g0 = last_result(lab, hid, "G0")
    ok = g0 is not None and not g0["passed"] and g0["reason_code"] == expected_code \
        and lab.hypothesis(hid)["status"] == S.DATA_READY
    return Result(name, ok, f"G0 {g0['reason_code'] if g0 else 'missing'} (expected {expected_code})")


def run_noise(root: Path, seeds: int = 20, overrides=QUICK) -> Result:
    lab = _lab(root)
    ev = QlabEvaluator(overrides, require_canaries=False)
    passed, deaths = [], {}
    for seed in range(seeds):
        card = copy.deepcopy(_card("noise"))
        card["title"] = f"Canary: random weekly weights, seed {seed}"
        card["family"] = f"canary-noise-{seed}"   # its own family: the harshest case, N stays at the floor
        card["signal"]["params"]["seed"] = {"value": seed, "grid": [seed - 1, seed, seed + 1] if seed else [0, 1, 2]}
        hid = submit_and_build(lab, card, CANARIES / "noise" / "strategy.py", ev)
        h = lab.hypothesis(hid)
        if h["status"] != S.REJECTED:
            passed.append((seed, h["status"]))
        deaths[h["reject_stage"]] = deaths.get(h["reject_stage"], 0) + 1
    return Result("noise", not passed, f"{seeds} seeds, died at {deaths}, survived {passed}")


def pick_overfit_seed(n: int = 2000) -> tuple[int, float]:
    """Seed with the best dev Sharpe excess over MKT among n random weekly timing patterns of the overfit canary,
    using a fast replica of the engine (decided at a week's first close, in force from the next day's intraday
    return; overnight one day later; 5 bps per switch)."""
    v = synthetic()
    dev = int(np.searchsorted(v.dates, np.datetime64(SYNTHETIC_DATASET["holdout_from"])))
    j = v.instruments.index("MKT")
    oc = v.ret_oc[:dev, j]                               # MKT has no overnight return (index traded at the open)
    week_start = period_starts(v.dates[:dev], "W")       # same schedule as the strategy (Monday-based)
    last = np.maximum.accumulate(np.where(week_start, np.arange(dev), 0))
    row = np.r_[0, last[:-1]]                            # day d earns the decision taken up to d-1
    sh = lambda x: x.mean() / x.std() * np.sqrt(252)  # noqa: E731
    target = sh(oc[1:])
    best, best_s = 0, -np.inf
    for seed in range(n):
        on = (np.random.default_rng(seed).random(dev) < 0.5).astype(float)[row]
        switch = np.abs(np.diff(on, prepend=0.0))
        day = (on * oc - switch * 0.0005)[1:]
        s = sh(day) - target
        if s > best_s:
            best, best_s = seed, s
    return best, float(best_s)


def run_overfit(root: Path, record_trials: bool, overrides=QUICK, n: int = 2000) -> Result:
    lab = _lab(root)
    seed, s = pick_overfit_seed(n)
    card = _card("overfit")
    card["signal"]["params"]["seed"] = {"value": seed, "grid": [seed - 1, seed, seed + 1]}
    if record_trials:   # the honest accounting: every searched seed is a trial of the family
        with lab.con:
            lab.con.executemany(
                "INSERT INTO trials (family, hypothesis_id, version, run_id, config_sha256, sharpe, n_obs, ts) "
                "VALUES ('canary-overfit', 'search', 0, 0, ?, NULL, 0, '')", [(f"seed{k}",) for k in range(n)])
    ev = QlabEvaluator(overrides, require_canaries=False)
    hid = submit_and_build(lab, card, CANARIES / "overfit" / "strategy.py", ev)
    h = lab.hypothesis(hid)
    g1 = last_result(lab, hid, "G1")
    # The canary is only informative if the lucky seed fools G1; then G2 or G3 must stop it.
    ok = bool(g1 and g1["passed"]) and h["status"] == S.REJECTED and h["reject_stage"] in ("G2", "G3")
    return Result(f"overfit ({'trials recorded' if record_trials else 'trials hidden'})", ok,
                  f"seed {seed} (proxy Sharpe excess {s:.2f}), G1 {'passed' if g1 and g1['passed'] else 'failed'}"
                  f" -> {h['status']} at {h['reject_stage']}: {h['reject_code']}")


def run_positive(root: Path, overrides=QUICK) -> Result:
    lab = _lab(root)
    ev = QlabEvaluator(overrides, require_canaries=False)
    hid = submit_and_build(lab, _card("positive"), CANARIES / "positive" / "strategy.py", ev)
    status = lab.hypothesis(hid)["status"]
    if status != S.SKEPTIC_REVIEW:
        h = lab.hypothesis(hid)
        return Result("positive", False, f"stopped at {status} {h['reject_stage']}: {h['reject_code']}")
    clean_review(lab, hid, ev)
    h = lab.hypothesis(hid)
    return Result("positive", h["status"] == S.PAPER, f"-> {h['status']} {h['reject_code'] or ''}".strip())


def run_replica(name: str, root: Path, overrides=QUICK) -> Result:
    """The benchmark itself on real data must fail G1 (skipped when the data snapshot is not on this machine)."""
    from lab.framework.data import DATA_ROOT
    if not DATA_ROOT.exists():
        return Result(name, True, "skipped: no data snapshot")
    lab = _lab(root)
    ev = QlabEvaluator(overrides, require_canaries=False)
    hid = submit_and_build(lab, _card(name), CANARIES / name / "strategy.py", ev)
    h = lab.hypothesis(hid)
    return Result(name, h["status"] == S.REJECTED and h["reject_stage"] == "G1",
                  f"-> {h['status']} at {h['reject_stage']}: {h['reject_code']}")


def framework_sha(gates_path: Path) -> str:
    """Hash of everything that makes up the judge: framework code and schemas, canaries, gates.yaml."""
    h = hashlib.sha256()
    roots = [Path(__file__).parent, CANARIES]
    files = sorted(f for r in roots for f in r.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
    for f in files + [gates_path]:
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def record(lab: Lab, results: list[Result]) -> bool:
    ok = all(r.ok for r in results)
    with lab.con:
        lab.con.execute("INSERT INTO canary_runs (framework_sha256, ok, detail_json, ts) VALUES (?, ?, ?, ?)",
                        (framework_sha(lab.paths.gates), int(ok), json.dumps([r.__dict__ for r in results]), now()))
    return ok


def passed_for_current(lab: Lab) -> bool:
    row = lab.con.execute("SELECT ok FROM canary_runs WHERE framework_sha256 = ? ORDER BY id DESC LIMIT 1",
                          (framework_sha(lab.paths.gates),)).fetchone()
    return bool(row and row["ok"])


def run_all(quick: bool = True) -> list[Result]:
    overrides = QUICK if quick else {}
    with tempfile.TemporaryDirectory(prefix="lab-canaries-") as tmp:
        root = Path(tmp)
        return [run_integrity("lookahead", root / "lookahead", "g0_lookahead", overrides),
                run_integrity("hindsight", root / "hindsight", "g0_static", overrides),
                run_noise(root / "noise", overrides=overrides),
                run_overfit(root / "overfit1", True, overrides),
                run_overfit(root / "overfit2", False, overrides),
                run_positive(root / "positive", overrides),
                run_replica("replica_6040", root / "r6040", overrides),
                run_replica("replica_btc", root / "rbtc", overrides)]
