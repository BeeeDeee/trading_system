"""Gatekeeper: the deterministic gate runner. Not an LLM.

It runs one gate for one hypothesis version with an `Evaluator`, stores the run, the result and the
trials, tells the agents (GATE_RESULT) and moves the hypothesis: pass -> next state, fail -> REJECTED
(G0 failure -> stays DATA_READY and goes back to the Builder, it is not a trial).

Evaluators compute metrics. The real evaluator over qlab arrives in step 2; until then only `StubEvaluator`
exists, and it refuses to run unless explicitly allowed (tests and the demo). Every result records which
evaluator produced it, so stub results can never be mistaken for real ones.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import yaml

from lab.framework.blackboard import Lab, LabError
from lab.framework.db import Tx, dumps, now
from lab.framework.states import GATE_FOR, GATE_FROM, S

GATES = ("G0", "G1", "G2", "G3", "G4", "G5")
TARGET = {gate: state for state, gate in GATE_FOR.items()}
COUNTS_AS_TRIAL = {"G1", "G2", "G3"}   # dev-period evaluations feed the family's N


@dataclass(frozen=True)
class Trial:
    config_sha256: str
    sharpe: float | None = None
    n_obs: int | None = None


@dataclass(frozen=True)
class Outcome:
    passed: bool
    metrics: dict
    reason_code: str | None = None
    trials: list[Trial] = field(default_factory=list)


class Evaluator(Protocol):
    name: str

    def evaluate(self, lab: Lab, hid: str, card: dict, gate: str, thresholds: dict) -> Outcome: ...


class StubEvaluator:
    """Scripted outcomes for the walking skeleton. `script[(hid, gate)] = (passed, reason_code)`;
    anything not scripted passes. Each call yields as many trials as the gate would evaluate configs."""

    name = "stub"

    def __init__(self, script: dict | None = None, *, allow: bool = False):
        if not allow:
            raise LabError("the stub evaluator only runs in tests and the demo (allow=True)")
        self.script = script or {}

    def evaluate(self, lab, hid, card, gate, thresholds):
        version = lab.hypothesis(hid)["version"]
        passed, code = self.script.get((hid, gate), (True, None))
        n_configs = {"G1": 1, "G2": max(1, _grid_neighbors(card))}.get(gate, 0)
        trials = [Trial(hashlib.sha256(f"{hid}:{version}:{gate}:{k}".encode()).hexdigest()[:16], 0.0, 0)
                  for k in range(n_configs)]
        metrics = {"stub": True, "family_trials_before": lab.family_trials(card["family"])}
        return Outcome(passed, metrics, None if passed else (code or f"{gate.lower()}_failed"), trials)


def _grid_neighbors(card: dict) -> int:
    """Number of one-step neighbor configurations of the primary configuration (for trial counting)."""
    n = 0
    for p in card.get("signal", {}).get("params", {}).values():
        grid, value = p["grid"], p["value"]
        i = grid.index(value) if value in grid else 0
        n += (i > 0) + (i < len(grid) - 1)
    return n


def thresholds(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def strategy_sha(lab: Lab, hid: str) -> str | None:
    d = lab.paths.strategies / hid
    if not d.exists():
        return None
    h = hashlib.sha256()
    for f in sorted(p for p in d.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(d)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def run_gate(lab: Lab, hid: str, gate: str, evaluator: Evaluator) -> Outcome:
    if gate not in GATES:
        raise LabError(f"unknown gate {gate}")
    h = lab.hypothesis(hid)
    if h["status"] != GATE_FROM[gate]:
        raise LabError(f"{hid} is {h['status']}; {gate} runs only from {GATE_FROM[gate]}")
    card = lab.card(hid)
    config, config_sha = thresholds(lab.paths.gates)
    if gate == "G4":
        if lab.con.execute("SELECT 1 FROM gate_results WHERE hypothesis_id = ? AND gate = 'G4'",
                           (hid,)).fetchone():
            raise LabError(f"{hid} already used its one holdout attempt")
        if lab.family_holdout_attempts(card["family"]) >= config["G4"]["family_max_attempts"]:
            raise LabError(f"family {card['family']} reached its holdout cap; the owner must approve more")

    with Tx(lab.con):
        run_id = lab.con.execute(
            "INSERT INTO runs (hypothesis_id, version, gate, evaluator, code_sha256, started_at, status)"
            " VALUES (?, ?, ?, ?, ?, ?, 'running')",
            (hid, h["version"], gate, evaluator.name, strategy_sha(lab, hid), now())).lastrowid
    try:
        outcome = evaluator.evaluate(lab, hid, card, gate, config.get(gate, {}))
    except Exception as e:  # an evaluator crash is an error of the run, never a verdict
        with Tx(lab.con):
            lab.con.execute("UPDATE runs SET status = 'error', error = ?, ended_at = ? WHERE id = ?",
                            (f"{type(e).__name__}: {e}", now(), run_id))
        lab.send("ALERT", "gatekeeper", "human", hid,
                 {"severity": "warning", "text": f"{gate} run {run_id} crashed: {type(e).__name__}: {e}"})
        raise

    with Tx(lab.con):
        lab.con.execute("UPDATE runs SET status = 'done', ended_at = ? WHERE id = ?", (now(), run_id))
        lab.con.execute(
            "INSERT INTO gate_results (run_id, hypothesis_id, version, gate, passed, reason_code, metrics_json,"
            " thresholds_sha256, evaluator, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, hid, h["version"], gate, int(outcome.passed), outcome.reason_code, dumps(outcome.metrics),
             config_sha, evaluator.name, now()))
        if gate in COUNTS_AS_TRIAL:
            for t in outcome.trials:
                lab.con.execute(
                    "INSERT INTO trials (family, hypothesis_id, version, run_id, config_sha256, sharpe, n_obs, ts)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (card["family"], hid, h["version"], run_id, t.config_sha256, t.sharpe, t.n_obs, now()))

    payload = {"gate": gate, "passed": outcome.passed, "reason_code": outcome.reason_code,
               "metrics": outcome.metrics, "run_id": run_id}
    lab.send("GATE_RESULT", "gatekeeper", "builder" if gate == "G0" else "librarian", hid, payload)
    reason = f"{gate} {'passed' if outcome.passed else 'failed'} (run {run_id}, {evaluator.name})"
    if outcome.passed:
        lab.transition(hid, TARGET[gate], "gatekeeper", reason)
    elif gate != "G0":
        lab.transition(hid, S.REJECTED, "gatekeeper", reason, reason_code=outcome.reason_code, stage=gate)
    return outcome
