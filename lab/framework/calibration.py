"""Calibration of the judge on real data: strategies whose verdict is already known from the owner's earlier
studies are run through the real gates in a throwaway lab (own lab.db, no trials in the production registry).

The canaries test the judge on synthetic data; this tests it on the real thing. The questions: does a strategy
that was good in the dev period reach the later gates (is the judge not too strict), do the ones that failed in
earlier studies fail here too (is it not too lenient), and does a dev-good strategy that failed its holdout
fail G4 as well. Not part of the canaries: it needs the data and takes an hour. Results are committed as JSON
under docs/research-lab/calibration/.

Fixtures: lab/calibration/<name>/{card.yaml, strategy.py, expected.yaml}. The Skeptic step is a stub
(`clean_review`): the point is the gates, not the review.
"""

import json
import shutil
import time
from pathlib import Path

import yaml

from lab.framework import canaries as C
from lab.framework.evaluator import QlabEvaluator
from lab.framework.paths import LAB_DIR

CALIBRATION = LAB_DIR / "calibration"
OUT = LAB_DIR.parent / "docs" / "research-lab" / "calibration"


def names() -> list[str]:
    return sorted(p.name for p in CALIBRATION.iterdir() if p.is_dir())


def run(name: str, root: Path, through_holdout: bool = True) -> dict:
    d = CALIBRATION / name
    card = yaml.safe_load((d / "card.yaml").read_text())
    expected = yaml.safe_load((d / "expected.yaml").read_text())
    shutil.rmtree(root, ignore_errors=True)
    lab = C._lab(root)
    ev = QlabEvaluator(require_canaries=False)
    t0 = time.monotonic()
    hid = C.submit_and_build(lab, card, d / "strategy.py", ev)
    if through_holdout and lab.hypothesis(hid)["status"] == "SKEPTIC_REVIEW":
        C.clean_review(lab, hid, ev)
    h = lab.hypothesis(hid)
    gates = []
    for r in lab.con.execute("SELECT gate, passed, reason_code, metrics_json FROM gate_results ORDER BY id"):
        m = json.loads(r["metrics_json"])
        gates.append({"gate": r["gate"], "passed": bool(r["passed"]), "reason": r["reason_code"],
                      "failed_checks": [k for k, c in (m.get("checks") or {}).items() if c.get("pass") is False],
                      "checks": {k: c.get("value") for k, c in (m.get("checks") or {}).items()},
                      "headline": {k: m[k] for k in ("sharpe", "bench_sharpe", "benchmark", "cagr", "bench_cagr",
                                                     "max_dd", "mean_exposure", "dsr", "n_trials") if k in m}})
    return {"name": name, "status": h["status"], "died_at": h["reject_stage"], "reason": h["reject_code"],
            "gates": gates, "expected": expected, "seconds": round(time.monotonic() - t0)}


def save(result: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{result['name']}.json"
    f.write_text(json.dumps(result, indent=1, default=str) + "\n")
    return f
