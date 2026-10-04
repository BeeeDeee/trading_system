"""The walking skeleton end to end: one hand-entered hypothesis through every stage."""

import yaml

from lab.demo import run_demo
from lab.framework import report


def test_demo_walks_every_stage(tmp_path):
    lab = run_demo(tmp_path)
    assert {h["id"]: h["status"] for h in lab.hypotheses()} == {"H-0001": "LIVE_CANDIDATE", "H-0002": "REJECTED"}

    path = [t["to"] for t in lab.history("H-0001")]
    assert path == ["IDEA", "SPECIFIED", "BLOCKED_DATA", "DATA_READY", "IMPLEMENTED", "GATE_1", "GATE_2",
                    "GATE_3", "SKEPTIC_REVIEW", "DATA_READY", "IMPLEMENTED", "GATE_1", "GATE_2", "GATE_3",
                    "SKEPTIC_REVIEW", "HOLDOUT", "PAPER", "LIVE_CANDIDATE"]
    card = yaml.safe_load(lab.paths.card("H-0001").read_text())
    assert card["status"] == "LIVE_CANDIDATE" and len(card["history"]) == len(path)

    h2 = lab.hypothesis("H-0002")
    assert (h2["reject_stage"], h2["reject_code"]) == ("G2", "param_isolated_peak")

    agents = {e["actor"] for e in report.trace(lab, "H-0001") if e["kind"] == "invocation"}
    assert agents == {"archivist", "builder", "skeptic"}
    outcomes = {r["outcome"] for r in report.invocations(lab)}
    assert outcomes == {"applied"}
    assert lab.con.execute("SELECT COUNT(*) FROM gate_results WHERE evaluator != 'stub'").fetchone()[0] == 0
