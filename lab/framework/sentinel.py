"""Sentinel: deterministic risk control. Not an LLM. Veto over PAPER and LIVE_CANDIDATE.

Step 1 implements admission to PAPER (kill switch, capacity, forward data). Kill rules on running paper
strategies (drawdown, tracking) come with the paper runner in step 4.
"""

from lab.framework import catalog
from lab.framework.blackboard import Lab
from lab.framework.gates import thresholds
from lab.framework.states import S


def admit_to_paper(lab: Lab, hid: str) -> S:
    """Decide HOLDOUT -> PAPER or PARKED. Returns the new state, or HOLDOUT if it has to wait."""
    if lab.hypothesis(hid)["status"] != S.HOLDOUT:
        raise ValueError(f"{hid} is not in HOLDOUT")
    config, _ = thresholds(lab.paths.gates)
    limits = config["sentinel"]

    if lab.paths.kill_switch.exists():
        lab.send("ALERT", "sentinel", "human", hid,
                 {"severity": "warning", "text": f"kill switch present, {hid} waits in HOLDOUT"})
        return S.HOLDOUT

    datasets = [r["dataset"] for r in lab.card(hid).get("data_requirements", [])]
    no_forward = catalog.missing_forward_source(lab.paths.catalog, datasets)
    if no_forward:
        reason = "no forward data source for " + ", ".join(no_forward) + " (decision Q3)"
        lab.send("VERDICT", "sentinel", "chair", hid,
                 {"decision": "park", "reason": reason, "reason_code": "no_forward_data"})
        lab.transition(hid, S.PARKED, "sentinel", reason, reason_code="no_forward_data")
        return S.PARKED

    in_paper = len(lab.hypotheses(S.PAPER))
    if in_paper >= limits["max_paper_strategies"]:
        return S.HOLDOUT  # waits for a free slot; no message every tick

    lab.send("VERDICT", "sentinel", "librarian", hid,
             {"decision": "admit", "reason": f"paper slot {in_paper + 1}/{limits['max_paper_strategies']}"})
    lab.transition(hid, S.PAPER, "sentinel", "admitted to paper trading")
    return S.PAPER
