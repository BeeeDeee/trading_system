"""Framework reactions and gate driving."""

import pytest

from lab import stubs
from lab.framework import invocations
from lab.framework.blackboard import LabError
from lab.framework.gates import StubEvaluator, run_gate
from lab.framework.invocations import stage
from lab.framework.states import S
from lab.framework.tick import tick

from .conftest import Evaluator, submit


def build(lab, hid, evaluator):
    inv = invocations.start(lab, "builder", hid)
    stubs.builder(lab, inv)
    invocations.finish(lab, inv.id)
    tick(lab, evaluator)


def skeptic_says(lab, hid, evaluator, objections=0):
    inv = invocations.start(lab, "skeptic", hid)
    stubs.make_skeptic({hid: objections})(lab, inv)
    invocations.finish(lab, inv.id)
    tick(lab, evaluator)


def test_incomplete_card_stays_idea_with_a_question(lab, card):
    del card["falsification_criteria"]
    hid = submit(lab, card, actor="scout")
    assert lab.hypothesis(hid)["status"] == S.IDEA
    q = lab.inbox("scout", hid)
    assert q and "falsification_criteria" in q[0]["payload_json"]


def test_duplicate_stays_idea(lab, card):
    submit(lab, card)
    hid = submit(lab, {**card, "title": "Same idea under another name"}, actor="scout")
    assert lab.hypothesis(hid)["status"] == S.IDEA
    assert "duplicate of H-0001" in lab.inbox("scout", hid)[0]["payload_json"]


def test_missing_data_blocks_and_ingest_unblocks(lab, blocked_card):
    hid = submit(lab, blocked_card)
    assert lab.hypothesis(hid)["status"] == S.BLOCKED_DATA
    assert [m["type"] for m in lab.inbox("archivist")] == ["DATA_REQUEST"]
    inv = invocations.start(lab, "archivist")
    stubs.archivist(lab, inv)
    invocations.finish(lab, inv.id)
    tick(lab, None)
    assert lab.hypothesis(hid)["status"] == S.BLOCKED_DATA      # ingest is the judge's work
    tick(lab, Evaluator())
    assert lab.hypothesis(hid)["status"] == S.DATA_READY
    assert (lab.paths.lab / "data" / "sources" / "deribit_dvol_1d.py").exists()
    assert "deribit_dvol_1d" in lab.paths.catalog.read_text()


def test_period_outside_the_catalog_range_blocks(lab, card):
    card["data_requirements"][0]["period"] = ["1990-01-02", "2020-12-31"]
    hid = submit(lab, card)
    assert lab.hypothesis(hid)["status"] == S.BLOCKED_DATA


def test_infeasible_data_parks_and_chair_reopens_as_new_version(lab, blocked_card):
    hid = submit(lab, blocked_card)
    inv = invocations.start(lab, "archivist")
    stage(inv.workspace, "VERDICT", "system", hid, {"decision": "infeasible", "reason": "no free DVOL history"})
    invocations.finish(lab, inv.id)
    tick(lab, None)
    h = lab.hypothesis(hid)
    assert (h["status"], h["reject_code"]) == (S.PARKED, "infeasible")
    lab.send("VERDICT", "chair", "system", hid, {"decision": "reopen", "reason": "Deribit API found"})
    tick(lab, None)
    assert lab.hypothesis(hid)["status"] == S.IDEA
    lab.send("REVISION", "scout", "system", hid, {"card": blocked_card, "reason": "same card, data now planned"})
    tick(lab, None)
    assert lab.hypothesis(hid)["version"] == 2


def test_g0_failure_returns_to_builder_and_is_not_a_trial(lab, card):
    hid = submit(lab, card)
    ev = Evaluator({(hid, "G0"): (False, "lookahead")})
    build(lab, hid, ev)
    assert lab.hypothesis(hid)["status"] == S.DATA_READY
    assert lab.family_trials(card["family"]) == 0
    assert "lookahead" in lab.inbox("builder", hid)[0]["payload_json"]


def test_gates_run_in_order_and_count_trials(lab, card, evaluator):
    hid = submit(lab, card)
    build(lab, hid, evaluator)
    assert lab.hypothesis(hid)["status"] == S.SKEPTIC_REVIEW
    gates = [r[0] for r in lab.con.execute("SELECT gate FROM gate_results ORDER BY id")]
    assert gates == ["G0", "G1", "G2", "G3"]
    # G1: primary config; G2: one-step neighbors (lookback 252 -> 189/315, top_n 3 -> 2/4)
    assert lab.family_trials(card["family"]) == 1 + 4


def test_gate_failure_rejects_with_gate_as_stage(lab, card):
    hid = submit(lab, card)
    build(lab, hid, Evaluator({(hid, "G1"): (False, "below_benchmark")}))
    h = lab.hypothesis(hid)
    assert (h["status"], h["reject_stage"], h["reject_code"]) == (S.REJECTED, "G1", "below_benchmark")


def test_objection_loop_has_a_limit(lab, card, evaluator):
    hid = submit(lab, card)
    for _ in range(2):
        build(lab, hid, evaluator)
        skeptic_says(lab, hid, evaluator, objections=1)
        assert lab.hypothesis(hid)["status"] == S.DATA_READY
    build(lab, hid, evaluator)
    skeptic_says(lab, hid, evaluator, objections=1)
    h = lab.hypothesis(hid)
    assert (h["status"], h["reject_code"]) == (S.REJECTED, "objection_limit")
    assert lab.family_trials(card["family"]) == 3 * 5   # every re-run of G1-G2 counted


def test_holdout_once_per_hypothesis(lab, card, evaluator):
    hid = submit(lab, card)
    build(lab, hid, evaluator)
    skeptic_says(lab, hid, evaluator)
    assert lab.hypothesis(hid)["status"] == S.PARKED   # passed G4, no forward data for sharadar_sfp (Q3)
    assert lab.hypothesis(hid)["reject_code"] == "no_forward_data"
    lab.con.execute("UPDATE hypotheses SET status = 'SKEPTIC_REVIEW' WHERE id = ?", (hid,))
    with pytest.raises(LabError, match="one holdout attempt"):
        run_gate(lab, hid, "G4", evaluator)


def test_family_holdout_cap(lab, card, evaluator):
    for k in range(4):
        c = {**card, "title": f"Sector momentum variant {k}",
             "signal": {**card["signal"], "description": f"variant {k}: " + "x" * 20 + " different words" * k}}
        hid = submit(lab, c, evaluator=evaluator)
        build(lab, hid, evaluator)
        if k < 3:
            skeptic_says(lab, hid, evaluator)
    assert lab.family_holdout_attempts(card["family"]) == 3
    with pytest.raises(LabError, match="holdout cap"):
        run_gate(lab, hid, "G4", evaluator)


def test_sentinel_admits_crypto_and_g5_promotes(lab, blocked_card):
    ev = Evaluator(paper_ready=True)
    hid = submit(lab, blocked_card)
    inv = invocations.start(lab, "archivist")
    stubs.archivist(lab, inv)
    invocations.finish(lab, inv.id)
    tick(lab, ev)
    build(lab, hid, ev)
    skeptic_says(lab, hid, ev)
    assert lab.hypothesis(hid)["status"] == S.LIVE_CANDIDATE


def test_kill_switch_holds_admission(lab, blocked_card):
    ev = Evaluator()
    lab.paths.kill_switch.parent.mkdir(parents=True, exist_ok=True)
    lab.paths.kill_switch.write_text("stop")
    hid = submit(lab, blocked_card)
    inv = invocations.start(lab, "archivist")
    stubs.archivist(lab, inv)
    invocations.finish(lab, inv.id)
    tick(lab, ev)
    build(lab, hid, ev)
    skeptic_says(lab, hid, ev)
    assert lab.hypothesis(hid)["status"] == S.HOLDOUT
    assert any("kill switch" in m["payload_json"] for m in lab.inbox("human"))


def test_stub_evaluator_must_be_allowed():
    with pytest.raises(LabError, match="only runs in tests"):
        StubEvaluator()


def test_tick_without_evaluator_leaves_gate_work_waiting(lab, card):
    hid = submit(lab, card)
    inv = invocations.start(lab, "builder", hid)
    stubs.builder(lab, inv)
    invocations.finish(lab, inv.id)
    tick(lab, None)
    assert lab.hypothesis(hid)["status"] == S.DATA_READY
    tick(lab, Evaluator())
    assert lab.hypothesis(hid)["status"] == S.SKEPTIC_REVIEW
