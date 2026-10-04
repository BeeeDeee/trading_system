"""The permission tables themselves: principle 1 (the judge is code) is a property of these tables."""

import pytest

from lab.framework import states
from lab.framework.blackboard import LabError
from lab.framework.states import LLM_AGENTS, S

from .conftest import submit

JUDGED = {S.IMPLEMENTED, S.GATE_1, S.GATE_2, S.GATE_3, S.SKEPTIC_REVIEW, S.HOLDOUT, S.PAPER, S.LIVE_CANDIDATE}


def test_no_llm_agent_can_move_a_hypothesis_into_a_judged_state():
    for (frm, to), actors in states.TRANSITIONS.items():
        if to in JUDGED:
            assert not actors & set(LLM_AGENTS), f"{frm}->{to} allows {actors & set(LLM_AGENTS)}"


def test_no_llm_agent_can_pass_or_skip_the_data_check():
    for (frm, to), actors in states.TRANSITIONS.items():
        if to == S.DATA_READY and frm != S.SKEPTIC_REVIEW:
            assert actors == {"system"}


def test_skeptic_can_only_block_what_it_reviews():
    moves = {(frm, to) for (frm, to), actors in states.TRANSITIONS.items() if "skeptic" in actors}
    assert moves == {(S.SKEPTIC_REVIEW, S.REJECTED), (S.SKEPTIC_REVIEW, S.DATA_READY), (S.SKEPTIC_REVIEW, S.IDEA)}


def test_librarian_and_steward_never_change_states():
    for actors in states.TRANSITIONS.values():
        assert "librarian" not in actors and "steward" not in actors


def test_terminal_states_only_reopen_from_parked():
    for frm, to in states.TRANSITIONS:
        if frm in states.TERMINAL:
            assert (frm, to) == (S.PARKED, S.IDEA)


def test_every_gate_target_has_a_gate_and_a_source_state():
    for state, gate in states.GATE_FOR.items():
        assert (states.GATE_FROM[gate], state) in states.TRANSITIONS


def test_transition_refuses_unlisted_actor(lab, card):
    hid = submit(lab, card)
    with pytest.raises(LabError, match="cannot move"):
        lab.transition(hid, S.IMPLEMENTED, "builder", "I am done")
    with pytest.raises(LabError, match="cannot move"):
        lab.transition(hid, S.REJECTED, "librarian", "boring")


def test_gatekeeper_cannot_enter_a_gate_state_without_a_passing_result(lab, card):
    hid = submit(lab, card)
    assert lab.hypothesis(hid)["status"] == S.DATA_READY
    with pytest.raises(LabError, match="no passing G0"):
        lab.transition(hid, S.IMPLEMENTED, "gatekeeper", "forged")


def test_human_override_is_logged(lab, card):
    hid = submit(lab, card)
    lab.transition(hid, S.PARKED, "human", "not now", reason_code="owner")
    h = lab.hypothesis(hid)
    assert (h["status"], h["reject_stage"], h["reject_code"]) == ("PARKED", "DATA_READY", "owner")
    assert lab.history(hid)[-1]["agent"] == "human"
