import sqlite3

import pytest
import yaml

from lab.framework.blackboard import LabError

from .conftest import submit


def test_append_only_tables(lab, card):
    hid = submit(lab, card)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        lab.con.execute("UPDATE transitions SET reason = 'x' WHERE hypothesis_id = ?", (hid,))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        lab.con.execute("DELETE FROM transitions")
    with pytest.raises(sqlite3.IntegrityError):
        lab.con.execute("DELETE FROM messages")


def test_messages_can_only_be_marked_handled_once_and_never_edited(lab):
    mid = lab.send("QUESTION", "human", "scout", None, {"question": "anything on carry?"})
    with pytest.raises(sqlite3.IntegrityError):
        lab.con.execute("UPDATE messages SET payload_json = '{}' WHERE id = ?", (mid,))
    lab.mark_handled([mid], "test")
    with pytest.raises(sqlite3.IntegrityError):
        lab.con.execute("UPDATE messages SET handled_by = 'other' WHERE id = ?", (mid,))


def test_send_is_idempotent_on_key(lab):
    a = lab.send("QUESTION", "human", "scout", None, {"question": "first?"}, key="k1")
    b = lab.send("QUESTION", "human", "scout", None, {"question": "first?"}, key="k1")
    assert a == b
    assert len(lab.inbox("scout")) == 1


@pytest.mark.parametrize("msg_type,sender,to,payload,match", [
    ("VERDICT", "librarian", "system", {"decision": "reject", "reason": "dull idea"}, "cannot send"),
    ("VERDICT", "skeptic", "system", {"decision": "park", "reason": "later maybe"}, "cannot take the decision"),
    ("IMPL_DONE", "scout", "gatekeeper", {"files": ["strategy/a.py"], "summary": "x"}, "cannot send"),
    ("GATE_RESULT", "builder", "librarian", {"gate": "G1", "passed": True, "metrics": {}}, "cannot send"),
    ("NEW_HYPOTHESIS", "scout", "builder", {"card": {}}, "cannot be addressed"),
    ("SHOUT", "scout", "system", {}, "unknown message type"),
    ("QUESTION", "scout", "human", {"q": "missing field"}, "invalid QUESTION payload"),
])
def test_message_rules(lab, msg_type, sender, to, payload, match):
    with pytest.raises(LabError, match=match):
        lab.send(msg_type, sender, to, None, payload)


def test_skeptic_no_objection_needs_the_full_checklist(lab, card):
    hid = submit(lab, card)
    checklist = yaml.safe_load(lab.paths.gates.read_text())["skeptic"]["checklist"]
    partial = {c: {"status": "ok", "evidence": "looked at it carefully"} for c in checklist[:-1]}
    with pytest.raises(LabError, match=checklist[-1]):
        lab.check_message("VERDICT", "skeptic", "system", hid,
                          {"decision": "no_objection", "reason": "all fine", "checklist": partial})
    concern = {**partial, checklist[-1]: {"status": "concern", "evidence": "regime dependence unclear"}}
    with pytest.raises(LabError, match="OBJECTION instead"):
        lab.check_message("VERDICT", "skeptic", "system", hid,
                          {"decision": "no_objection", "reason": "all fine", "checklist": concern})


def test_invalid_card_is_refused(lab, card):
    del card["mechanism"]
    with pytest.raises(LabError, match="mechanism"):
        lab.create_hypothesis(card, "scout")


def test_revision_cannot_change_family_or_happen_outside_idea(lab, card):
    hid = submit(lab, card)
    with pytest.raises(LabError, match="only be revised in IDEA"):
        lab.revise(hid, card, "scout", "tweak")
    lab.con.execute("UPDATE hypotheses SET status = 'IDEA' WHERE id = ?", (hid,))
    with pytest.raises(LabError, match="family"):
        lab.revise(hid, {**card, "family": "fresh-start"}, "scout", "new family resets N")


def test_card_file_mirrors_db(lab, card):
    hid = submit(lab, card)
    out = yaml.safe_load(lab.paths.card(hid).read_text())
    assert out["id"] == hid and out["status"] == "DATA_READY" and out["author_agent"] == "human"
    assert [h["to"] for h in out["history"]] == ["IDEA", "SPECIFIED", "DATA_READY"]
