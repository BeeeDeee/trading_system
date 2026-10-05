"""Librarian lessons: qualitative only, rendered for the Scout, one open case per decided hypothesis."""

import json

import pytest

from lab.framework import agents, invocations, lessons
from lab.framework.invocations import stage
from lab.framework.states import S
from lab.framework.tick import tick

from .conftest import Evaluator, submit

LESSON = {"family": "sector-momentum", "mechanism_class": "behavioral", "instruments": "SPDR sector ETFs",
          "outcome": "rejected at G1: did not beat the 60/40 benchmark",
          "lesson": "Twelve-month sector momentum did not add anything over a static stock/bond mix in this period.",
          "avoid": "Another lookback of the same ranking."}


def rejected(lab, card):
    hid = submit(lab, card)
    lab.transition(hid, S.REJECTED, "human", "test", reason_code="g1_sharpe_excess")
    return hid


def send(lab, hid, payload):
    inv = invocations.start(lab, "librarian", None, task="lessons")
    stage(inv.workspace, "LESSON", "system", hid, payload)
    assert invocations.finish(lab, inv.id) == "applied"
    tick(lab, None)


def test_lesson_is_rendered_and_the_case_closes(lab, card):
    hid = rejected(lab, card)
    assert [c["id"] for c in agents.cases(lab)] == [hid]
    send(lab, hid, LESSON)
    text = lessons.path(lab).read_text()
    assert hid in text and "static stock/bond mix" in text
    assert agents.cases(lab) == []
    inv = invocations.start(lab, "scout", None, task="propose")
    assert "static stock/bond mix" in (inv.workspace / "lessons.md").read_text()


@pytest.mark.parametrize("text", ["Sharpe was 0.73 against 0.37.", "It lost 31 % in 2018.",
                                  "Costs at x2 killed it.", "The bound was -0.06 only."])
def test_numbers_from_results_are_refused(lab, card, text):
    hid = rejected(lab, card)
    send(lab, hid, dict(LESSON, lesson=text + " More words to pass the minimum length."))
    assert not lessons.path(lab).exists()
    (q,) = [m for m in lab.inbox("librarian") if m["type"] == "QUESTION"]
    assert "qualitative" in json.loads(q["payload_json"])["question"]


def test_years_and_ids_are_fine_and_only_decided_hypotheses_get_lessons(lab, card):
    open_hid = submit(lab, card)
    send(lab, open_hid, LESSON)
    assert not lessons.path(lab).exists()          # still DATA_READY: refused
    hid = rejected(lab, dict(card, title="Another sector rotation idea for the test"))
    send(lab, hid, dict(LESSON, lesson="Like H-0001, it depended on the 2008 bond rally rather than on sectors."))
    assert "2008 bond rally" in lessons.path(lab).read_text()


def test_wrong_family_is_refused(lab, card):
    hid = rejected(lab, card)
    send(lab, hid, dict(LESSON, family="something-else"))
    assert not lessons.path(lab).exists()


def test_librarian_definition_and_workspace(lab, card):
    rejected(lab, card)
    s = agents.spec("librarian")
    assert s.lab_verbs == ("send", "inbox", "context") and "WebSearch" not in s.tools
    inv = invocations.start(lab, "librarian", None, task="lessons")
    cases = json.loads((inv.workspace / "cases.json").read_text())
    assert cases and "gate_results" in cases[0] and (inv.workspace / "lessons.md").exists()
