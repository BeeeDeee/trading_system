"""Knowledge base: append-only entries, supersession, qualitative market entries, rendered for agents."""

from lab.framework import invocations, knowledge
from lab.framework.invocations import stage
from lab.framework.tick import tick

ENTRY = {"kind": "data", "title": "SFP adjusted-close spikes", "statement": "191 one-day spikes in 67 funds; the "
         "loader drops them.", "evidence": ["decision log 2026-10-05"], "confidence": "high"}


def send(lab, payload, agent="librarian"):
    inv = invocations.start(lab, agent, None, task="t")
    stage(inv.workspace, "KNOWLEDGE", "system", None, payload)
    assert invocations.finish(lab, inv.id) == "applied"
    tick(lab, None)


def test_entries_render_and_supersede(lab):
    send(lab, ENTRY)
    send(lab, dict(ENTRY, statement="Corrected: spikes are split-adjustment errors; the loader drops them.",
                   supersedes="K-0001"), agent="chair")
    rows = knowledge.entries(lab)
    assert [(r["id"], r["current"]) for r in rows] == [("K-0001", False), ("K-0002", True)]
    text = knowledge.path(lab).read_text()
    assert "Corrected" in text and "## Superseded" in text
    inv = invocations.start(lab, "scout", None, task="propose")
    assert "Corrected" in (inv.workspace / "knowledge.md").read_text()


def test_market_entries_are_qualitative(lab):
    send(lab, dict(ENTRY, kind="market", statement="Turn-of-month timing earned 0.38 Sharpe in dev."))
    assert knowledge.entries(lab) == []
    send(lab, dict(ENTRY, kind="market", title="Binary stock/bond switching loses to a constant mix",
                   statement="Whole-portfolio switches between SPY and IEF lost to 60/40 on risk-adjusted terms."))
    assert len(knowledge.entries(lab)) == 1
