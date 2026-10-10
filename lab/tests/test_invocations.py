"""Agent runs: workspace in, outbox out, applied all-or-nothing with the role recorded in lab.db."""

import json
import os

import pytest

from lab import stubs
from lab.framework import cli, invocations
from lab.framework.blackboard import LabError
from lab.framework.invocations import stage

from .conftest import submit


def test_workspace_contains_what_the_agent_needs_and_nothing_else(lab, card):
    hid = submit(lab, card)
    inv = invocations.start(lab, "builder", hid)
    names = {p.name for p in inv.workspace.iterdir()}
    assert names == {"context.json", "inbox.json", "outbox.json", "registry.json", "catalog.yaml", "card.yaml",
                     "gate_results.json", "strategy", "gates.yaml", "knowledge.md"}
    assert not any(p.suffix == ".db" for p in inv.workspace.rglob("*"))


def test_outbox_is_applied_and_files_copied(lab, card):
    hid = submit(lab, card)
    inv = invocations.start(lab, "builder", hid)
    assert stubs.builder(lab, inv) == 0
    assert invocations.finish(lab, inv.id) == "applied"
    assert (lab.paths.strategies / hid / "strategy.py").exists()
    msgs = lab.con.execute("SELECT * FROM messages WHERE invocation_id = ?", (inv.id,)).fetchall()
    assert [(m["type"], m["from_agent"]) for m in msgs] == [("IMPL_DONE", "builder")]


def test_finish_twice_is_a_no_op(lab, card):
    hid = submit(lab, card)
    inv = invocations.start(lab, "builder", hid)
    stubs.builder(lab, inv)
    invocations.finish(lab, inv.id)
    invocations.finish(lab, inv.id)
    assert lab.con.execute("SELECT COUNT(*) FROM messages WHERE type = 'IMPL_DONE'").fetchone()[0] == 1


def test_role_comes_from_the_db_not_the_workspace(lab, card):
    """A builder that rewrites its context and outbox to act as the skeptic gets nothing applied."""
    hid = submit(lab, card)
    inv = invocations.start(lab, "builder", hid)
    (inv.workspace / "context.json").write_text(json.dumps({"agent": "skeptic", "invocation_id": inv.id}))
    stage(inv.workspace, "VERDICT", "system", hid, {"decision": "reject", "reason": "forged by the builder"})
    assert invocations.finish(lab, inv.id) == "outbox_rejected"
    row = lab.con.execute("SELECT error FROM agent_invocations WHERE id = ?", (inv.id,)).fetchone()
    assert "builder cannot send VERDICT" in row["error"]
    assert lab.hypothesis(hid)["status"] == "DATA_READY"


def test_rejected_outbox_applies_nothing_and_keeps_the_inbox(lab, card):
    hid = submit(lab, card)
    lab.send("QUESTION", "human", "builder", hid, {"question": "use the total return index?"})
    inv = invocations.start(lab, "builder", hid)
    stubs.builder(lab, inv)
    box = json.loads((inv.workspace / "outbox.json").read_text())
    box["messages"].append({"type": "QUESTION", "to": "human", "hypothesis_id": "H-0999", "payload": {"question": "x?"}})
    (inv.workspace / "outbox.json").write_text(json.dumps(box))
    assert invocations.finish(lab, inv.id) == "outbox_rejected"
    assert not (lab.paths.strategies / hid).exists()
    assert len(lab.inbox("builder", hid)) == 1


def test_files_outside_strategy_dir_are_refused(lab, card):
    hid = submit(lab, card)
    inv = invocations.start(lab, "builder", hid)
    stubs.builder(lab, inv)
    box = json.loads((inv.workspace / "outbox.json").read_text())
    box["messages"][0]["payload"]["files"] = ["strategy/../context.json"]
    (inv.workspace / "outbox.json").write_text(json.dumps(box))
    assert invocations.finish(lab, inv.id) == "outbox_rejected"


def test_failed_run_applies_nothing(lab, card):
    hid = submit(lab, card)
    inv = invocations.start(lab, "builder", hid)
    stubs.builder(lab, inv)
    assert invocations.finish(lab, inv.id, exit_code=1) == "failed"
    assert lab.con.execute("SELECT COUNT(*) FROM messages WHERE type = 'IMPL_DONE'").fetchone()[0] == 0


def test_a_hypothesis_has_one_agent_at_a_time(lab, card):
    hid = submit(lab, card)
    inv = invocations.start(lab, "builder", hid)
    with pytest.raises(LabError, match="leased"):
        invocations.start(lab, "skeptic", hid)
    invocations.finish(lab, inv.id)
    invocations.finish(lab, invocations.start(lab, "skeptic", hid).id)


def test_messages_about_other_hypotheses_are_refused(lab, card, blocked_card):
    hid = submit(lab, card)
    other = submit(lab, blocked_card)
    inv = invocations.start(lab, "builder", hid)
    stage(inv.workspace, "QUESTION", "human", other, {"question": "can I peek at the other one?"})
    assert invocations.finish(lab, inv.id) == "outbox_rejected"


def test_staged_cli_only_offers_send_inbox_context(lab, card, monkeypatch, capsys):
    hid = submit(lab, card)
    inv = invocations.start(lab, "builder", hid)
    monkeypatch.setenv("LAB_WORKSPACE", str(inv.workspace))
    assert cli.main(["tick"]) == 2
    assert cli.main(["move", hid, "GATE_1", "--reason", "trust me"]) == 2
    assert "not available inside an agent run" in capsys.readouterr().err
    assert cli.main(["send", "QUESTION", "--to", "human", "--hyp", hid, "--payload",
                     '{"question": "which cost tier for XLC?"}']) == 0
    assert cli.main(["send", "VERDICT", "--to", "system", "--hyp", hid, "--payload",
                     '{"decision": "reject", "reason": "nope nope"}']) == 2
    monkeypatch.delenv("LAB_WORKSPACE")
    assert invocations.finish(lab, inv.id) == "applied"
    assert lab.inbox("human", hid)[0]["type"] == "QUESTION"
    assert "LAB_WORKSPACE" not in os.environ


@pytest.mark.parametrize("argv", [["status"], ["list"], ["inbox"], ["invocations"], ["catalog"], ["usage"], ["knowledge", "list"]])
def test_direct_commands_run(lab, card, monkeypatch, argv, capsys):
    """Every read-only direct command works on a populated lab (a local import once shadowed `report`)."""
    from lab.framework import paths
    submit(lab, card)
    monkeypatch.setattr(cli, "default_paths", lambda: lab.paths)
    assert cli.main(argv) == 0
