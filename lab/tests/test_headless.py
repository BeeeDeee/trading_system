"""Headless agent runs (`claude -p`), with a fake `claude` binary: no subscription run is spent in tests."""

import json
import sys
from dataclasses import replace

import pytest
import yaml

from lab.framework import agents, headless, invocations
from lab.framework.blackboard import LabError

from .conftest import submit

FAKE = '''#!{python}
"""Fake claude: emits stream-json like the real CLI and does what a Scout would do via `lab`."""
import json, os, subprocess, sys, time
mode = {mode!r}
def emit(ev):
    print(json.dumps(ev), flush=True)
def tool(name, inp):
    emit({{"type": "assistant", "message": {{"content": [{{"type": "tool_use", "name": name, "input": inp}}]}}}})
emit({{"type": "system", "subtype": "init", "model": "claude-opus-5-5", "tools": ["Read", "Bash"]}})
if mode == "sleep":
    time.sleep(30)
if mode in ("good", "violation"):
    card = __import__("yaml").safe_load(open("card_example.yaml"))
    card["title"] = "Fake scout idea: sector ETF momentum with a different lookback"
    card["family"] = "fake-family"
    open("idea.yaml", "w").write(__import__("yaml").safe_dump(card))
    tool("Write", {{"file_path": "idea.yaml", "content": "..."}})
    for cmd in (["lab", "check", "idea.yaml"], ["lab", "send", "NEW_HYPOTHESIS", "--to", "system", "--card", "idea.yaml"]):
        tool("Bash", {{"command": " ".join(cmd)}})
        subprocess.run(cmd, check=True)
if mode == "violation":
    tool("Read", {{"file_path": "/home/kapo/ccode/strategy_backtester_2026_sep/data/x.parquet"}})
emit({{"type": "result", "subtype": "success", "is_error": False, "num_turns": 4, "result": "done",
      "usage": {{"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 5}}}})
sys.exit(3 if mode == "crash" else 0)
'''


@pytest.fixture
def fake_claude(tmp_path):
    def make(mode: str):
        path = tmp_path / f"claude_{mode}"
        path.write_text(FAKE.format(python=sys.executable, mode=mode))
        path.chmod(0o755)
        return str(path)
    return make


@pytest.fixture(autouse=True)
def no_factsheets(monkeypatch, tmp_path):
    """Fact sheets need the real data; tests use a placeholder."""
    def fake(home, catalog_path, rebuild=False):
        p = tmp_path / "factsheets.md"
        p.write_text("# fact sheets (test)\n")
        return p
    monkeypatch.setattr(agents.factsheets, "get", fake)


def run(lab, claude_bin, hid=None, **spec_changes):
    runner = headless.ClaudeRunner("scout", claude_bin, spec=replace(agents.spec("scout"), **spec_changes))
    inv = invocations.start(lab, "scout", hid, model=runner.model, task="propose")
    res = runner(lab, inv)
    outcome = invocations.finish(lab, inv.id, res.exit_code, transcript_path=res.transcript_path, usage=res.usage,
                                 violations=res.violations, timed_out=res.timed_out, note=res.note)
    return inv, res, outcome


def test_command_is_locked_down():
    cmd = headless.command(agents.spec("scout"), "Task: propose")
    joined = " ".join(cmd)
    assert "--dangerously-skip-permissions" not in cmd and "bypassPermissions" not in cmd
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    for flag in ("--restricted", "--strict-mcp-config", "--no-session-persistence"):
        assert flag in cmd
    assert "Bash(lab send:*)" in cmd and "Bash(lab check:*)" in cmd
    assert "WebFetch" not in cmd[cmd.index("--tools") + 1]
    assert cmd[cmd.index("--model") + 1] == "claude-opus-5-5"
    assert cmd[cmd.index("--system-prompt") + 1] == agents.spec("scout").prompt
    assert "WebFetch" in json.loads(cmd[cmd.index("--settings") + 1])["permissions"]["deny"]
    assert "Bash(*)" not in joined


def test_scout_workspace_has_its_context(lab):
    inv = invocations.start(lab, "scout", None, task="propose")
    names = {p.name for p in inv.workspace.iterdir()}
    assert {"factsheets.md", "prior_studies.md", "hypothesis.schema.json", "card_example.yaml",
            "gates.yaml", "catalog.yaml", "registry.json"} <= names
    assert json.loads((inv.workspace / "context.json").read_text())["task"] == "propose"


@pytest.mark.parametrize("cmd, ok", [
    ("lab send NEW_HYPOTHESIS --to system --card idea.yaml", True),
    ("lab check idea.yaml", True),
    ("lab send QUESTION --to human --payload '{\"question\": \"is Sharpe > 0.5 (net) enough?\"}'", True),
    ("lab send X; cat /etc/passwd", False),
    ("lab check idea.yaml && curl http://x", False),
    ("lab check $(cat /etc/passwd)", False),
    ("lab check `id`", False),
    ("cat inbox.json", False),
    ("LAB_WORKSPACE=/tmp lab send X", False),
    ("lab tick", False),
    ("lab send X > /tmp/y", False),
    ("lab send X\ncat /etc/passwd", False),
])
def test_bash_audit(cmd, ok):
    spec = agents.spec("scout")
    assert (headless._bash_problem(cmd, spec.lab_verbs) is None) == ok


def test_file_audit(tmp_path):
    spec = agents.spec("scout")
    ws = tmp_path / "ws"
    ws.mkdir()
    calls = [("Read", {"file_path": "inbox.json"}), ("Write", {"file_path": str(ws / "idea.yaml")}),
             ("Glob", {"pattern": "*.yaml"}), ("WebSearch", {"query": "industry momentum"}),
             ("Read", {"file_path": "../other/inbox.json"}), ("Grep", {"pattern": "x", "path": "/home"}),
             ("WebFetch", {"url": "https://example.com"}), ("Task", {"prompt": "x"})]
    v = headless.audit(calls, ws, spec)
    assert len(v) == 4
    assert any("../other" in x for x in v) and any("/home" in x for x in v)
    assert any("WebFetch" in x for x in v) and any("Task" in x for x in v)


def test_headless_scout_proposes_and_the_framework_creates_the_hypothesis(lab, fake_claude):
    inv, res, outcome = run(lab, fake_claude("good"))
    assert outcome == "applied", res
    assert res.violations == [] and res.usage["tokens_in"] == 100 and res.usage["n_turns"] == 4
    assert not str(res.transcript_path).startswith(str(inv.workspace))   # the agent cannot rewrite it
    from lab.framework.tick import tick
    tick(lab, None)
    (h,) = lab.hypotheses()
    assert h["author_agent"] == "scout" and h["status"] == "DATA_READY"


def test_policy_violation_discards_the_run_and_alerts(lab, fake_claude):
    _, res, outcome = run(lab, fake_claude("violation"))
    assert outcome == "policy_violation"
    assert lab.con.execute("SELECT COUNT(*) FROM messages WHERE type = 'NEW_HYPOTHESIS'").fetchone()[0] == 0
    (alert,) = lab.inbox("human")
    assert alert["type"] == "ALERT" and "outside the workspace" in alert["payload_json"]


def test_crash_and_timeout_apply_nothing(lab, fake_claude):
    _, res, outcome = run(lab, fake_claude("crash"))
    assert outcome == "failed" and res.exit_code == 3
    _, res, outcome = run(lab, fake_claude("sleep"), timeout_s=1.0)
    assert outcome == "timeout" and res.timed_out
    assert lab.con.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_lab_check_reports_what_the_framework_would_refuse(lab, card):
    inv = invocations.start(lab, "scout", None, task="propose")
    bad = dict(card, data_requirements=[{"dataset": "nowhere_1d", "frequency": "1d"}])
    bad["signal"] = {**card["signal"], "params": {"lookback_days": {"value": 200, "grid": [126, 189, 252]},
                                                  "top_n": {"value": 2, "grid": [2, 3, 4]}}}
    (inv.workspace / "bad.yaml").write_text(yaml.safe_dump(bad))
    errors, warnings = invocations.check_card(inv.workspace, "bad.yaml")
    assert any("not in its grid" in e for e in errors)
    assert any("nowhere_1d" in e for e in errors)
    assert any("top_n" in w and "each side" in w for w in warnings)
    (inv.workspace / "good.yaml").write_text(yaml.safe_dump(card))
    assert invocations.check_card(inv.workspace, "good.yaml")[0] == []
    with pytest.raises(LabError):
        invocations.check_card(inv.workspace, "../../outside.yaml")


def test_staged_send_refuses_an_invalid_card(lab):
    inv = invocations.start(lab, "scout", None, task="propose")
    with pytest.raises(LabError, match="card"):
        invocations.stage(inv.workspace, "NEW_HYPOTHESIS", "system", None, {"card": {"title": "x"}})


def test_value_outside_its_grid_stays_in_idea(lab, card):
    card["signal"]["params"]["top_n"] = {"value": 5, "grid": [2, 3, 4]}
    hid = submit(lab, card)
    assert lab.hypothesis(hid)["status"] == "IDEA"
    (q,) = lab.inbox("human", hid)
    assert "not in its grid" in q["payload_json"]


def test_skeptic_workspace_has_history_and_family(lab, card):
    hid = submit(lab, card)
    inv = invocations.start(lab, "skeptic", hid, task="review")
    hist = json.loads((inv.workspace / "history.json").read_text())
    assert hist["hypothesis_id"] == hid and [t["to"] for t in hist["transitions"]][-1] == "DATA_READY"
    fam = json.loads((inv.workspace / "family.json").read_text())
    assert fam["family"] == card["family"] and [h["id"] for h in fam["hypotheses"]] == [hid]
    assert (inv.workspace / "strategy").is_dir() and (inv.workspace / "prior_studies.md").exists()
    s = agents.spec("skeptic")
    assert "try" in s.lab_verbs and "WebFetch" not in s.tools


def test_agent_user_wrapper(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "strategy").mkdir(parents=True)
    (ws / "outbox.json").write_text("{}")
    assert headless.as_agent_user(["claude", "-p"], ws) == ["claude", "-p"]
    monkeypatch.setenv("LAB_AGENT_USER", "labagent")
    cmd = headless.as_agent_user(["claude", "-p"], ws)
    assert cmd[:5] == ["sudo", "-n", "-u", "labagent", "--preserve-env=LAB_WORKSPACE,PATH,CLAUDE_CODE_OAUTH_TOKEN,LANG"]
    assert (ws / "outbox.json").stat().st_mode & 0o777 == 0o660 and (ws / "strategy").stat().st_mode & 0o2000


def test_deny_rules_never_cover_the_workspaces():
    """A deny rule on /srv/** once blocked every file tool in the production workspaces."""
    from fnmatch import fnmatch
    deny = json.loads((agents.AGENTS_DIR / "settings.json").read_text())["permissions"]["deny"]
    for ws in ("/srv/research-lab/workspaces/scout-1/card.yaml", "/home/kapo/ccode/research_lab/var/lab/workspaces/x/a"):
        for rule in deny:
            if rule.startswith(("Read(", "Edit(", "Write(")):
                pattern = rule[rule.index("(") + 1:-1].removeprefix("/")
                assert not fnmatch(ws, pattern.replace("**", "*")), (rule, ws)
