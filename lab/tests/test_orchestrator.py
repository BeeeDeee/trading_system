"""Production cycle: budget, quiet window, pauses, rate limits, one process at a time."""

import fcntl
import json
from datetime import datetime, timedelta, timezone

import yaml

from lab import stubs
from lab.framework.headless import RunResult
from lab.orchestrator import Policy, gate, pause_until, production_cycle

from .conftest import Evaluator, submit

NOON = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)   # budget days are real dates


def policy(tmp_path, **over):
    c = yaml.safe_load(open("lab/budget.yaml"))
    c.update(over)
    p = tmp_path / "budget.yaml"
    p.write_text(yaml.safe_dump(c))
    return p


def test_budget_quiet_window_and_pause(lab, tmp_path):
    pol = Policy.load(policy(tmp_path))
    allow, why = gate(lab, pol, NOON)
    assert why is None and allow("scout") and not allow("nobody")
    assert gate(lab, pol, NOON.replace(hour=1))[1].startswith("quiet window")
    (lab.paths.home / "PAUSE").write_text("owner")
    assert gate(lab, pol, NOON)[1] == "PAUSE file present"


def test_cycle_runs_one_agent_within_budget(lab, card, tmp_path):
    submit(lab, card)                                   # DATA_READY: the builder has work
    runners = {"builder": stubs.builder}
    rec = production_cycle(lab, policy(tmp_path), at=NOON, runners=runners, evaluator=Evaluator())
    assert [a["agent"] for a in rec["agents"]] == ["builder"]
    rec = production_cycle(lab, policy(tmp_path, daily={"builder": 1}, total_daily=8), at=NOON, runners=runners,
                           evaluator=Evaluator())
    assert rec["agents"] == []                          # builder's daily budget is used
    log = [json.loads(x) for x in (lab.paths.home / "cycles.jsonl").read_text().splitlines()]
    assert len(log) == 2


def test_rate_limit_pauses_agents(lab, card, tmp_path):
    submit(lab, card)
    rec = production_cycle(lab, policy(tmp_path), at=NOON, evaluator=Evaluator(),
                           runners={"builder": lambda lab, inv: RunResult(1, note="API Error: 429 rate limit")})
    assert rec["agents"][0]["outcome"] == "failed"
    assert (lab.paths.home / "RATE_LIMITED_UNTIL").exists()
    assert "rate limited" in gate(lab, Policy.load(policy(tmp_path)), NOON)[1]


def test_subscription_session_limit_pauses_agents(lab, card, tmp_path):
    submit(lab, card)
    msg = "success: You've hit your session limit \u00b7 resets 7am (UTC)"
    rec = production_cycle(lab, policy(tmp_path), at=NOON, evaluator=Evaluator(),
                           runners={"builder": lambda lab, inv: RunResult(1, note=msg)})
    assert rec["agents"][0]["outcome"] == "failed"
    assert (lab.paths.home / "RATE_LIMITED_UNTIL").exists()


def test_pause_ends_at_the_reset_time_the_message_names():
    msg = "success: You've hit your session limit · resets 7am (UTC)"
    at = NOON.replace(hour=6, minute=35)
    assert pause_until(msg, at, 5) == NOON.replace(hour=7)
    assert pause_until(msg, NOON, 5) == NOON + timedelta(hours=5)          # next 7am is 19 h away: capped
    assert pause_until("resets 9:30pm (UTC)", NOON.replace(hour=19), 5) == NOON.replace(hour=21, minute=30)
    assert pause_until("429 rate limit", at, 5) == at + timedelta(hours=5)


def test_scout_proposes_when_the_pipeline_is_empty(lab, tmp_path):
    rec = production_cycle(lab, policy(tmp_path), at=NOON, dry_run=True, evaluator=Evaluator())
    assert rec["would_run"][0]["agent"] == "scout" and "Brief" in rec["would_run"][0]["why"]


def test_one_cycle_at_a_time(lab, tmp_path):
    lab.paths.home.mkdir(parents=True, exist_ok=True)
    with open(lab.paths.home / "cycle.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        rec = production_cycle(lab, policy(tmp_path), at=NOON, evaluator=Evaluator())
    assert rec["skipped"] == "another cycle is running"


def test_infrastructure_failures_pause_and_cost_no_budget(lab, card, tmp_path):
    from lab.orchestrator import used_today
    submit(lab, card)
    broken = lambda lab, inv: RunResult(1, note="sudo: no new privileges flag is set")  # noqa: E731
    for _ in range(3):
        production_cycle(lab, policy(tmp_path, daily={"builder": 5, "scout": 0}, total_daily=8), at=NOON,
                         evaluator=Evaluator(), runners={"builder": broken})
    assert (lab.paths.home / "PAUSE").exists()
    assert any("paused" in m["payload_json"] for m in lab.inbox("human"))
    assert used_today(lab, NOON.date().isoformat()) == {}


def test_an_unanswered_owner_question_stops_the_agent_loop(lab, card, tmp_path):
    from lab.orchestrator import pending_tasks
    hid = submit(lab, card)
    assert [t.agent for t in pending_tasks(lab)] == ["builder"]
    q = lab.send("QUESTION", "builder", "human", hid, {"question": "cannot implement this as written, please decide"})
    assert pending_tasks(lab) == []                       # the Builder is not started again and again
    lab.mark_handled([q], "human")
    assert [t.agent for t in pending_tasks(lab)] == ["builder"]
