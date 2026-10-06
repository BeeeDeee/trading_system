"""Orchestrator: picks pending work, runs one agent at a time, applies the result. Deterministic.

One cycle = tick (deterministic work) -> pick the next agent task -> start invocation -> run the agent
-> finish (apply outbox) -> tick. Who runs the agent is pluggable: stub functions (tests, demo) or
`headless.ClaudeRunner` (`claude -p` per lab/agents/agents.yaml).

Production (`lab cycle`, systemd timer every 30 min): one process at a time (file lock), the deterministic
phase always runs, agent runs only inside the budget (`lab/budget.yaml`: per agent and total per UTC day,
a quiet window, a pause after a rate-limit error, a PAUSE file the owner can create).
"""

import fcntl
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from collections.abc import Callable
from dataclasses import dataclass

from lab.framework import invocations, tick
from lab.framework.blackboard import Lab
from lab.framework.db import now
from lab.framework.gates import Evaluator
from lab.framework.headless import RunResult
from lab.framework.invocations import Invocation
from lab.framework.states import S

# runner(lab, invocation) -> exit code (stubs) or RunResult (headless claude); the agent's effects are only
# the files in its workspace and, for headless runs, a transcript that finish() audits
Runner = Callable[[Lab, Invocation], "int | RunResult"]


@dataclass(frozen=True)
class Task:
    agent: str
    hypothesis_id: str | None
    why: str


def pending_tasks(lab: Lab, propose: str | None = None) -> list[Task]:
    """Agent work in priority order: unblock data, finish late stages, then early ones, then housekeeping.
    Chair's priorities will reorder hypotheses inside each rule (step 3)."""
    locked = {r[0] for r in lab.con.execute("SELECT hypothesis_id FROM locks WHERE lease_until > ?", (now(),))}
    free = lambda status: [h["id"] for h in lab.hypotheses(status) if h["id"] not in locked]  # noqa: E731
    tasks = []
    if lab.inbox("archivist"):
        tasks.append(Task("archivist", None, "DATA_REQUEST in inbox"))
    tasks += [Task("skeptic", hid, "review before the holdout") for hid in free(S.SKEPTIC_REVIEW)]
    tasks += [Task("builder", hid, "fix" if lab.inbox("builder", hid) else "implement")
              for hid in free(S.DATA_READY)]
    tasks += [Task("scout", hid, "answer") for hid in free(S.IDEA) if lab.inbox("scout", hid)]
    if lab.inbox("chair"):
        tasks.append(Task("chair", None, "backlog"))
    if lab.inbox("librarian"):
        tasks.append(Task("librarian", None, "record results and lessons"))
    if steward_due(lab):
        tasks.append(Task("steward", None, "weekly paper report"))
    if propose:
        tasks.append(Task("scout", None, propose))
    return tasks


def steward_due(lab: Lab, days: int = 7) -> bool:
    """A weekly Steward report while something is paper trading (or an ALERT about paper waits for it)."""
    if not lab.hypotheses(S.PAPER):
        return False
    last = lab.con.execute("SELECT MAX(started_at) FROM agent_invocations WHERE agent = 'steward'").fetchone()[0]
    if last is None:
        return True
    from datetime import datetime, timedelta, timezone
    return datetime.fromisoformat(last) < datetime.now(timezone.utc) - timedelta(days=days)


def run_cycle(lab: Lab, evaluator: Evaluator | None, runners: dict[str, Runner],
              allow: Callable[[str], bool] | None = None, propose: str | None = None,
              tick_fn: Callable[[], list[str]] | None = None) -> Task | None:
    """One cycle. Returns the task that ran, or None if there was nothing to do (or nothing allowed)."""
    tick_fn = tick_fn or (lambda: tick.tick(lab, evaluator))
    tick_fn()
    for task in pending_tasks(lab, propose):
        if task.agent not in runners or (allow and not allow(task.agent)):
            continue
        inv = invocations.start(lab, task.agent, task.hypothesis_id, model=getattr(
            runners[task.agent], "model", "stub"), task=task.why)
        try:
            res = runners[task.agent](lab, inv)
        except Exception as e:
            res = RunResult(1, note=f"runner crashed: {type(e).__name__}: {e}")
        if isinstance(res, int):   # stub agents return an exit code
            res = RunResult(res)
        invocations.finish(lab, inv.id, res.exit_code, transcript_path=res.transcript_path, usage=res.usage,
                           violations=res.violations, timed_out=res.timed_out, note=res.note)
        tick_fn()
        return task
    return None


def run_until_idle(lab: Lab, evaluator: Evaluator | None, runners: dict[str, Runner],
                   max_cycles: int = 100) -> list[Task]:
    done = []
    for _ in range(max_cycles):
        task = run_cycle(lab, evaluator, runners)
        if task is None:
            break
        done.append(task)
    return done


# ---------------------------------------------------------------------------- production cycle

@dataclass(frozen=True)
class Policy:
    daily: dict
    total_daily: int
    quiet_utc: tuple[int, int]
    max_agent_runs_per_cycle: int
    propose_when_waiting_below: int
    rate_limit_pause_hours: float

    @classmethod
    def load(cls, path: Path) -> "Policy":
        c = yaml.safe_load(path.read_text())
        return cls(c["daily"], int(c["total_daily"]), tuple(c["quiet_utc"]), int(c["max_agent_runs_per_cycle"]),
                   int(c["propose_when_waiting_below"]), float(c["rate_limit_pause_hours"]))


def used_today(lab: Lab, day: str) -> dict[str, int]:
    rows = lab.con.execute("SELECT agent, COUNT(*) FROM agent_invocations WHERE substr(started_at, 1, 10) = ? "
                           "AND COALESCE(error, '') != 'dry run' GROUP BY agent", (day,))
    return {a: n for a, n in rows}


def gate(lab: Lab, policy: Policy, at: datetime) -> tuple[Callable[[str], bool], str | None]:
    """(may agent X run now?, why nothing may run) from the budget, the quiet window and the pauses."""
    home = lab.paths.home
    if (home / "PAUSE").exists():
        return (lambda a: False), "PAUSE file present"
    paused = home / "RATE_LIMITED_UNTIL"
    if paused.exists() and datetime.fromisoformat(paused.read_text().strip()) > at:
        return (lambda a: False), f"rate limited until {paused.read_text().strip()}"
    lo, hi = policy.quiet_utc
    if lo <= at.hour < hi:
        return (lambda a: False), f"quiet window {lo:02d}-{hi:02d} UTC"
    used = used_today(lab, at.date().isoformat())
    if sum(used.values()) >= policy.total_daily:
        return (lambda a: False), f"daily total {policy.total_daily} used"
    return (lambda a: used.get(a, 0) < int(policy.daily.get(a, 0))), None


def propose_brief(lab: Lab, policy: Policy) -> str | None:
    """A Scout `propose` task when the pipeline has room (few hypotheses waiting for the Builder)."""
    from lab.framework import briefs
    if len(lab.hypotheses(S.DATA_READY)) + len(lab.hypotheses(S.IDEA)) >= policy.propose_when_waiting_below:
        return None
    return briefs.text(briefs.choose(lab))


RATE_LIMIT_WORDS = ("rate limit", "rate_limit", "usage limit", "429", "overloaded", "quota")


def production_cycle(lab: Lab, policy_path: Path, *, agents: bool = True, dry_run: bool = False,
                     at: datetime | None = None, runners: dict[str, Runner] | None = None,
                     evaluator: Evaluator | None = None) -> dict:
    """One timer cycle. Returns a record that is also appended to LAB_HOME/cycles.jsonl."""
    from lab.framework import canaries
    at = at or datetime.now(timezone.utc)
    home = lab.paths.home
    home.mkdir(parents=True, exist_ok=True)
    rec = {"ts": at.isoformat(timespec="seconds"), "dry_run": dry_run}
    with open(home / "cycle.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return rec | {"skipped": "another cycle is running"}
        if evaluator is None:
            if canaries.passed_for_current(lab):
                from lab.framework.evaluator import QlabEvaluator
                evaluator = QlabEvaluator()
            else:
                rec["warning"] = "canaries have not passed for the current framework: no gates, no ingest"
        policy = Policy.load(policy_path)
        allow, blocked = gate(lab, policy, at)
        propose = propose_brief(lab, policy)
        rec["blocked"] = blocked
        if dry_run:
            tasks = [t for t in pending_tasks(lab, propose) if allow(t.agent)]
            return rec | {"would_run": [t.__dict__ for t in tasks[:policy.max_agent_runs_per_cycle]]}
        def tick_fn() -> list[str]:
            """Ingest in this process (network); the judge phase in the sandbox in production."""
            out = tick.tick(lab, evaluator, phase="ingest")
            if os.environ.get("LAB_SANDBOX") == "1" and evaluator is not None:
                from lab.framework import sandbox
                code, text = sandbox.judge(lab.paths)
                out += text.splitlines() + ([f"sandboxed judge exited {code}"] if code else [])
            else:
                out += tick.tick(lab, evaluator, phase="judge")
            return out

        rec["tick"] = tick_fn()
        ran = []
        if agents and blocked is None:
            if runners is None:
                from lab.framework.headless import ClaudeRunner
                runners = {}
                for a in policy.daily:
                    try:
                        runners[a] = ClaudeRunner(a)
                    except KeyError:
                        pass          # no headless definition yet (steward before step 4c)
            for _ in range(policy.max_agent_runs_per_cycle):
                allow, blocked = gate(lab, policy, datetime.now(timezone.utc) if at is None else at)
                if blocked:
                    break
                task = run_cycle(lab, evaluator, runners, allow, propose_brief(lab, policy), tick_fn)
                if task is None:
                    break
                row = lab.con.execute("SELECT id, outcome, error FROM agent_invocations ORDER BY started_at DESC "
                                      "LIMIT 1").fetchone()
                ran.append({"agent": task.agent, "hypothesis": task.hypothesis_id, "invocation": row["id"],
                            "outcome": row["outcome"], "error": row["error"]})
                if row["outcome"] == "failed" and any(w in (row["error"] or "").lower() for w in RATE_LIMIT_WORDS):
                    until = at + timedelta(hours=policy.rate_limit_pause_hours)
                    (home / "RATE_LIMITED_UNTIL").write_text(until.isoformat(timespec="seconds"))
                    lab.send("ALERT", "system", "human", None, {"severity": "warning",
                             "text": f"rate limit hit by {task.agent}; agent runs paused until {until:%Y-%m-%d %H:%M} UTC"})
                    break
        rec["agents"] = ran
    with open(home / "cycles.jsonl", "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec
