"""Orchestrator: picks pending work, runs one agent at a time, applies the result. Deterministic.

One cycle = tick (deterministic work) -> pick the next agent task -> start invocation -> run the agent
-> finish (apply outbox) -> tick. Who runs the agent is pluggable: stub functions (tests, demo) or
`headless.ClaudeRunner` (`claude -p` per lab/agents/agents.yaml). Budget and the systemd timer come in step 4.
"""

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


def pending_tasks(lab: Lab) -> list[Task]:
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
    if lab.inbox("librarian"):
        tasks.append(Task("librarian", None, "record results and lessons"))
    return tasks


def run_cycle(lab: Lab, evaluator: Evaluator | None, runners: dict[str, Runner]) -> Task | None:
    """One cycle. Returns the task that ran, or None if there was nothing to do."""
    tick.tick(lab, evaluator)
    for task in pending_tasks(lab):
        if task.agent not in runners:
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
        tick.tick(lab, evaluator)
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
