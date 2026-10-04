"""Read-only views for people: funnel, rejection reasons, the per-hypothesis trace, invocation log.
The dashboard (step 5) renders the same data."""

import json
from collections import Counter

from lab.framework.blackboard import Lab
from lab.framework.states import FUNNEL


def funnel(lab: Lab) -> dict[str, int]:
    counts = Counter(h["status"] for h in lab.hypotheses())
    return {s.value: counts.get(s.value, 0) for s in FUNNEL}


def rejection_reasons(lab: Lab) -> list[dict]:
    rows = lab.con.execute("SELECT status, reject_stage, reject_code, COUNT(*) n FROM hypotheses "
                           "WHERE reject_stage IS NOT NULL GROUP BY 1, 2, 3 ORDER BY n DESC")
    return [dict(r) for r in rows]


def trace(lab: Lab, hid: str) -> list[dict]:
    """Everything that happened to one hypothesis, in time order: the 'conversation' of the agents."""
    ev = []
    for r in lab.con.execute("SELECT * FROM transitions WHERE hypothesis_id = ?", (hid,)):
        ev.append({"ts": r["ts"], "kind": "transition", "actor": r["actor"],
                   "text": f"{r['from_status'] or '∅'} -> {r['to_status']} (v{r['version']}): {r['reason']}"})
    # Messages about this hypothesis, plus hypothesis-less ones that caused its transitions (its creation,
    # an ingest that unblocked it), and every agent run that wrote any of them.
    msgs = lab.con.execute(
        "SELECT * FROM messages WHERE hypothesis_id = ? OR id IN "
        "(SELECT message_id FROM transitions WHERE hypothesis_id = ? AND message_id IS NOT NULL)",
        (hid, hid)).fetchall()
    for r in msgs:
        ev.append({"ts": r["created_at"], "kind": "message", "actor": r["from_agent"],
                   "text": f"{r['type']} -> {r['to_agent']}: {_summary(r['type'], json.loads(r['payload_json']))}"
                           + ("" if r["handled_at"] else "  [unread]")})
    inv_ids = {r["invocation_id"] for r in msgs if r["invocation_id"]}
    for r in lab.con.execute("SELECT * FROM agent_invocations WHERE hypothesis_id = ? OR id IN (%s)"
                             % ",".join("?" * len(inv_ids)), (hid, *inv_ids)):
        ev.append({"ts": r["started_at"], "kind": "invocation", "actor": r["agent"],
                   "text": f"run {r['id']} ({r['model']}) -> {r['outcome'] or 'running'}"
                           + (f": {r['error']}" if r["error"] else "")})
    for r in lab.con.execute("SELECT * FROM gate_results WHERE hypothesis_id = ?", (hid,)):
        ev.append({"ts": r["ts"], "kind": "gate", "actor": "gatekeeper",
                   "text": f"{r['gate']} v{r['version']} {'PASS' if r['passed'] else 'FAIL'}"
                           f"{' ' + r['reason_code'] if r['reason_code'] else ''} [{r['evaluator']}]"})
    return sorted(ev, key=lambda e: e["ts"])


def _summary(msg_type: str, p: dict) -> str:
    match msg_type:
        case "NEW_HYPOTHESIS" | "REVISION":
            return p["card"].get("title", "") + (f" ({p['reason']})" if p.get("reason") else "")
        case "DATA_REQUEST" | "DATA_READY":
            return p["dataset"]
        case "IMPL_DONE":
            return f"{len(p['files'])} files: {p['summary']}"
        case "GATE_RESULT":
            return f"{p['gate']} {'pass' if p['passed'] else 'fail'} {p.get('reason_code') or ''}".strip()
        case "OBJECTION":
            return f"to {p['return_to']}: " + "; ".join(f"{i['check']}: {i['finding']}" for i in p["items"])
        case "VERDICT":
            return f"{p['decision']}: {p['reason']}"
        case "QUESTION":
            return p["question"]
        case "ALERT":
            return f"[{p['severity']}] {p['text']}"
    return json.dumps(p)[:120]


def invocations(lab: Lab, limit: int = 50) -> list[dict]:
    rows = lab.con.execute("SELECT id, agent, model, hypothesis_id, started_at, ended_at, outcome, error "
                           "FROM agent_invocations ORDER BY started_at DESC LIMIT ?", (limit,))
    return [dict(r) for r in rows]
